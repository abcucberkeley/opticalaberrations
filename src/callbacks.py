import sys
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras import backend

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TensorBoardCallback(TensorBoard):
    def __init__(self, **kwargs):
        super(TensorBoardCallback, self).__init__(**kwargs)

    def _collect_learning_rate(self, logs):
        lr_schedule = getattr(self.model.optimizer, 'learning_rate', 0.)
        wd_schedule = getattr(self.model.optimizer, 'weight_decay', 1.)

        if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
            logs['learning_rate'] = lr_schedule(self.model.optimizer.iterations)
        elif lr_schedule is not None:
            logs['learning_rate'] = lr_schedule

        if isinstance(wd_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
            logs['weight_decay'] = wd_schedule(self.model.optimizer.iterations)
        elif wd_schedule is not None:
            logs['weight_decay'] = wd_schedule

        return logs


class LRLogger(Callback):
    def __init__(self):
        super(LRLogger, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        attrs = vars(self.model.optimizer)

        lr_schedule = getattr(self.model.optimizer, 'learning_rate', 0.)
        if isinstance(lr_schedule, LearningRateSchedule):
            lr_schedule = lr_schedule(self.model.optimizer.iterations)

        logger.info(lr_schedule)
        # logger.info('\n'.join("%s: %s" % item for item in attrs.items()))


class Defibrillator(Callback):
    def __init__(
        self,
        monitor: str = 'loss',
        mode: str = 'auto',
        min_delta: int = 0,
        patience: int = 0,
        verbose: int = 0,
    ):
        super(Defibrillator, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            logger.warning(f'Defibrillator mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if (self.monitor.endswith('acc') or self.monitor.endswith('accuracy') or
                    self.monitor.endswith('auc')):
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logger.warning(
                f'Defibrillator conditioned on metric `{self.monitor}` '
                f'which is not available. Available metrics are: {",".join(list(logs.keys()))}'
            )
        return monitor_value

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value - self.min_delta, reference_value)

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_epoch = 0
        self.best_weights = None
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.best_weights is None:
            self.best_weights = self.model.get_weights()

        if self._is_improvement(current, self.best):
            self.wait = 0
            self.best = current
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1

        if self.wait >= self.patience and epoch > 0:
            if self.best_weights is not None:

                self.model.set_weights(self.best_weights)

                if self.verbose > 0:
                    logger.info(f'Restoring model weights from epoch: {self.best_epoch + 1}')


class LearningRateScheduler(Callback):
    def __init__(
        self,
        initial_learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        decay_period: int = 1,
        warmup_epochs: int = 0,
        decay_multiplier: float = 2.0,
        decay: float = 1.0,
        alpha: float = 0.0,
        verbose: int = 0,
        fixed: bool = False,
    ):
        """
        Args:
            decay_period: Number of epochs to decay over before restarting LR
            warmup_epochs: Number of epochs for the initial linear warmup
            decay_multiplier: Used to derive the number of iterations in the i-th period
            decay: Used to derive the learning rate of the i-th period
            alpha: Minimum learning rate value as a fraction of the initial learning rate
        """
        super(LearningRateScheduler, self).__init__()

        dtype = tf.float32
        self.initial_learning_rate = tf.cast(initial_learning_rate, dtype)
        self.weight_decay = tf.cast(weight_decay, dtype)
        self.decay_period = tf.cast(decay_period, dtype)
        self.warmup_epochs = tf.cast(warmup_epochs, dtype)
        self.decay_multiplier = tf.cast(decay_multiplier, dtype)
        self.decay = tf.cast(decay, dtype)
        self.alpha = tf.cast(alpha, dtype)
        self.verbose = verbose
        self.fixed = fixed

    def on_epoch_begin(self, epoch, logs=None):
        try:
            lr = backend.get_value(self.model.optimizer.lr)
        except AttributeError:
            raise ValueError('Optimizer must have a `learning_rate`')

        if not self.fixed:
            lr = tf.cond(
                epoch < self.warmup_epochs,
                lambda: self.linear_warmup(
                    val=self.initial_learning_rate,
                    step=epoch,
                ),
                lambda: self.cosine_decay(
                    val=self.initial_learning_rate,
                    step=epoch - self.warmup_epochs,
                )
            )
            backend.set_value(self.model.optimizer.lr, backend.get_value(lr))

        if self.verbose > 0:
            logger.info(f'Scheduler setting learning rate: {lr}')

        tf.summary.scalar('learning rate', data=lr, step=epoch)

        if hasattr(self.model.optimizer, 'weight_decay'):
            backend.set_value(self.model.optimizer.weight_decay, backend.get_value(self.weight_decay))

            if self.verbose > 0:
                logger.info(f'Scheduler setting weight decay: {backend.get_value(self.model.optimizer.weight_decay)}')

            tf.summary.scalar('weight decay', data=self.weight_decay, step=epoch)

    def linear_warmup(self, val, step, power=1.0):
        completed_fraction = step / self.warmup_epochs
        decay = tf.math.pow(completed_fraction, power)
        return tf.multiply(val, decay)

    def cosine_decay(self, val, step):
        completed_fraction = step / self.decay_period

        def compute_step(completed_fraction, geometric=False):
            if geometric:
                i_restart = tf.floor(
                    tf.math.log(
                        1.0 - completed_fraction * (1.0 - self.decay_multiplier)
                    ) / tf.math.log(self.decay_multiplier)
                )

                sum_r = (1.0 - self.decay_multiplier ** i_restart) / (1.0 - self.decay_multiplier)
                completed_fraction = (completed_fraction - sum_r) / self.decay_multiplier ** i_restart
            else:
                i_restart = tf.floor(completed_fraction)
                completed_fraction -= i_restart

            return i_restart, completed_fraction

        i_restart, completed_fraction = tf.cond(
            tf.equal(self.decay_multiplier, 1.0),
            lambda: compute_step(completed_fraction, geometric=False),
            lambda: compute_step(completed_fraction, geometric=True)
        )

        m_fac = self.decay ** i_restart
        cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(
            tf.constant(np.pi) * completed_fraction))
        decay = (1 - self.alpha) * cosine_decayed + self.alpha

        return tf.multiply(val, decay)