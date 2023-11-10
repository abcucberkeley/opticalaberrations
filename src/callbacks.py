import sys
import numpy as np
import logging

from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import TensorBoard

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

        if isinstance(lr_schedule, LearningRateSchedule):
            logs['learning_rate'] = lr_schedule(self.model.optimizer.iterations)
        elif lr_schedule is not None:
            logs['learning_rate'] = lr_schedule

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
        logger.info('\n'.join("%s: %s" % item for item in attrs.items()))


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
