import logging
import sys
from abc import ABC


import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
from scipy.special import binom

from base import Base
from roi import ROI

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Stem(layers.Layer):
    def __init__(
            self,
            kernel_size=7,
            activation='gelu',
            no_phase=False,
            mul=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.activation = activation
        self.no_phase = no_phase
        self.mul = mul

        self.conv = layers.Conv3D(
            filters=1,
            kernel_size=(1, self.kernel_size, self.kernel_size),
            padding='same'
        )
        self.act = layers.Activation(self.activation)

    def build(self, input_shape):
        super(Stem, self).build(input_shape)

    def get_config(self):
        config = super(Stem, self).get_config()
        config.update({
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "mul": self.mul,
            "no_phase": self.no_phase,
        })
        return config

    def call(self, inputs, training=True, **kwargs):

        if self.mul:
            emb = layers.multiply([inputs[:, :3], inputs[:, 3:]])
            emb = self.conv(emb)
            emb = self.act(emb)
            return emb
        else:
            if self.no_phase:
                alpha = self.conv(inputs[:, :3])
                alpha = self.act(alpha)
                return alpha
            else:  # two independent kernels for alpha and phi
                alpha = self.conv(inputs[:, :3])
                alpha = self.act(alpha)

                phi = self.conv(inputs[:, 3:])
                phi = self.act(phi)
                return layers.concatenate([alpha, phi], axis=1)


class Patchify(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.ln = layers.LayerNormalization(axis=-1, epsilon=1e-6)

    def build(self, input_shape):
        super(Patchify, self).build(input_shape)

    def get_config(self):
        config = super(Patchify, self).get_config()
        config.update({
            "patch_size": self.patch_size,
        })
        return config

    def call(self, inputs, training=True, **kwargs):
        planes = []
        for ax in range(inputs.shape[1]):
            i = tf.nn.space_to_depth(inputs[:, ax], self.patch_size)
            planes.append(i)
        patches = tf.stack(planes, axis=1)

        patches = self.ln(patches)
        patches = layers.Reshape((inputs.shape[1], -1, patches.shape[-1]))(patches)
        return patches


class Merge(layers.Layer):
    def __init__(self, patch_size, expansion=1, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.expansion = expansion

    def build(self, input_shape):
        super(Merge, self).build(input_shape)

    def get_config(self):
        config = super(Merge, self).get_config()
        config.update({
            "patch_size": self.patch_size,
            "expansion": self.expansion,
        })
        return config

    def call(self, inputs, training=True, **kwargs):
        planes = []
        d, n, p = inputs.shape[1], inputs.shape[-2], inputs.shape[-1]//self.expansion
        s = int(np.sqrt(n))

        img = layers.Reshape((d, s, s, -1))(inputs)

        for ax in range(d):
            i = tf.nn.depth_to_space(img[:, ax], self.patch_size)
            planes.append(i)

        return tf.stack(planes, axis=1)


class PatchEncoder(layers.Layer):
    def __init__(
            self,
            num_patches,
            embedding_size,
            positional_encoding_scheme='default',
            radial_encoding_periods=1,
            radial_encoding_nth_order=4,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.embedding_size = embedding_size
        self.positional_encoding_scheme = positional_encoding_scheme
        self.radial_encoding_periods = radial_encoding_periods
        self.radial_encoding_nth_order = radial_encoding_nth_order

        self.project_layer = layers.Dense(self.embedding_size)
        self.radial_embedding_layer = layers.Dense(self.embedding_size)
        self.positional_embedding_layer = layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.embedding_size
        )

    def build(self, input_shape):
        super(PatchEncoder, self).build(input_shape)

    def get_config(self):
        config = super(PatchEncoder, self).get_config()
        config.update({
            "num_patches": self.num_patches,
            "embedding_size": self.embedding_size,
            "positional_encoding_scheme": self.positional_encoding_scheme,
            "radial_encoding_periods": self.radial_encoding_periods,
            "radial_encoding_nth_order": self.radial_encoding_nth_order,
        })
        return config

    def _calc_radius(self):
        grid_size = int(np.sqrt(self.num_patches))
        d = np.linspace(-1+1/grid_size, 1-1/grid_size, grid_size, dtype=np.float32)
        ygrid, xgrid = np.meshgrid(d, d, indexing='ij')
        r = np.sqrt(ygrid.flatten()**2 + xgrid.flatten()**2)
        theta = np.arctan2(ygrid.flatten(), xgrid.flatten())
        return r.astype(np.float32), theta.astype(np.float32)

    def _nm_polynomial(self, n, m, rho, theta, normed=True):
        def _nm_normalization(n, m):
            """ return orthonormal zernike """
            return np.sqrt((1. + (m == 0)) / (2. * n + 2))

        if (n - m) % 2 == 1:
            poly = 0 * rho + 0 * theta
            return poly.astype(np.float32)

        radial = 0
        m0 = abs(m)

        for k in range((n - m0) // 2 + 1):
            a = binom(n - k, k)
            b = binom(n - 2 * k, (n - m0) // 2 - k)
            radial += (-1.) ** k * a * b * rho ** (n - 2 * k)

        # no clipping needed here
        # radial *= (rho <= 1.)

        if normed:  # return orthonormal zernike
            prefac = 1. / _nm_normalization(n, m)
        else:
            prefac = 1.

        if m >= 0:
            poly = prefac * radial * np.cos(m0 * theta)
        else:
            poly = prefac * radial * np.sin(m0 * theta)

        return poly.astype(np.float32)

    def _zernike_polynomials(self, radial_encoding_nth_order=4):
        r, theta = self._calc_radius()
        nm_pairs = set((n, m) for n in range(radial_encoding_nth_order + 1) for m in range(-n, n + 1, 2))
        polynomials = np.zeros((r.shape[0], len(nm_pairs)), dtype=np.float32)

        for i, (pr, pt) in enumerate(zip(r, theta)):
            for j, (n, m) in enumerate(nm_pairs):
                polynomials[i, j] = self._nm_polynomial(n=n, m=m, rho=pr, theta=pt, normed=True)

        return tf.constant(polynomials)

    def _fourier_decomposition(self, periods=1):
        r, theta = self._calc_radius()
        r = tf.constant(r, dtype=tf.float32)
        theta = tf.constant(theta, dtype=tf.float32)

        encodings = [r]
        for p in range(1, periods + 1):
            encodings.append(tf.sin(p * r))
            encodings.append(tf.cos(p * r))
            encodings.append(tf.sin(p * theta))
            encodings.append(tf.cos(p * theta))

        return tf.stack(encodings, axis=-1)

    def _power_decomposition(self, periods=1, radial_encoding_nth_order=4):
        r, theta = self._calc_radius()
        r = tf.constant(r, dtype=tf.float32)
        theta = tf.constant(theta, dtype=tf.float32)

        encodings = []
        for n in range(1, radial_encoding_nth_order + 1):
            encodings.append(tf.pow(r, n))

        for p in range(1, periods + 1):
            encodings.append(tf.sin(p * theta))
            encodings.append(tf.cos(p * theta))

        return tf.stack(encodings, axis=-1)

    def _rotational_symmetry(self, periods=1):
        r, theta = self._calc_radius()
        r = tf.constant(r, dtype=tf.float32)
        theta = tf.constant(theta, dtype=tf.float32)

        encodings = [r]
        for p in range(1, periods + 1):
            encodings.append(tf.sin(p * theta))
            encodings.append(tf.cos(p * theta))

        return tf.stack(encodings, axis=-1)

    def _patch_number(self):
        return tf.range(start=0, limit=self.num_patches, delta=1)

    def positional_encoding(self, inputs, scheme, periods, radial_encoding_nth_order):

        if scheme == 'rotational_symmetry' or scheme == 'rot_sym':
            pos = self._rotational_symmetry(periods=periods)

        elif scheme == 'fourier_decomposition':
            pos = self._fourier_decomposition(periods=periods)

        elif scheme == 'power_decomposition':
            pos = self._power_decomposition(periods=periods, radial_encoding_nth_order=radial_encoding_nth_order)

        elif scheme == 'zernike_polynomials':
            pos = self._zernike_polynomials(radial_encoding_nth_order=radial_encoding_nth_order)

        else:
            pos = self._patch_number()

        emb = []
        for _ in range(inputs.shape[1]):
            if scheme is None or scheme == 'default':
                emb.append(self.positional_embedding_layer(pos))
            else:
                emb.append(self.radial_embedding_layer(pos))

        return tf.stack(emb, axis=0)

    def call(
        self,
        inputs,
        training=True,
        periods=None,
        zernike_nth_order=None,
        scheme=None,
        **kwargs
    ):
        linear_projections = self.project_layer(inputs)

        if scheme is not None:
            self.positional_encoding_scheme = scheme
            self.radial_encoding_periods = periods
            self.radial_encoding_nth_order = zernike_nth_order

        positional_embeddings = self.positional_encoding(
            inputs,
            scheme=self.positional_encoding_scheme,
            periods=self.radial_encoding_periods,
            radial_encoding_nth_order=self.radial_encoding_nth_order,
        )

        return linear_projections + positional_embeddings


class MLP(layers.Layer):
    def __init__(
        self,
        expand_rate,
        dropout_rate,
        activation,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.expand_rate = expand_rate
        self.dropout_rate = dropout_rate
        self.activation = activation

    def build(self, input_shape):
        super(MLP, self).build(input_shape)
        self.expand = layers.Dense(int(self.expand_rate * input_shape[-1]), activation=self.activation)
        self.proj = layers.Dense(input_shape[-1], activation=self.activation)
        self.dropout = layers.Dropout(self.dropout_rate)

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            "expand_rate": self.expand_rate,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
        })
        return config

    def call(self, inputs, training=True, **kwargs):
        x = self.expand(inputs)
        x = self.dropout(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Transformer(layers.Layer):
    def __init__(
        self,
        heads,
        dims,
        activation,
        dropout_rate,
        expand_rate=4,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.dims = dims
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.expand_rate = expand_rate

        self.drop_path = tfa.layers.StochasticDepth(survival_probability=1-dropout_rate)
        self.ln = layers.LayerNormalization(axis=-1, epsilon=1e-6)
        self.msa = layers.MultiHeadAttention(
            num_heads=self.heads,
            key_dim=self.dims,
            dropout=self.dropout_rate
        )
        self.mlp = MLP(
            expand_rate=self.expand_rate,
            dropout_rate=self.dropout_rate,
            activation=self.activation
        )

    def build(self, input_shape):
        super(Transformer, self).build(input_shape)

    def get_config(self):
        config = super(Transformer, self).get_config()
        config.update({
            "heads": self.heads,
            "dims": self.dims,
            "dropout_rate": self.dropout_rate,
            "expand_rate": self.expand_rate,
            "activation": self.activation,
        })
        return config

    def call(self, inputs, training=True, **kwargs):
        ln1 = self.ln(inputs)
        att = self.msa(ln1, ln1)
        s1 = self.drop_path([att, inputs])

        ln2 = self.ln(s1)
        s2 = self.mlp(ln2)
        return self.drop_path([s1, s2])


class OpticalTransformer(Base, ABC):
    def __init__(
            self,
            roi=None,
            patches=(8, 8, 8, 8),
            heads=(2, 4, 8, 16),
            repeats=(2, 4, 6, 2),
            depth_scalar=1.0,
            width_scalar=1.0,
            activation='gelu',
            dropout_rate=0.05,
            expand_rate=4,
            rho=.05,
            mul=False,
            no_phase=False,
            radial_encoding_period=1,
            positional_encoding_scheme='default',
            radial_encoding_nth_order=4,
            decrease_dropout_depth=False,
            increase_dropout_depth=False,
            sam=False,
            stem=False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.roi = roi
        self.stem = stem
        self.patches = patches
        self.heads = heads
        self.repeats = repeats
        self.depth_scalar = depth_scalar
        self.width_scalar = width_scalar
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.expand_rate = expand_rate
        self.avg = layers.GlobalAvgPool2D()
        self.rho = rho
        self.mul = mul
        self.no_phase = no_phase
        self.radial_encoding_period = radial_encoding_period
        self.positional_encoding_scheme = positional_encoding_scheme
        self.radial_encoding_nth_order = radial_encoding_nth_order
        self.increase_dropout_depth = increase_dropout_depth
        self.decrease_dropout_depth = decrease_dropout_depth
        self.sam = sam

    def _calc_channels(self, channels, width_scalar):
        return int(tf.math.ceil(width_scalar * channels))

    def _calc_repeats(self, repeats, depth_scalar):
        return int(tf.math.ceil(depth_scalar * repeats))

    def sharpness_aware_minimization(self, x, y, sample_weight=None, rho=0.05, eps=1e-12):
        """
            Sharpness-Aware-Minimization (SAM): https://openreview.net/pdf?id=6Tm1mposlrM
            https://github.com/Jannoshh/simple-sam
        """

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # first step
        e_ws = []
        grad_norm = tf.linalg.global_norm(gradients)
        ew_multiplier = rho / (grad_norm + eps)
        for i in range(len(trainable_vars)):
            e_w = tf.math.multiply(gradients[i], ew_multiplier)
            trainable_vars[i].assign_add(e_w)
            e_ws.append(e_w)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        for i in range(len(trainable_vars)):
            trainable_vars[i].assign_sub(e_ws[i])

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):

        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        if self.sam:
            return self.sharpness_aware_minimization(x=x, y=y, sample_weight=sample_weight)
        else:

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)
            return {m.name: m.result() for m in self.metrics}

    def transitional_block(self, inputs, img_shape, patch_size, expansion=1, org_patch_size=None):
        if org_patch_size is not None:
            inputs = Merge(patch_size=org_patch_size, expansion=expansion)(inputs)

        m = Patchify(patch_size=patch_size)(inputs)

        try:
            m = PatchEncoder(
                num_patches=(img_shape//patch_size) ** 2,
                embedding_size=self._calc_channels(patch_size**2, width_scalar=self.width_scalar),
                positional_encoding_scheme=self.positional_encoding_scheme,
                radial_encoding_periods=self.radial_encoding_period,
                radial_encoding_nth_order=self.radial_encoding_nth_order
            )(m)
        except ValueError:
            m = PatchEncoder(
                num_patches=(img_shape // patch_size) ** 2,
                embedding_size=self._calc_channels(patch_size ** 2, width_scalar=self.width_scalar),
            )(
                m,
                periods=self.radial_encoding_period,
                scheme=self.positional_encoding_scheme,
                zernike_nth_order=self.radial_encoding_nth_order
            )
        return m

    def call(self, inputs, training=True, **kwargs):

        if self.stem:
            m = Stem(
                kernel_size=7,
                activation=self.activation,
                no_phase=self.no_phase,
                mul=self.mul,
            )(inputs)
        else:
            m = inputs

        if self.roi is not None:
            m = ROI(crop_shape=self.roi)(m)

        for i, (p, h, r) in enumerate(zip(self.patches, self.heads, self.repeats)):
            m = self.transitional_block(
                m,
                img_shape=self.roi[-1] if self.roi is not None else inputs.shape[-2],
                patch_size=self.patches[i],
                org_patch_size=None if i == 0 else self.patches[i-1],
            )

            res = m
            for j in range(self._calc_repeats(r, depth_scalar=self.depth_scalar)):
                if self.increase_dropout_depth:
                    dropout_rate = self.dropout_rate * (i + 1) / sum(self.repeats)
                elif self.decrease_dropout_depth:
                    dropout_rate = self.dropout_rate / (i + 1)
                else:
                    dropout_rate = self.dropout_rate

                m = Transformer(
                    heads=self._calc_channels(h, width_scalar=self.width_scalar),
                    dims=64,
                    activation=self.activation,
                    dropout_rate=dropout_rate,
                    expand_rate=self.expand_rate,
                )(m)
            m = layers.add([res, m])

        m = self.avg(m)
        return self.regressor(m)
