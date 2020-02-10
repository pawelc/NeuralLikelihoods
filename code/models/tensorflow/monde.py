from conf import conf
import  models.tensorflow.mykeras.layers as mylayers
import tensorflow as tf

from models.tensorflow.common import TfModel
from models.tensorflow.utils import constrain_cdf

tfk = tf.keras
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

class MONDELayer(tfk.layers.Layer):
    def __init__(self, cov_type, arch_hxy, arch_x_transform=None,
                 arch_cov_transform=None, hxy_x_size=0, covariance_learning_rate=None, **kwargs):
        super().__init__(**kwargs)

        if cov_type not in ['const_cov', 'param_cov', 'param_cor', None]:
            raise ValueError(cov_type)

        self._cov_type = cov_type

        if self._cov_type == 'param_cov' or self._cov_type == 'param_cor':
            if not arch_cov_transform:
                raise ValueError("arch_cov missing")
            self._arch_cov_transform = arch_cov_transform
        else:
            self._arch_cov_transform = None

        self._arch_hxy = arch_hxy
        self._arch_x_transform = arch_x_transform
        self._hxy_x_size = hxy_x_size
        self._covariance_learning_rate = covariance_learning_rate

    def build(self, input_shape):
        self._y_size = input_shape[0][-1]
        self._x_size = input_shape[1][-1]

        if self._arch_x_transform:
            self._x_transform = tfk.Sequential(layers=[tfk.layers.Dense(units, activation='tanh')
                                                       for units in self._arch_x_transform], name="x_transform")

        mon_size_ins = [1] + [units - self._hxy_x_size for units in self._arch_hxy]
        non_mon_size_ins = [self._arch_x_transform[-1] if self._arch_x_transform else self._x_size] + [self._hxy_x_size for _ in self._arch_hxy]
        mon_size_outs = [units - self._hxy_x_size for units in self._arch_hxy] + [1]
        non_mon_size_outs = [self._hxy_x_size for _ in self._arch_hxy] + [0]

        self._h_xys_transforms = [tfk.Sequential(
            layers=[mylayers.Dense(units, activation='sigmoid' if layer == len(self._arch_hxy) else 'tanh',
                                   kernel_constraint=mylayers.MonotonicConstraint(mon_size_in, non_mon_size_in,
                                                                                  mon_size_out, non_mon_size_out),
                                   name="h_xy_%d_%d" % (i, layer))
                    for layer, units, mon_size_in, non_mon_size_in, mon_size_out, non_mon_size_out in
                    zip(range(len(self._arch_hxy) + 1), self._arch_hxy + [1], mon_size_ins, non_mon_size_ins, mon_size_outs,
                        non_mon_size_outs)]
            , name="h_xy_%d" % i) for i in range(self._y_size)]

        if self._arch_cov_transform:
            self._cov_transform = tfk.Sequential(layers=[tfk.layers.Dense(units, activation="tanh", name="cov_layer_%d" % i,
                                                                   kernel_initializer=tfk.initializers.RandomNormal(
                                                                       mean=0, stddev=0.01))
                                                         for i, units in enumerate(self._arch_cov_transform)])
        else:
            self._cov_var = tf.Variable(tfk.initializers.constant(np.nan)((self._y_size, self._y_size)),
                                        name="cov_var",
                                        dtype=getattr(tf, "float%s" % conf.precision), trainable=False)

        ######
        if self._cov_type in ['param_cov', 'param_cor', None]:
            if self._x_size > 0:
                if self._arch_cov_transform:
                    last_hidden_units_size = self._arch_cov_transform[-1]
                    init = tfk.initializers.RandomNormal(mean=0, stddev=0.01)

                    self._W_cov_u = tf.Variable(init(shape=[int(last_hidden_units_size / 2), self._y_size]),
                                                dtype=getattr(tf, "float%s" % conf.precision), name="W_cov_u")

                    self._b_cov_u = tf.Variable(tf.initializers.zeros()(shape=[1, self._y_size]),
                                                dtype=getattr(tf, "float%s" % conf.precision), name="b_cov_u")

                    self._W_cov_d = tf.Variable(
                        init(shape=[last_hidden_units_size - int(last_hidden_units_size / 2), self._y_size]),
                        dtype=getattr(tf, "float%s" % conf.precision), name="W_cov_d")

                    self._b_cov_d = tf.Variable(tf.initializers.zeros()(shape=[1, self._y_size]),
                                                dtype=getattr(tf, "float%s" % conf.precision), name="b_cov_d")

                else:
                    raise ValueError("missing arch_cov_transform")
            else:
                self._cov_u = tf.Variable(tf.initializers.zeros()(shape=(1, self._y_size)),
                                          dtype=getattr(tf, "float%s" % conf.precision), name="cov_u")
                self._cov_d_raw = tf.Variable(tf.initializers.zeros()(shape=(1, self._y_size)),
                                              dtype=getattr(tf, "float%s" % conf.precision), name="cov_d_raw")

        if self._cov_type == "param_cor":
            r = self.cor_rank if hasattr(self, "cor_rank") else self._y_size
            num_theta = int((r - 1) * (self._y_size - r / 2))
            if self._x_size > 0:
                self._theta_transform = tfk.Dense(num_theta, name="cov")
            else:
                self._cov_u = tf.Variable(tf.initializers.zeros()(shape=(1, num_theta)),
                                          dtype=getattr(tf, "float%s" % conf.precision), name="cov_u")
        super(MONDELayer, self).build(input_shape)

    # @tf.function
    def call(self, inputs):
        return self.log_prob(inputs[0], inputs[1])

    # @tf.function
    def prob(self, y, x, marginal=None, training=False):
        return tf.math.exp(self.log_prob(y, x, marginal=marginal, training=training))

    # @tf.function
    def log_prob(self, y, x, marginal=None, marginals=None, training=False):
        if y is None:
            raise NotImplementedError

        if x is not None and self._arch_x_transform:
            x = self._x_transform(x)

        cdfs, pdfs = self.marginal_density_estimator(x, y)

        predictions = self.marginals_and_joint(cdfs, pdfs, x, training=training)

        return predictions["log_likelihood"]

    def marginal_density_estimator(self, x, y):
        pdfs = []
        cdfs = []

        for i in range(self._y_size):
            y_margin = tf.slice(y, [0, i], size=[-1, 1])

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(y_margin)
                if x is not None:
                    yx = tfk.layers.Concatenate()([y_margin, x])
                else:
                    yx = y_margin

                cdf = constrain_cdf(self._h_xys_transforms[i](yx))

                grads = tape.gradient(cdf, y_margin)

            cdfs.append(cdf)
            pdfs.append(grads)
        return cdfs, pdfs

    def marginals_and_joint(self, cdfs, pdfs, x, training):
        predictions = {}
        lls = []

        for i, cdf, pdf in zip(range(len(pdfs)), cdfs, pdfs):
            predictions["cdf%d" % i] = cdf
            if pdf is not None:
                predictions["pdf%d" % i] = pdf
                lls.append(tf.math.log(pdf + 1e-27))
            else:
                lls.append(None)

        predictions.update(self.log_likelihood_from_cdfs_transforms(cdfs, lls, x, training=training))
        return predictions

    def log_likelihood_from_cdfs_transforms(self, cdfs, lls, x, training, marginals=None):

        if len(lls) == 1 or self._cov_type is None:
            return {'log_likelihood': lls[0]}

        quantiles = []
        std_normal = tfd.Normal(loc=0., scale=1.)
        for i, cdf in enumerate(cdfs):
            quantiles.append(std_normal.quantile(cdf, name="quantile"))

        quantiles_combined = tf.concat(quantiles, axis=1, name="quantiles_combined")
        # TODO stop gradient here quantiles_combined so when computing gradients for the joint part we don't affect the marginals ...
        # quantiles_combined = tf.stop_gradient(quantiles_combined)
        if self._cov_type == "param_cov":
            cov = self.covariance(x, len(cdfs))
            cor = self.correlation_matrix(cov)
            cor = tf.minimum(tf.maximum(cor, -1.0), 1.0)
        elif self._cov_type == "param_cor":
            cor = self.correlation_directly(x)
        elif self._cov_type == "const_cov":
            is_nan = tf.reduce_all(tf.math.is_nan(self._cov_var))
            if training or is_nan:
                batch_cov = self.tf_cov(quantiles_combined)
                batch_cov = self.assert_cov_positive_definite(batch_cov)
                if is_nan:
                    self._cov_var.assign(batch_cov)
                else:
                    self._cov_var.assign_sub(self._covariance_learning_rate * (self._cov_var - batch_cov))

            cor = self.corr()
            cor = tf.expand_dims(cor, axis=0)
            # cor = tf.minimum(tf.maximum(cor, -1.0), 1.0)
        else:
            raise ValueError("Not recognized covariance type: %s" % self._cov_type)

        # copula implementation https://math.stackexchange.com/questions/1767940/density-of-a-distribution-given-by-a-gaussian-copula-and-a-set-of-marginals

        #### wrong copula
        # c_pdf = tp.distributions.MultivariateNormalFullCovariance(loc=tf.constant([0.0] * len(cdfs)),
        #                                                       covariance_matrix=cor)
        # c_pdf_log_prob = tf.reshape(c_pdf.log_prob(tf.stop_gradient(quantiles_combined)), [-1, 1], name="copula_log_prob")

        # if x is not None:
        ##### original met copula where I copute x' * (inv(R)-I) * x
        # c_pdf_log_prob = -0.5*tf.math.log(tf.matrix_determinant(cor) + 1e-6)
        #
        # inv_cor_min_eye = tf.matrix_inverse(cor) - tf.eye(len(cdfs))
        #
        # inv_cor_min_eye = tf.tile(inv_cor_min_eye,
        #                           [tf.math.divide(tf.shape(quantiles_combined)[0], tf.shape(inv_cor_min_eye)[0]), 1, 1])
        # c_pdf_log_exponent = tf.matmul(tf.expand_dims(quantiles_combined, axis=1), inv_cor_min_eye)
        # c_pdf_log_exponent = - 0.5 * tf.matmul(c_pdf_log_exponent, tf.expand_dims(quantiles_combined, axis=-1))
        # c_pdf_log_exponent = tf.reshape(c_pdf_log_exponent, [-1, 1])
        # c_pdf_log_prob = tf.reshape(c_pdf_log_prob, [-1, 1]) + c_pdf_log_exponent

        ##### compute x'* inv(R) * x and than separately x'* I * x
        # c_pdf_log_prob = -0.5 * tf.math.log(tf.matrix_determinant(cor) + 1e-6)
        #
        # inv_cor = tf.matrix_inverse(cor)
        # inv_cor = tf.tile(inv_cor, [tf.math.divide(tf.shape(quantiles_combined)[0], tf.shape(inv_cor)[0]), 1, 1])
        #
        # cor_prod = -0.5 * tf.matmul(tf.expand_dims(quantiles_combined, axis=1), inv_cor)
        # cor_prod = tf.matmul(cor_prod, tf.expand_dims(quantiles_combined, axis=-1))
        # cor_prod = tf.reshape(cor_prod, [-1, 1])
        #
        # id_prod = -0.5 * tf.matmul(quantiles_combined, tf.eye(len(cdfs)))
        # id_prod = tf.reduce_sum(tf.multiply(id_prod, quantiles_combined), axis=1)
        # id_prod = tf.reshape(id_prod, [-1, 1])
        #
        # c_pdf_log_prob = tf.reshape(c_pdf_log_prob, [-1, 1]) + cor_prod - id_prod

        ##### compute Cholesky decomposition to compute x'* inv(R) * x
        if marginals:
            log_likelihood = []
            for marginal in marginals:
                quantiles_combined_subset = tf.gather(quantiles_combined, marginal, axis=1)

                cor_subset = tf.gather(cor, marginal, axis=2)
                cor_subset = tf.gather(cor_subset, marginal, axis=1)
                lls_subset = [lls[i] for i in marginal]

                log_likelihood.append(
                    self.compute_log_likelihood_for_cor(quantiles_combined_subset, cor_subset, lls_subset))

        else:
            log_likelihood = self.compute_log_likelihood_for_cor(quantiles_combined, cor, lls)

        return {'log_likelihood': log_likelihood, 'cor': cor}

    def covariance(self, x, dim):
        if x is not None:
            layer = x
            if self._arch_cov_transform:
                last_hidden_units_size = self._arch_cov_transform[-1]
                layer = self._cov_transform(layer)
                layer_u = tf.slice(layer, [0, 0], [-1, int(last_hidden_units_size / 2)])
                layer_d = tf.slice(layer, [0, int(last_hidden_units_size / 2)], [-1, -1])

                cov_u = tf.expand_dims(tf.matmul(layer_u, self._W_cov_u) + self._b_cov_u, 1)

                cov_d = tf.add(1e-27, tf.square(tf.matmul(layer_d, self._W_cov_d) + self._b_cov_d))
        else:
            cov_u = self._cov_u
            cov_d = tf.add(1e-27, tf.square(self._cov_d_raw, name="cov_d"))

        diagonal = tf.linalg.diag(cov_d)

        return tf.add(tf.matmul(cov_u, cov_u, transpose_a=True), diagonal, name="covariance")

    def correlation_matrix(self, cov):
        std_dev = tf.linalg.diag(tf.sqrt(tf.linalg.diag_part(cov)))
        std_dev_inv = tf.linalg.inv(std_dev)

        corr = tf.matmul(tf.matmul(std_dev_inv, cov), std_dev_inv)
        # corr = ((tf.matrix_transpose(corr) + corr) / 2)
        # replicating upper part to be lower because of numerical precision the matrix could be not symetric even
        # if analytically it should

        # it looks it is not necessary but slows down computation
        # corr = tf.matrix_band_part(corr, -1, 0)
        # corr = corr + tf.matrix_transpose(corr)

        # setting diagonal to be 1s
        corr = tf.linalg.set_diag(corr, tf.fill(tf.slice(tf.shape(corr), [0], [2]), 1.0), name="correlation")

        return corr

    def correlation_directly(self, x):
        r = self.cor_rank if hasattr(self, "cor_rank") else self._y_size
        if x is not None:
            layer = x

            if self._arch_cov_transform:
                layer = self._cov_transform(layer)
                layer = self._theta_transform(layer)
            else:
                raise ValueError("arch_cov missing")
        else:
            layer = self._cov_u

        theta = np.pi / 2 - tf.tanh(layer)
        B = [self.b_mat_el(theta, i, j, r) for i in range(self._y_size) for j in range(r)]
        B = tf.concat(B, axis=1)
        B = tf.reshape(B, (-1, self._y_size, r))
        cor = tf.matmul(B, B, transpose_b=True)
        cor = tf.linalg.set_diag(cor, tf.fill(tf.slice(tf.shape(cor), [0], [2]), 1.0), name="correlation")
        return cor

    def b_mat_el(self, theta_mat, i, j, r):
        start_idx = np.sum(np.minimum(np.arange(i), r - 1))
        if i == 0 and j == 0:
            return tf.fill([tf.shape(theta_mat)[0], 1], 1.)
        elif j == 0:
            return tf.cos(theta_mat[:, start_idx, None])
        elif j == min(i, r - 1):
            return tf.reduce_prod(tf.sin(theta_mat[:, start_idx:start_idx + j, None]), axis=1)
        elif 1 <= j <= min(i - 1, r - 2) and i >= 1:
            return tf.cos(theta_mat[:, start_idx + j, None]) * tf.reduce_prod(
                tf.sin(theta_mat[:, start_idx:start_idx + j, None]), axis=1)
        else:
            return tf.fill([tf.shape(theta_mat)[0], 1], 0.)

    def tf_cov(self, x):
        mean_x = tf.reduce_mean(x, axis=0, keepdims=True, name="mean")
        mx = tf.matmul(tf.transpose(mean_x), mean_x, name="mx")
        vx = tf.divide(tf.matmul(tf.transpose(x), x), tf.cast(tf.shape(x)[0], tf.float32), name="vx")
        cov_xx = tf.subtract(vx, mx, name="cov")
        return cov_xx

    def corr(self):
        diag = tf.linalg.diag(tf.linalg.diag_part(self._cov_var))
        D = tf.math.sqrt(diag)
        DInv = tf.linalg.inv(D)
        corr = tf.matmul(tf.matmul(DInv, self._cov_var), DInv)

        corr = tf.linalg.band_part(corr, -1, 0)
        corr = corr + tf.linalg.matrix_transpose(corr)

        corr = tf.linalg.set_diag(corr, [1.] * corr.shape[0], name="correlation")
        return corr

    def assert_cov_positive_definite(self, cov):
        e, v = tf.linalg.eigh(cov)
        i = tf.constant(0)

        def adjust_cov(i, cov, eig_val):
            eig_val_max = tf.maximum(1e-6, tf.abs(eig_val))
            cov1 = cov + tf.eye(cov.shape[0]) * tf.abs(eig_val_max)
            e, v = tf.linalg.eigh(cov1)
            return tf.add(i, 1), cov1, e[0]

        def cond(i, cov, eig_val):
            return tf.less_equal(eig_val, 1e-6)

        i, cov, _ = tf.while_loop(cond, adjust_cov, [i, cov, e[0]])
        return cov

    def compute_log_likelihood_for_cor(self, quantiles_combined, cor, ll_marginals):
        ll_marginals_sum = tf.add_n(ll_marginals)

        cor = cor + tf.linalg.diag([1e-3] * cor.shape[-1])
        c_pdf_log_prob = -0.5 * tf.math.log(tf.linalg.det(cor) + 1e-27)

        V = tf.linalg.cholesky(tf.tile(cor, [tf.math.divide(tf.shape(quantiles_combined)[0], tf.shape(cor)[0]), 1, 1]))
        z = tf.linalg.solve(V, tf.expand_dims(quantiles_combined, -1))
        cor_prod = -0.5 * tf.reshape(tf.matmul(z, z, transpose_a=True), [-1, 1])

        id_prod = -0.5 * tf.matmul(quantiles_combined, tf.eye(quantiles_combined.shape[1]))
        id_prod = tf.reduce_sum(tf.multiply(id_prod, quantiles_combined), axis=1)
        id_prod = tf.reshape(id_prod, [-1, 1])

        c_pdf_log_prob = tf.reshape(c_pdf_log_prob, [-1, 1]) + cor_prod - id_prod

        ##### do not use tiliing in case of one cor matrix but strangely it didn't help with throughput
        # c_pdf_log_prob = -0.5 * tf.math.log(tf.matrix_determinant(cor[0]) + 1e-3)
        #
        # inv_cor_min_eye = tf.linalg.inv(cor[0]) - tf.eye(self._y_size)
        # c_pdf_log_exponent = tf.matmul(quantiles_combined, inv_cor_min_eye)
        # c_pdf_log_exponent = - 0.5 * tf.reduce_sum(tf.multiply(c_pdf_log_exponent, quantiles_combined), 1,
        #                                            keep_dims=True)
        # c_pdf_log_prob = tf.reshape(c_pdf_log_prob, [-1, 1]) + c_pdf_log_exponent

        return tf.add(c_pdf_log_prob, ll_marginals_sum)

    def get_config(self):
        return {'cov_type': self._cov_type, 'arch_hxy': self._arch_hxy,
                'arch_x_transform': self._arch_x_transform, 'arch_cov_transform': self._arch_cov_transform,
                'hxy_x_size': self._hxy_x_size, 'covariance_learning_rate': self._covariance_learning_rate}


class MONDE(TfModel):

    def __init__(self, cov_type, arch_hxy, arch_x_transform = None ,
                 arch_cov_transform =None, hxy_x_size = 0, covariance_learning_rate = None, **kwargs):
        super().__init__(**kwargs)
        self.monde_layer = MONDELayer(cov_type=cov_type, arch_hxy=arch_hxy, arch_x_transform = arch_x_transform ,
                 arch_cov_transform =arch_cov_transform, hxy_x_size = hxy_x_size, covariance_learning_rate=covariance_learning_rate)

    def build(self, input_shape):
        self.monde_layer.build(input_shape)
        super(MONDE, self).build(input_shape)

    @tf.function
    def call(self, inputs):
        return self.monde_layer.call(inputs)

    @tf.function
    def prob(self, y, x, marginal=None, training=False):
        return self.monde_layer.prob(y=y, x=x, marginal=marginal, training=training)

    @tf.function
    def log_prob(self, y, x, marginal=None, marginals=None, training=False):
        return self.monde_layer.log_prob(y=y, x=x, marginal=marginal, marginals=marginals, training=training)

    def get_config(self):
        return self.monde_layer.get_config()

    @property
    def model_name(self):
        return "monde_" + self.monde_layer._cov_type



