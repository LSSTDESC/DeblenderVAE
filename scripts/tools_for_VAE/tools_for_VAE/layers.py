import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K



class ImageScaleLayer(Layer):
    """
    Multiplies an image by a scalar
    """

    def __init__(self, *args, **kwargs):
        super(ImageScaleLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        scale, image = inputs
        return Multiply()([scale, image])


class MultMatrixVector(Layer):
    """
    Perform the multiplication of a matrix and a vector.
    """

    def __init__(self, *args, **kwargs):
        super(MultMatrixVector, self).__init__(*args, **kwargs)

    def call(self, inputs):
        A, x = inputs
        return tf.einsum('aij,aj->ai', A, x)


class SampleMultivariateGaussian(Layer):
    """
    Samples from a multivariate Gaussian given a mean and a full covariance matrix or just diagonal std.
    """

    def __init__(self, full_cov, add_KL, return_KL, coeff_KL=1.0, *args, **kwargs):
        """
        full_cov: whether to use a full covariance matrix or just the diagonal.
        add_KL: boolean, whether to add the (sample average) KL divergence of the input distribution with respect to a standard Gaussian
        return_KL: whether to return the value of the KL divergence (one value per sample).
        """
        self.full_cov = full_cov
        self.add_KL = add_KL
        self.return_KL = return_KL
        self.coeff_KL = coeff_KL

        if full_cov:
            self.distrib = tfp.distributions.MultivariateNormalFullCovariance
        else:
            self.distrib = tfp.distributions.MultivariateNormalDiag

        super(SampleMultivariateGaussian,
              self).__init__(*args, **kwargs)

    def call(self, inputs):
        """
        inputs = if full_cov is True, [mu, cov] where mu is the mean vector and cov the covariance matrix, otherwise [mu,sigma] where sigma is the std.
        """
        if self.full_cov:
            z_mu, z_cov = inputs
            dist_z = self.distrib(loc=z_mu, covariance_matrix=z_cov)
            dist_0 = self.distrib(
                loc=tf.zeros_like(z_mu), covariance_matrix=tf.identity(z_cov))

        else:
            z_mu, z_sigma = inputs
            dist_z = self.distrib(loc=z_mu, scale_diag=z_sigma)
            dist_0 = self.distrib(loc=tf.zeros_like(
                z_mu), scale_diag=tf.ones_like(z_sigma))

        z = dist_z.sample()
        
        if self.add_KL or self.return_KL:
            kl_divergence = tfp.distributions.kl_divergence(
                dist_z, dist_0, name='KL_divergence_full_cov')
            if self.add_KL:
                self.add_loss(self.coeff_KL*K.mean(kl_divergence), inputs=inputs)
            if self.return_KL:
                return z, kl_divergence

        return z

    def compute_output_shape(self, input_shape):
        """
        Same shape as the mean vector
        """
        return input_shape[0]


class FillLowerMatrix(Layer):
    """
    Fill a vector of size N*(N-1)/2 into the lower triangular part of a matrix of size N-by-N with ones on the diagonal.
    """

    def __init__(self, output_dim, *args, **kwargs):
        self.output_dim = output_dim

        indices = []
        for diag in range(1, output_dim):
            for k in range(output_dim - diag):
                indices.append([diag + k, k])
        indices = np.array(indices)

        self.indices = tf.constant(indices, dtype=tf.int64)

        self.unit_matrix = tf.Variable(
            np.identity(self.output_dim, dtype=np.float32))

        self.fn = lambda _x: tf.scatter_nd(self.indices, _x, shape=(
            self.output_dim, self.output_dim,)) + self.unit_matrix

        super(FillLowerMatrix, self).__init__(*args, **kwargs)

    def call(self, inputs):
        return tf.map_fn(self.fn, inputs)


class SPDMatrix(Layer):
    """
    Returns a symmetric positive definite matrix of size M-by-M from M vectors of size N packed into an N-by-M matrix, as in http://arxiv.org/abs/1711.06540v2
    """

    def __init__(self, input_dim, output_dim, alpha=1., dtype='float32', *args, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha

        N = input_dim
        M = output_dim
        A = np.zeros((M, M, N, M, N), dtype=dtype)
        for i in range(M):
            for j in range(M):
                for k in range(N):
                    A[i, j, k, i, k] += 1.0
                    A[i, j, k, j, k] -= 1.0

        self.A = tf.constant(A)
        super(SPDMatrix, self).__init__(*args, **kwargs)

    def call(self, inputs):
        # Suppose the input is an N*M matrix where M is out_dim
        return K.exp(- self.alpha * K.mean(K.square(tf.transpose(tf.tensordot(self.A, inputs, axes=([3, 4], [1, 2])), perm=[3, 0, 1, 2])), axis=3))
