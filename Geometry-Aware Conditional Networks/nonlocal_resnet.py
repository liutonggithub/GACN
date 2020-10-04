import tensorflow as tf
from ops import *


def NonLocalBlock(input_x, output_channels, sub_sample=False, is_bn=True, is_training=True, scope="NonLocalBlock"):
    batch_size, height, width, in_channels = input_x.get_shape().as_list()
    with tf.variable_scope(scope):
        with tf.variable_scope("g"):
            g = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="g_conv")
            if sub_sample:
                g = tf.layers.max_pooling2d(inputs=g, pool_size=2, strides=2, padding="valid", name="g_max_pool")
                print(g.shape)

        with tf.variable_scope("phi"):
            phi = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="phi_conv")
            if sub_sample:
                phi = tf.layers.max_pooling2d(inputs=phi, pool_size=2, strides=2, padding="valid", name="phi_max_pool")
                #print(phi.shape)

        with tf.variable_scope("theta"):
            theta = tf.layers.conv2d(inputs=input_x, filters=output_channels, kernel_size=1, strides=1, padding="same", name="theta_conv")
            print(theta.shape)

        #g_x = tf.reshape(g, [-1, output_channels, height * width])
        g_x = tf.reshape(g, [-1, height * width, output_channels])
        #g_x = tf.transpose(g_x, [0, 2, 1])
        #print(g_x.shape)

        phi_x = tf.reshape(phi, [-1, output_channels, height * width])
        #print(phi_x.shape)

        #theta_x = tf.reshape(theta, [-1, output_channels, height * width])
        #theta_x = tf.transpose(theta_x, [0, 2, 1])
        theta_x = tf.reshape(theta, [-1, height * width, output_channels])
        print(theta_x.shape)

        f = tf.matmul(theta_x, phi_x)
        f_softmax = tf.nn.softmax(f, -1)
        y = tf.matmul(f_softmax, g_x)

        y = tf.reshape(y, [-1, height, width, output_channels])

        with tf.variable_scope("w"):
            w_y = tf.layers.conv2d(inputs=y, filters=in_channels, kernel_size=1, strides=1, padding="same", name="w_conv")
            if is_bn:
                w_y= tf.layers.batch_normalization(w_y, axis=3, training=is_training)    ### batch_normalization
        z = input_x + w_y

        return z

def nonlocalfc(x, output_units):
    x = tf.reshape(x, shape=[-1, x.get_shape()[-1]])
    x = tf.layers.dense(inputs=x,
                        units=output_units)
    return x

def global_avg_pool(x):
    axis = [1, 2]
    return tf.reduce_mean(x, axis, keep_dims=True)

def residual(x, output_channels, strides, type, is_training):
    # type: #short cut type, "conv" or "identity"
    short_cut = x

    # short_cut
    if type == "conv":
        short_cut = batch_norm_nonlocal("conv1_b1_bn", short_cut, is_training)
        short_cut = relu(short_cut)
        short_cut = pad_conv("conv1_b1", short_cut, 1, output_channels, strides)

    # bottleneck residual block
    x = batch_norm_nonlocal("conv1_b2_bn", x, is_training)
    x = relu(x)
    x = pad_conv("conv1_b2", x, 1, output_channels / 4, 1)
    x = batch_norm_nonlocal("conv2_b2_bn", x, is_training)
    x = relu(x)
    x = pad_conv("conv2_b2", x, 3, output_channels / 4, strides)
    x = batch_norm_nonlocal("conv3_b2_bn", x, is_training)
    x = relu(x)
    x = pad_conv("conv3_b2", x, 1, output_channels, 1)

    return x + short_cut

# def relu(x):
#     return tf.nn.relu(x)

def batch_norm_nonlocal(name, x, is_training=True):
    with tf.variable_scope(name):
        return tf.layers.batch_normalization(inputs=x,
                                    axis=3,
                                    momentum=0.99,
                                    epsilon=1e-5,
                                    training=is_training,
                                    fused=True)

def pad_conv(name, x, kernel_size, output_channels, strides, bias=False):
    if strides > 1:
        x = padding(x, kernel_size)
    with tf.variable_scope(name):
        x = tf.layers.conv2d(inputs=x,
                             filters=output_channels,
                             kernel_size=kernel_size,
                             strides=strides,
                             padding=("same" if strides == 1 else "valid"),
                             use_bias=bias,
                             kernel_initializer=tf.variance_scaling_initializer())
        return x


def padding(x, kernel_size):
    # Padding based on kernel_size
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return x
