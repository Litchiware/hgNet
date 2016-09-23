import config
from batch_norm import batch_norm
import tensorflow as tf

def _conv_block(x, depth_in, depth_out, is_training):
    with tf.variable_scope('conv1'):
        weights = tf.get_variable('weights', [1, 1, depth_in, depth_out/2],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1,
                    seed=config.SEED,
                    dtype=tf.float32))
        biases = tf.get_variable('biases', [depth_out/2],
                initializer=tf.constant_initializer(0.0))
        net = batch_norm(x, is_training)
        net = tf.nn.relu(net)
        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, biases)

    with tf.variable_scope('conv2'):
        weights = tf.get_variable('weights', [3, 3, depth_out/2, depth_out/2],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1,
                    seed=config.SEED,
                    dtype=tf.float32))
        biases = tf.get_variable('biases', [depth_out/2],
                initializer=tf.constant_initializer(0.0))
        net = batch_norm(net, is_training)
        net = tf.nn.relu(net)
        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, biases)

    with tf.variable_scope('conv3'):
        weights = tf.get_variable('weights', [1, 1, depth_out/2, depth_out],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1,
                    seed=config.SEED,
                    dtype=tf.float32))
        biases = tf.get_variable('biases', [depth_out],
                initializer=tf.constant_initializer(0.0))
        net = batch_norm(net, is_training)
        net = tf.nn.relu(net)
        net = tf.nn.conv2d(net, weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.bias_add(net, biases)

    return net


def _skip_layer(x, depth_in, depth_out):
    if depth_in == depth_out:
        return x

    weights = tf.get_variable('weights', [1, 1, depth_in, depth_out],
            initializer=tf.truncated_normal_initializer(
                stddev=0.1,
                seed=config.SEED,
                dtype=tf.float32))
    biases = tf.get_variable('biases', [depth_out],
            initializer=tf.constant_initializer(0.0))
    net = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.bias_add(net, biases)
    return net


def residual(x, depth_in, depth_out, is_training):
    with tf.variable_scope('skip'):
        skip = _skip_layer(x, depth_in, depth_out)

    with tf.variable_scope('conv_block'):
        conv_block = _conv_block(x, depth_in, depth_out, is_training)

    return  skip + conv_block


def hourglass(n, x, depth, is_training):
    with tf.variable_scope('branch%d' %n):
        with tf.variable_scope('residual1'):
            up1 = residual(x, depth, depth, is_training)

        pool = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        with tf.variable_scope('residual2'):
            low1 = residual(pool, depth, depth, is_training)
        if n > 1:
            low2 = hourglass(n-1, low1, depth, is_training)
        else:
            with tf.variable_scope('residual3'):
                low2 = residual(low1, depth, depth, is_training)

        with tf.variable_scope('residual4'):
            low3 = residual(low2, depth, depth, is_training)
        height, width = low3.get_shape().as_list()[1:3]
        up2 = tf.image.resize_nearest_neighbor(low3, [2 * height, 2 * width])

    return up1 + up2


def lin(x, depth_in, depth_out, is_training):
    weights = tf.get_variable('weights', [1, 1, depth_in, depth_out],
            initializer=tf.truncated_normal_initializer(
                stddev=0.1,
                seed=config.SEED,
                dtype=tf.float32))
    biases = tf.get_variable('biases', [depth_out],
            initializer=tf.constant_initializer(0.0))

    net = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
    net = tf.nn.bias_add(net, biases)
    net = batch_norm(net, is_training)
    net = tf.nn.relu(net)

    return net


def inference(x, is_training):
    with tf.variable_scope('input'):
        weights = tf.get_variable('weights', [7, 7, 3, 64],
                initializer=tf.truncated_normal_initializer(
                    stddev=0.1,
                    seed=config.SEED,
                    dtype=tf.float32))
        biases = tf.get_variable('biases', [64],
                initializer=tf.constant_initializer(0.0))

    net = tf.nn.conv2d(x, weights, strides=[1, 2, 2, 1], padding='SAME')
    # TODO: try to remove this batch_norm and relu layer
    net = batch_norm(net, is_training)
    net = tf.nn.relu(net)
    with tf.variable_scope('residual1'):
        net = residual(net, 64, 128, is_training)

    net = tf.nn.max_pool(net, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('residual2'):
        net = residual(net, 128, 128, is_training)
    with tf.variable_scope('residual3'):
        net = residual(net, 128, config.N_FEATS, is_training)

    out = []
    for i in range(config.N_STACK):
        with tf.variable_scope('hourglass%d' %i):
            hg = hourglass(4, net, config.N_FEATS, is_training)

        # TODO: try to remove this lin layer
        with tf.variable_scope('lin%d' %i):
            ll = lin(hg, config.N_FEATS, config.N_FEATS, is_training)

        with tf.variable_scope('output%d' %i):
            weights = tf.get_variable('weights', [1, 1, config.N_FEATS, config.OUT_CHA],
                    initializer=tf.truncated_normal_initializer(
                        stddev=0.1,
                        seed=config.SEED,
                        dtype=tf.float32))
            biases = tf.get_variable('biases', [config.OUT_CHA],
                    initializer=tf.constant_initializer(0.0))

        interOut = tf.nn.conv2d(ll, weights, strides=[1, 1, 1, 1], padding='SAME')
        out.append(interOut)
        
        net += hg

    return out
