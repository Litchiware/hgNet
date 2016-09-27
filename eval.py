import config
import tensorflow as tf

def get_preds(hms):
    batch_size, channels = hms.get_shape().as_list()[:2]
    hms = tf.reshape(hms, [batch_size, channels, -1])

    pred_x = pred_y = tf.argmax(hms, 2)
    pred_x = pred_x % config.OUT_RES
    pred_y = pred_y / config.OUT_RES
    preds = tf.concat(2, [tf.expand_dims(pred_x, 2), tf.expand_dims(pred_y, 2)])
    return preds


def _acc(preds, labels):
    indices = tf.squeeze(tf.where(tf.reduce_all(labels > 0, 1)), [1])
    preds = tf.gather(preds, indices)
    labels = tf.gather(labels, indices)
    preds = tf.cast(preds, tf.float32)
    labels = tf.cast(labels, tf.float32)

    dists = tf.sqrt(tf.reduce_sum(tf.square(preds - labels), 1))
    threshold = 1. * config.OUT_RES * config.THRESHOLD / config.SCALE_REF
    acc = tf.reduce_sum(tf.cast(dists < threshold, tf.float32))
    num = tf.cast(tf.squeeze(tf.shape(dists)), tf.float32)
    acc = acc / num
    return acc


def accuracies(outputs, heatmaps):
    outputs = tf.transpose(outputs, (3, 0, 1, 2))
    heatmaps = tf.transpose(heatmaps, (3, 0, 1, 2))
    preds = get_preds(outputs)
    labels = get_preds(heatmaps)
    accs = tf.map_fn(lambda x: _acc(x[0], x[1]), (preds, labels), dtype=tf.float32)
    
    return accs


if __name__ == "__main__":
    import numpy as np
    a = np.zeros((3, 64, 64, 16))
    b = np.zeros((3, 64, 64, 16))
    a[:, 30, 30, :] = 1
    b[:, 35, 35, :] = 1
    a = tf.convert_to_tensor(a)
    b = tf.convert_to_tensor(b)
    accs = accuracies(a, b)
    sess = tf.Session()
    print accs.eval(session=sess)
