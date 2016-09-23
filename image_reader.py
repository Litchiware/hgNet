import os
import h5py
import config
import tensorflow as tf

def transform(pt, center, scale, res, invert):
    h = config.SCALE_REF * scale
    # scale
    t = tf.diag([res/h, res/h, 1])
    # translation
    t = t + tf.pad(tf.expand_dims(res*(-center/h + 0.5), 1),
            [[0, 1], [2, 0]], "CONSTANT")

    if invert:
        t = tf.matrix_inverse(t)

    pt = tf.cast(pt, tf.float32)
    pt = tf.concat(0, [tf.expand_dims(pt, 1), [[1.]]])
    pt = tf.squeeze(tf.matmul(t, pt))
    return tf.cast(tf.slice(pt, [0], [2]), tf.int32)


def _gauss(ul, br):
    kernal_width = 6*config.SIGMA + 1
    x = tf.cast(tf.range(0, kernal_width, 1), tf.float32)
    y = tf.expand_dims(x, 1)
    x0 = y0 = kernal_width / 2
    hm = tf.exp(-(tf.square(x-x0) + tf.square(y-y0))/(2 * config.SIGMA**2))
    
    ul1 = tf.maximum(0, -ul)
    br1 = tf.minimum(br, config.OUT_RES) - ul
    ul2 = tf.reverse(ul1, dims=[True])
    br2 = tf.reverse(br1, dims=[True])
    hm = tf.slice(hm, ul2, br2 - ul2)

    paddings = tf.concat(0, [[ul], [[config.OUT_RES, config.OUT_RES] - br]])
    paddings = tf.clip_by_value(paddings, 0, config.OUT_RES)
    paddings = tf.reverse(tf.transpose(paddings), dims=[True, False])
    hm = tf.pad(hm, paddings, mode="CONSTANT")
    return hm


def gaussian(pt, c, s):
    pt1 = transform(pt, c, s, config.OUT_RES, 0)
    ul = pt1 - 3 * config.SIGMA
    br = pt1 + 3 * config.SIGMA + 1
    return tf.cond(tf.reduce_any(tf.concat(0,
        [tf.slice(pt, [0], [1]) <= 0, ul >= config.OUT_RES, br < 0])),
        lambda: tf.zeros((config.OUT_RES, config.OUT_RES)), lambda: _gauss(ul, br))


def read_data(dset):
    assert dset in ('train', 'valid', 'test'), "invalid dataset"
    h5file = h5py.File(os.path.join(config.ANNOT_DIR, "%s.h5" %dset))
    image_files = [os.path.join(config.IMG_DIR, imgname) for imgname in h5file['imgname']]
    image_files = tf.convert_to_tensor(image_files, dtype=tf.string)
    centers = tf.convert_to_tensor(h5file['center'][:], dtype=tf.float32)
    scales = tf.convert_to_tensor(h5file['scale'][:], dtype=tf.float32)
    parts = tf.convert_to_tensor(h5file['part'][:], dtype=tf.float32)

    shuffle = True if dset == 'train' else False
    q = tf.train.slice_input_producer([image_files, centers, scales, parts], shuffle=shuffle)

    image = tf.read_file(q[0])
    image = tf.image.decode_jpeg(image, channels=3)

    # scale Augmentation
    s = q[2]
    if dset == 'train':
        rand = tf.clip_by_value(tf.squeeze(tf.random_normal([1])), -1, 1)
        s *= (config.SCALE_FACTOR * rand + 1)

    ul = transform([0, 0], q[1], s, config.INP_RES, 1)
    br = transform([config.INP_RES, config.INP_RES], q[1], s, config.INP_RES, 1)

    height, width, _ = tf.unpack(tf.shape(image))
    ul1 = tf.maximum(ul, 0)
    br1 = tf.minimum(br, [width, height])
    ul2 = tf.concat(0, [tf.reverse(ul1, dims=[True]), [0]])
    br2 = tf.concat(0, [tf.reverse(br1, dims=[True]), [3]])
    image = tf.slice(image, ul2, br2 - ul2)

    paddings = tf.concat(0, [[ul1 - ul], [br - br1]])
    paddings = tf.reverse(tf.transpose(paddings), dims=[True, False])
    paddings = tf.concat(0, [paddings, [[0, 0]]])
    image = tf.pad(image, paddings, mode="CONSTANT")

    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [config.INP_RES, config.INP_RES])
    image = tf.squeeze(image, [0])
    image = tf.reshape(image, tf.pack([config.INP_RES, config.INP_RES, 3]))

    pts = tf.cast(q[3], tf.int32)
    heatmaps = tf.map_fn(lambda x: gaussian(x, q[1], s), pts, dtype=tf.float32)
    heatmaps = tf.reshape(heatmaps, tf.pack([config.OUT_CHA, config.OUT_RES, config.OUT_RES]))

    # flip augmentation
    if dset == 'train':
        rand = tf.squeeze(tf.random_normal([1]))
        image = tf.cond(tf.less(rand, config.FLIP_CHANCE),
            lambda: tf.image.flip_left_right(image), lambda: image)
        heatmaps = tf.cond(tf.less(rand, config.FLIP_CHANCE),
            lambda: tf.reverse(tf.gather(heatmaps,
                config.PART_MATCH), dims=[False, False, True]),
            lambda: heatmaps)
        
    heatmaps = tf.transpose(heatmaps, [1, 2, 0])
    bs = config.TRAIN_BATCH_SIZE if dset == 'train' else config.VALID_BATCH_SIZE
    image_batch, hm_batch = tf.train.batch([image, heatmaps], batch_size=bs)
    return image_batch, hm_batch


if __name__ == "__main__":
    from scipy import misc
    import numpy as np
    import os
    with tf.Session() as sess:
        image_batch, hm_batch = read_data('train')
        init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        image_val, hm_val = sess.run([image_batch, hm_batch])
        coord.request_stop()
        coord.join(threads)
        sess.close()

    tmp_dir = os.path.join(config.WORK_DIR, "pictures")
    if tf.gfile.Exists(tmp_dir):
        tf.gfile.DeleteRecursively(tmp_dir)
    tf.gfile.MakeDirs(tmp_dir)

    for i, im in enumerate(image_val):
        misc.imsave(os.path.join(tmp_dir, "image%02d.jpg" %i), im)
    for i, hm in enumerate(hm_val):
        hm1 = np.sum(hm, axis = 2)
        misc.imsave(os.path.join(tmp_dir, "heatmap%02d.jpg" %i), hm1)

    print "%d preprocessed images and %d corresponding heatmaps are saved to %s." %(
            image_val.shape[0], hm_val.shape[0], tmp_dir)
