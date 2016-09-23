from hg_generic import inference
from image_reader import read_data
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from tqdm import trange, tqdm
import config

with tf.Session() as sess:
    is_training = tf.placeholder(tf.bool)
    train_images, train_heatmaps = read_data('train')
    outs = inference(train_images, is_training)
    mses = [tf.reduce_mean(tf.square(out - train_heatmaps)) for out in outs]
    loss = tf.reduce_sum(mses)
    train_op  = tf.train.AdamOptimizer().minimize(loss)

    saver = tf.train.Saver(tf.all_variables())

    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_dir = os.path.join(config.WORK_DIR, "model")
    if tf.gfile.Exists(train_dir):
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)
    checkpoint_path = os.path.join(train_dir, 'model.ckpt')

    batches_per_epoch = config.NUM_TRAIN / config.TRAIN_BATCH_SIZE
    for epoch in range(config.N_EPOCHS):
        for idx in trange(batches_per_epoch, desc="Epoch%d" %epoch):
            _, loss_val = sess.run([train_op, loss], feed_dict={is_training: True})
            assert not np.isnan(loss_val), 'Model diverged with loss = NaN'
            tqdm.write('%s loss: %.6f' %(datetime.now(), loss_val))
        saver.save(sess, checkpoint_path)

    coord.request_stop()
    coord.join(threads)
    sess.close()
