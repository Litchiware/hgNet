from eval import accuracies
from image_reader import read_data
from hg_generic import inference
import tensorflow as tf
import numpy as np
import config
import os
from tqdm import trange, tqdm
from datetime import datetime


with tf.Session() as sess:
    is_training = tf.placeholder(tf.bool)
    images, heatmaps = read_data('valid')
    outs = inference(images, is_training)
    accs = accuracies(outs[-1], heatmaps)

    model_dir = os.path.join(config.WORK_DIR, "model")
    saver = tf.train.Saver(tf.all_variables())
    ckpt = tf.train.get_checkpoint_state(model_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    mean_accs = np.zeros(config.OUT_CHA)
    batches = config.NUM_VALID / config.VALID_BATCH_SIZE

    headers = "          | " + " | ".join(map(lambda x: "%6s" %x, config.PART_NAMES))
    dashes = "----------|-" + "-|-".join(config.OUT_CHA * ["------"])
    for idx in trange(batches):
        accs_val = sess.run(accs, feed_dict={is_training: False})
        mean_accs = (idx * mean_accs + accs_val) / (idx + 1)

        batch_acc_str = "Batch Acc | " + " | ".join(map(lambda x: "%.4f" %x, accs_val))
        mean_acc_str = " Mean Acc | " + " | ".join(map(lambda x: "%.4f" %x, mean_accs))
        tqdm.write("\n".join((headers, dashes, batch_acc_str, mean_acc_str, "\n")))

    coord.request_stop()
    coord.join(threads)
    sess.close()
