from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import sys
import time
import json

dir_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(dir_path, "../src")
if module_path not in sys.path:
    sys.path.append(module_path)

from predictor import cnnsc

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="test data path", type=str)
parser.add_argument("--max_len", help="max sentence length", type=int, default=50)
parser.add_argument("--dropout", help="dropout parameter", type=float, default=0.5)
parser.add_argument("--vocab_size", help="vocabrary size", type=int, default=8001)
parser.add_argument("--batch_size", help="batch size", type=int, default=32)
parser.add_argument("--save_freq", help="save frequency", type=int, default=5000)
parser.add_argument("--progress_freq", help="view progressing frequency", type=int, default=100)
parser.add_argument("--summary_freq", help="view progressing frequency", type=int, default=1000)
parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)", default=1000000)
parser.add_argument("--output_dir", help="model output directory", type=str, default="test_learn")
a = parser.parse_args()


def create_model(inputs, labels):
    with tf.variable_scope("predictor"):
        outputs = cnnsc.create_model(inputs, a)
        loss = tf.reduce_mean(tf.abs(outputs - labels))

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([loss])

    tvars = tf.trainable_variables("predictor")
    opt = tf.train.AdamOptimizer(0.0002, 0.5)
    grads_and_vars = opt.compute_gradients(loss, var_list=tvars)
    train = opt.apply_gradients(grads_and_vars)
    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)
    
    return loss, tf.group(update_losses, incr_global_step, train), global_step, outputs


def main():
    sess = tf.InteractiveSession()

    def parse_tsv(line):
        words = line.decode('utf-8').strip().split(' ')
        words = np.array(words).astype(np.int32)
        return words[:-1], words[-1].astype(np.float32)

    dataset = tf.data.TextLineDataset(a.data) \
                .map(lambda x: tf.py_func(parse_tsv, [x], [tf.int32, tf.float32])) \
                .repeat() \
                .shuffle(100) \
                .batch(a.batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    batch[0].set_shape((a.batch_size, a.max_len))
    
    loss, train_op, global_step, predictions = create_model(*batch)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.global_variables('predictor'), max_to_keep=3)

    tf.summary.scalar("loss", loss)
    tf.summary.histogram("corrects", batch[1])
    tf.summary.histogram("predictions", predictions)
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(a.output_dir, sess.graph)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    start = time.time()

    for i in range(a.max_steps):
        fetches = {
            "loss": loss,
            "global_step": global_step,
            "train": train_op,
            "summary": summary
        }
        results = sess.run(fetches)

        if results["global_step"] % a.summary_freq == 0:
            print("recording summary")
            summary_writer.add_summary(results["summary"], results["global_step"])

        if results["global_step"] % a.progress_freq == 0:
            rate = (results["global_step"] + 1) * a.batch_size / (time.time() - start)
            print("progress  step %d  text/sec %0.1f" % (results["global_step"], rate))
            print("loss", results["loss"])

        if results["global_step"] % a.save_freq == 0:
            print("saving model")
            saver.save(sess, os.path.join(a.output_dir, "model"), global_step=global_step)


main()
