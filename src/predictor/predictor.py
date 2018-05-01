import os
import cnnsc
import json
import tensorflow as tf


class Args:
    pass


class Predictor():
    def __init__(self, model_path):
        a = Args()
        with open(os.path.join(model_path, "options.json"), "r") as f:
            for key, val in json.loads(f.read()).items():
                print("loaded", key, "=", val)
                setattr(a, key, val)
        a.dropout = 0.0
        self.build(a, model_path)

    def build(self, a, model_path):
        self.inputs = inputs = tf.placeholder(tf.int32, shape=(1, a.max_len))
        
        with tf.variable_scope("predictor"):
            self.outputs = cnnsc.create_model(inputs, a)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        loader = tf.train.Saver(tf.trainable_variables('predictor'))
        loader.restore(self.sess, tf.train.latest_checkpoint(model_path))

    def predict(self, inputs):
        return self.sess.run(self.outputs, feed_dict={self.inputs: inputs})


if __name__ == '__main__':
    import numpy as np
    p = Predictor("test_learn")
    inputs = np.array([89] * 50).reshape((1, 50))
    print(p.predict(inputs))
