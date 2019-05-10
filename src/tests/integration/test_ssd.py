import tensorflow as tf
import numpy as np
from neural_network.network_constants import FEATURE_MAPS
from neural_network.ssd import ssd_total_loss, ssd
import unittest
from dataset_generation.augmenter import NoAgumenter
from dataset_generation.data_feeder import TrainImageGenerator
from neural_network.ssd import simple_loss
from postprocessing.visualization import visualize_prediction
from postprocessing.interpret_labels import get_sample_from_batchy, get_sample_from_batchx


class TestSSDNetwork(unittest.TestCase):
    def test_dimensions(self):
        xs = tf.placeholder(tf.float32, (None, 300, 300, 3))
        ys = [tf.placeholder(tf.float32, (None, fm.width, fm.height, len(fm.aspect_ratios), 5)) for fm in FEATURE_MAPS]

        net = ssd(xs)

        for net_out, expected_out in zip(net,ys):
            self.assertEqual(net_out.shape.as_list(), expected_out.shape.as_list())

        tf.reset_default_graph()

    def test_learing(self):
        BATCH_SIZE = 2

        xs = tf.placeholder(tf.float32, (None, 300, 300, 3))
        ys = [tf.placeholder(tf.float32, (None, fm.width, fm.height, len(fm.aspect_ratios), 5)) for fm in FEATURE_MAPS]

        net = ssd(xs)
        loss = ssd_total_loss(ys, net)
        minimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

        inputs = np.random.uniform(0, 1, size=(BATCH_SIZE, 300, 300, 3))
        expected_list = [
            np.ones((BATCH_SIZE, fm.width, fm.height, len(fm.aspect_ratios), 5), np.float) * 0.8 for fm in FEATURE_MAPS
        ]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(500):
                calc_loss, _ = sess.run([loss, minimizer], {xs: inputs, **dict(zip(ys, expected_list))})
                print(f"{i}: {calc_loss}")

            predictions = sess.run(net, {xs: inputs})

            for prediciton_fm, expected_fm in zip(predictions, expected_list):
                self.assertTrue(np.all(np.abs(prediciton_fm - expected_fm) < 0.1))

if __name__ == "__main__":
    unittest.main()
