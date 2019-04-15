import tensorflow as tf
import numpy as np
from neural_network.network_constants import FEATURE_MAPS
from neural_network.ssd import ssd_total_loss, ssd
import unittest
from dataset_generation.augmenter import NoAgumenter
from dataset_generation.data_feeder import TrainImageGenerator
from neural_network.ssd import simple_loss

class TestSSDNetwork(unittest.TestCase):
    def test_dimensions(self):
        xs = tf.placeholder(tf.float32, (None, 300, 300, 3))
        ys = [tf.placeholder(tf.float32, (None, fm.width, fm.height, len(fm.aspect_ratios), 5)) for fm in FEATURE_MAPS]

        net = ssd(xs)

        for net_out, expected_out in zip(net,ys):
            self.assertEqual(net_out.shape.as_list(), expected_out.shape.as_list())

        tf.reset_default_graph()

    def test_learing(self):
        gen = TrainImageGenerator("../../datasets/micro/annotations.csv", "../../datasets/micro", batch_size=1,
                                  augumenter=NoAgumenter())

        xs = tf.placeholder(tf.float32, (None, 300, 300, 3))
        ys = [tf.placeholder(tf.float32, (None, fm.width, fm.height, len(fm.aspect_ratios), 5)) for fm in FEATURE_MAPS]

        net = ssd(xs)
        loss = ssd_total_loss(ys, net)
        minimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('../../graphs', sess.graph)

            inputs = np.random.uniform(0, 1.0, (1, 300, 300, 3))

            expected_list = []
            for fm in FEATURE_MAPS:
                expected = np.zeros((1, fm.width, fm.height, len(fm.aspect_ratios), 5))
                expected[:, :, :2, :, 0] = 1.0
                expected_list.append(expected)

            for i in range(1000):
                calc_loss, _ = sess.run([loss, minimizer], {xs: inputs, **dict(zip(ys, expected_list))})
                print(f"{i}: {calc_loss}")

            predictions = sess.run(net, {xs: inputs})

            for predicted, expected, fm in zip(predictions, expected_list, FEATURE_MAPS):
                for x in range(fm.width):
                    for y in range(fm.height):
                        for ar in range(len(fm.aspect_ratios)):
                            print(f"Expected: {expected[0,x,y,ar,0]}, got: {predicted[0,x,y,ar,0]} ")
                            self.assertTrue(abs(expected[0,x,y,ar,0] - predicted[0,x,y,ar,0]) < 0.2)

                            if predicted[0, x, y, ar, 0] > 0.5:
                                self.assertTrue(abs(expected[0, x, y, ar, 1] - predicted[0, x, y, ar, 1]) < 0.2)
                                self.assertTrue(abs(expected[0, x, y, ar, 2] - predicted[0, x, y, ar, 2]) < 0.2)
                                self.assertTrue(abs(expected[0, x, y, ar, 3] - predicted[0, x, y, ar, 3]) < 0.2)
                                self.assertTrue(abs(expected[0, x, y, ar, 4] - predicted[0, x, y, ar, 4]) < 0.2)

                            print(f"{x}, {y} {ar} ok!")

if __name__ == "__main__":
    unittest.main()
