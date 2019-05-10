import tensorflow as tf
from neural_network.network_constants import FEATURE_MAPS
from neural_network.ssd import ssd_total_loss, ssd
import unittest
from dataset_generation.augmenter import NoAgumenter
from dataset_generation.data_feeder import TrainImageGenerator
from postprocessing.visualization import visualize_prediction
from postprocessing.interpret_labels import get_sample_from_batchy, get_sample_from_batchx


class TestOverfitingSSDNetwork(unittest.TestCase):
    def test_overfitting(self):
        gen = TrainImageGenerator("../../datasets/micro/annotations.csv", "../../datasets/micro", batch_size=1,
                                  augumenter=NoAgumenter())

        xs = tf.placeholder(tf.float32, (None, 300, 300, 3))
        ys = [tf.placeholder(tf.float32, (None, fm.width, fm.height, len(fm.aspect_ratios), 5)) for fm in FEATURE_MAPS]

        net = ssd(xs)
        loss = ssd_total_loss(ys, net)
        minimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

        inputs, expected_list = gen.generate_batch([0])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('../../graphs', sess.graph)

            for i in range(500):
                calc_loss, _ = sess.run([loss, minimizer], {xs: inputs, **dict(zip(ys, expected_list))})
                print(f"{i}: {calc_loss}")

            predictions = sess.run(net, {xs: inputs})

            visualize_prediction(
                get_sample_from_batchx(inputs, 0),
                get_sample_from_batchy(predictions, 0)
            )

if __name__ == "__main__":
    unittest.main()
