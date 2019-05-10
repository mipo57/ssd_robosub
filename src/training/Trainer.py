from neural_network.ssd import ssd, ssd_total_loss
from dataset_generation.data_feeder import TrainImageGenerator
from dataset_generation.augmenter import NoAgumenter
import tensorflow as tf
from neural_network.network_constants import FEATURE_MAPS


class Trainer:
    def __init__(self, batch_size=8, alpha=1e-4, images_path="../../datasets/micro",
                 annotations_path="../../datasets/micro/annotations.csv", model_path="models/model.ckpt", max_epoch=10,
                 num_early_stopping=3, save_every_epoch=True):
        self.batch_size = batch_size
        self.alpha = alpha
        self.images_path = images_path
        self.annotations_path = annotations_path
        self.output_model_path = model_path
        self.max_epoch = max_epoch
        self.save_every_epoch = save_every_epoch

    def fit(self):
        gen = TrainImageGenerator(self.annotations_path, self.images_path,
                                  batch_size=self.batch_size, augumenter=NoAgumenter())

        xs = tf.placeholder(tf.float32, (None, 300, 300, 3))
        ys = [tf.placeholder(tf.float32, (None, fm.width, fm.height, len(fm.aspect_ratios), 5)) for fm in FEATURE_MAPS]

        net = ssd(xs)
        loss = ssd_total_loss(ys, net)
        minimizer = tf.train.AdamOptimizer(self.alpha).minimize(loss)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.max_epoch):
                for batch_x, batch_y in gen.get_batches_in_epoch():
                    calc_loss, _ = sess.run([loss, minimizer], {xs: batch_x, **dict(zip(ys, batch_y))})
                    print(f"{i}: {calc_loss}")

                if self.save_every_epoch:
                    saver.save(sess, self.output_model_path + f"-epoch{i}")

            saver.save(sess, self.output_model_path + "-final")
