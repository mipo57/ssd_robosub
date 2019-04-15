import unittest
from dataset_generation.data_feeder import TrainImageGenerator
import numpy as np
from postprocessing.interpret_labels import interpret_label
from neural_network.network_constants import FEATURE_MAPS

class TestImageGenerator(unittest.TestCase):
    def test_initalization(self):
        trainer = TrainImageGenerator(annotation_path="../../datasets/mini/annotations.csv",
                                      images_path="../../datasets/mini")
        self.assertEqual(trainer.batch_size, 8)
        self.assertEqual(trainer.num_samples, 32)
        self.assertEqual(trainer.num_batches, 32/8)

    def test_single_sample_generation(self):
        trainer = TrainImageGenerator(annotation_path="../../datasets/mini/annotations.csv",
                                      images_path="../../datasets/mini")
        x, y = trainer.generate_sample(0)

        self.assertEqual(np.shape(x), (300, 300, 3))
        self.assertEqual(len(y), len(FEATURE_MAPS))

        for layer, fm in zip(y, FEATURE_MAPS):
            self.assertEqual(fm.width, layer.shape[0])
            self.assertEqual(fm.height, layer.shape[1])
            self.assertEqual(len(fm.aspect_ratios), layer.shape[2])
            self.assertEqual(layer.shape[3], 5)

    def test_finds_bounding_boxes(self):
        trainer = TrainImageGenerator(annotation_path="../../datasets/micro/annotations.csv",
                                      images_path="../../datasets/micro")
        x, y = trainer.generate_sample(0)

        found_boxes = interpret_label(y, FEATURE_MAPS)

        self.assertTrue(len(found_boxes) > 0)

    def test_single_batch_generation(self):
        trainer = TrainImageGenerator(annotation_path="../../datasets/mini/annotations.csv",
                                      images_path="../../datasets/mini")

        x, y = trainer.generate_batch([0,1,2,3,4,5,6,7])

        self.assertEqual(np.shape(x), (8, 300, 300, 3))
        self.assertEqual(len(FEATURE_MAPS), len(y))

        for layer, fm in zip(y, FEATURE_MAPS):
            self.assertEqual(8, layer.shape[0])
            self.assertEqual(fm.width, layer.shape[1])
            self.assertEqual(fm.height, layer.shape[2])
            self.assertEqual(len(fm.aspect_ratios), layer.shape[3])
            self.assertEqual(layer.shape[4], 5)

    def test_get_batches_in_epoch(self):
        trainer = TrainImageGenerator(annotation_path="../../datasets/mini/annotations.csv",
                                      images_path="../../datasets/mini")

        counter = 0
        for x_batch, y_batch in trainer.get_batches_in_epoch():
            counter = counter + 1

            self.assertEqual((8, 300, 300, 3), np.shape(x_batch))
            self.assertEqual(len(y_batch), len(FEATURE_MAPS))

            for layer, fm in zip(y_batch, FEATURE_MAPS):
                self.assertEqual(8, layer.shape[0])
                self.assertEqual(fm.width, layer.shape[1])
                self.assertEqual(fm.height, layer.shape[2])
                self.assertEqual(len(fm.aspect_ratios), layer.shape[3])
                self.assertEqual(layer.shape[4], 5)

        self.assertEqual(counter, 4)
