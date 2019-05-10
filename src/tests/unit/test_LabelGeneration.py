import unittest
from structures.BoundingBox import BoundingBox
import numpy as np
from dataset_generation.label_generation import generate_label, generate_ssd_labels
from structures.FeatureMaps import AspectRatio, FeatureMap
from neural_network.network_constants import FEATURE_MAPS


class LabelGeneratorTest(unittest.TestCase):
    @staticmethod
    def test_simple_label_generation():
        box = BoundingBox(0.5, 0.5, 1/3.0, 1/3.0)
        aspect_ratios = [AspectRatio(scale=1/3, ratio=1.0)]
        feature_map = FeatureMap(width=3, height=3, aspect_ratios=aspect_ratios)

        label = generate_label(box, feature_map)

        assert label.shape == (3, 3, 1, 5)
        expected_label = np.zeros((3, 3, 1, 5))
        expected_label[1, 1, 0, :] = [1.0, 0, 0, 0, 0]

        assert np.all(label == expected_label)

    @staticmethod
    def test_multiple_aspect_ratios():
        box = BoundingBox(0.5, 0.5, 1 / 3.0, 1 / 3.0)
        aspect_ratios = [AspectRatio(scale=1/3, ratio=1.0),
                         AspectRatio(scale=1/3, ratio=1.0),
                         AspectRatio(scale=1/3, ratio=1.0),
                         AspectRatio(scale=1/3, ratio=1.0)]
        feature_map = FeatureMap(width=3, height=3, aspect_ratios=aspect_ratios)

        label = generate_label(box, feature_map)

        assert label.shape == (3, 3, 4, 5)

        expected_label = np.zeros((3, 3, 4, 5))
        expected_label[1, 1, 0, :] = [1.0, 0, 0, 0, 0]
        expected_label[1, 1, 1, :] = [1.0, 0, 0, 0, 0]
        expected_label[1, 1, 2, :] = [1.0, 0, 0, 0, 0]
        expected_label[1, 1, 3, :] = [1.0, 0, 0, 0, 0]

        assert np.all(label == expected_label)

    @staticmethod
    def test_scale():
        box = BoundingBox(0.5, 0.5, 1/3.0, 1/3.0)
        aspect_ratios = [AspectRatio(scale=0.8*1/3, ratio=1.0)]
        feature_map = FeatureMap(width=3, height=3, aspect_ratios=aspect_ratios)

        label = generate_label(box, feature_map)

        assert label.shape == (3, 3, 1, 5)

        expected_label = np.zeros((3, 3, 1, 5))
        expected_label[1, 1, 0, :] = [1.0, 0, 0, 0.25, 0.25]

        assert np.all(label == expected_label)

    @staticmethod
    def test_ratio():
        box = BoundingBox(0.5, 0.5, 1 / 3.0, 1 / 3.0)
        aspect_ratios = [AspectRatio(scale=1/3, ratio=2.0)]
        feature_map = FeatureMap(width=3, height=3, aspect_ratios=aspect_ratios)

        label = generate_label(box, feature_map)

        assert label.shape == (3, 3, 1, 5)

        expected_label = np.zeros((3, 3, 1, 5))
        expected_label[1, 1, 0, :] = [1.0, 0, 0, 1/np.sqrt(2) - 1.0, np.sqrt(2) - 1.0]

        assert np.all(label == expected_label)

    @staticmethod
    def test_position_delta():
        box = BoundingBox(1.6/3.0, 1.45/3.0, 1/3.0, 1/3.0)
        aspect_ratios = [AspectRatio(scale=1/3, ratio=1.0)]
        feature_map = FeatureMap(width=3, height=3, aspect_ratios=aspect_ratios)

        label = generate_label(box, feature_map)

        assert label.shape == (3, 3, 1, 5)

        expected_label = np.zeros((3, 3, 1, 5))
        expected_label[1, 1, 0, :] = [1.0, 0.1, -0.05, 0, 0]

        assert np.allclose(label, expected_label)


class TESTGenerateSsdLabels(unittest.TestCase):
    def test_output_dimensions(self):
        box = BoundingBox(1.6/3.0, 1.45/3.0, 1/3.0, 1/3.0)
        labels = generate_ssd_labels(box, FEATURE_MAPS)

        self.assertEqual(len(FEATURE_MAPS), len(labels))

        for label, fm in zip(labels, FEATURE_MAPS):
            self.assertEqual(fm.width, label.shape[0])
            self.assertEqual(fm.height, label.shape[1])
            self.assertEqual(len(fm.aspect_ratios), label.shape[2])
            self.assertEqual(label.shape[3], 5)


if __name__ == "__main__":
    unittest.main()
