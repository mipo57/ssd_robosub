import unittest
from structures.BoundingBox import BoundingBox
import numpy as np
from dataset_generation.label_generation import generate_label, generate_ssd_labels
from postprocessing.interpret_labels import interpret_label,get_sample_from_batchx, get_sample_from_batchy
from neural_network.network_constants import FEATURE_MAPS


class TESTInterpretingLabels(unittest.TestCase):
    def test_interpreted_bbs_matches_set_ones(self):
        box = BoundingBox(1.6/3.0, 1.45/3.0, 1/3.0, 1/3.0)
        labels = generate_ssd_labels(box, )
        interpreted_labels = interpret_label(labels, FEATURE_MAPS)

        assert len(interpreted_labels) > 0
        
        for box_found in interpreted_labels:
            assert np.isclose(box_found.x, box.x)
            assert np.isclose(box_found.y, box.y)
            assert np.isclose(box_found.w, box.w)
            assert np.isclose(box_found.h, box.h)


class TESTExtractingLabels(unittest.TestCase):
    def test_extracting_batchx(self):
        xs = np.random.uniform(0, 1, (5, 300, 300, 3))
        single_img = get_sample_from_batchx(xs, 2)

        self.assertTrue(np.all(xs[2, :, :, :] == single_img))

    def test_extracing_batchy(self):
        batchy = []
        expected_ys = []

        for fm in FEATURE_MAPS:
            fm_y = np.random.uniform(0, 1, (5, fm.width, fm.height, len(fm.aspect_ratios), 5))
            fm_y[1,:,:,:,:] = 3.0

            expected_ys.append(fm_y[1,:,:,:,:])

        sampley = get_sample_from_batchy(batchy, 1)

        for sample, expected in zip(sampley, expected_ys):
            self.assertTrue(np.all(sample == expected))


if __name__ == "__main__":
    unittest.main()
