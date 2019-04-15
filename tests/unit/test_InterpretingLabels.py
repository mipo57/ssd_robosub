import unittest
from structures.BoundingBox import BoundingBox
import numpy as np
from dataset_generation.label_generation import generate_label, generate_ssd_labels
from postprocessing.interpret_labels import interpret_label
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


if __name__ == "__main__":
    unittest.main()
