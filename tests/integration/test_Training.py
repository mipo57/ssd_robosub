import unittest
from training.Trainer import Trainer
import os
from shutil import rmtree


class TestImageGenerator(unittest.TestCase):
    def test_trainer(self):
        trainer = Trainer(batch_size=1, max_epoch=2, num_early_stopping=None,
                          annotations_path="../../datasets/micro/annotations.csv",
                          images_path="../../datasets/micro", model_path="../../models/trainer_test/model")

        trainer.fit()

        self.assertTrue(os.path.exists("../../models/trainer_test"))
        self.assertTrue(os.path.exists("../../models/trainer_test/checkpoint"))

        self.assertTrue(os.path.exists("../../models/trainer_test/model-epoch0.data-00000-of-00001"))
        self.assertTrue(os.path.exists("../../models/trainer_test/model-epoch0.index"))
        self.assertTrue(os.path.exists("../../models/trainer_test/model-epoch0.meta"))

        self.assertTrue(os.path.exists("../../models/trainer_test/model-epoch1.data-00000-of-00001"))
        self.assertTrue(os.path.exists("../../models/trainer_test/model-epoch1.index"))
        self.assertTrue(os.path.exists("../../models/trainer_test/model-epoch1.meta"))

        self.assertTrue(os.path.exists("../../models/trainer_test/model-final.data-00000-of-00001"))
        self.assertTrue(os.path.exists("../../models/trainer_test/model-final.index"))
        self.assertTrue(os.path.exists("../../models/trainer_test/model-final.meta"))

        rmtree("../../models/trainer_test")


if __name__ == "__main__":
    unittest.main()
