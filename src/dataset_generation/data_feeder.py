import os
import numpy as np
import pandas as pd
import cv2
import h5py
from structures.BoundingBox import BoundingBox
from neural_network.network_constants import FEATURE_MAPS
from dataset_generation.augmenter import Augmenter
from dataset_generation.label_generation import generate_ssd_labels

class TrainImageGenerator:
    batch_size = 10

    def __init__(self, annotation_path="../unity_data/annotations.csv",
                 images_path="../unity_data",
                 images_width=300,
                 images_height=300,
                 batch_size=8,
                 augumenter=Augmenter(32)):

        self.annotations_path = annotation_path
        self.images_path = images_path
        self.input_image_width = images_width
        self.input_image_height = images_height
        self.batch_size = batch_size
        self.augmenter = augumenter

        self.annotations_table: pd.DataFrame = None
        self.num_samples: int = None
        self.num_batches: int = None

        if not os.path.exists(self.annotations_path):
            raise RuntimeError(f"Couldn't find annotation file in location {self.annotations_path}")

        if not os.path.exists(self.annotations_path):
            raise RuntimeError(f"Couldn't find image folder in location {self.images_path}")

        self._load_annotations()

        if len(self.annotations_table) == 0:
            raise RuntimeError(f"No annotations found in file {self.annotations_path}")

    def _load_annotations(self):
        self.annotations_table = pd.read_csv(self.annotations_path)
        self.num_samples = len(self.annotations_table)
        self.num_batches = self.num_samples // self.batch_size

    def generate_sample(self, index):
        image_path = f'{self.images_path}/{self.annotations_table["filename"][index]}'
        x = self.annotations_table["x"][index]
        y = self.annotations_table["y"][index]
        w = self.annotations_table["w"][index]
        h = self.annotations_table["h"][index]
        p = self.annotations_table["p"][index]

        image = cv2.imread(image_path)
        img_c = image
        bounding_box = BoundingBox(x, y, w, h)

        image, bounding_box, p = self.augmenter.augment_image(image, bounding_box, p)
        image = image / 255.0

        bounding_box = bounding_box.normalize(image.shape[1], image.shape[0])
        image = cv2.resize(image, (self.input_image_width, self.input_image_height))

        label = generate_ssd_labels(bounding_box, FEATURE_MAPS)

        return np.asarray(image), label

    def generate_batch(self, indexes):
        batch_x = []
        outputs_y = [[] for _ in FEATURE_MAPS]

        for i in indexes:
            x, y = self.generate_sample(i)

            batch_x.append(x)

            for i in range(len(y)):
                new_y = np.expand_dims(y[i], 0)
                outputs_y[i].append(new_y)

        batch_y = []
        for i in range(len(outputs_y)):
            batch_y.append(np.concatenate(outputs_y[i], 0))

        return np.asarray(batch_x), batch_y

    def get_batches_in_epoch(self):
        indexes = np.arange(self.num_samples)
        np.random.shuffle(indexes)
        indexes = indexes[:self.num_batches*self.batch_size]

        for b in range(self.num_batches):
            batch_x, batch_y = self.generate_batch(indexes[b * self.batch_size:b * self.batch_size + self.batch_size])
            yield batch_x, batch_y

    def generate_to_file(self, path: str):
        with h5py.File(path, "w") as hf:
            hf.create_dataset("x_train", maxshape=(None, 300, 300, 3), shape=(0, 300, 300, 3))
            hf.create_dataset("y_train", maxshape=(None, 8732, 5), shape=(0, 8732, 5))

            for i, (x_batch, y_batch) in enumerate(self.get_batches_in_epoch()):
                hf["x_train"].resize(hf["x_train"].shape[0] + self.batch_size, axis=0)
                hf["x_train"][-self.batch_size:] = x_batch

                hf["y_train"].resize(hf["y_train"].shape[0] + self.batch_size, axis=0)
                hf["y_train"][-self.batch_size:] = y_batch
                print(f"Saved batch {i+1} out of {self.num_batches}")

    @staticmethod
    def get_batches_from_file(path, batch_size):
        with h5py.File(path) as hf:
            num_batches = hf["x_train"].shape[0] // batch_size

            for b in range(num_batches):
                x_batch = hf["x_train"][b*batch_size:(b+1)*batch_size]
                y_batch = hf["y_train"][b * batch_size:(b + 1) * batch_size]

                yield x_batch, y_batch

