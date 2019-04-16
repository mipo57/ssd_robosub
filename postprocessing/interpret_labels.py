import numpy as np
from structures.FeatureMaps import FeatureMap, AspectRatio
from structures.BoundingBox import BoundingBox


def calculate_bounding_box(label: np.ndarray, x, y, ar_id, feature_map: FeatureMap):
    ar = feature_map.aspect_ratios[ar_id]
    tile_box = BoundingBox(
        (x + 0.5) / feature_map.width,
        (y + 0.5) / feature_map.height,
        ar.scale * np.sqrt(ar.ratio),
        ar.scale / np.sqrt(ar.ratio))

    probability = label[x, y, ar_id, 0]
    x_coord = (label[x, y, ar_id, 1] + x + tile_box.x) / feature_map.width
    y_coord = (label[x, y, ar_id, 2] + y + tile_box.y) / feature_map.height
    width = (label[x, y, ar_id, 3] + 1.0) * tile_box.w
    height = (label[x, y, ar_id, 4] + 1.0) * tile_box.h

    return probability, BoundingBox(x_coord, y_coord, width, height)


def get_sample_from_batchx(xs, sample_id):
    return xs[sample_id, :, :, :]


def get_sample_from_batchy(ys, sample_id):
    outputs = []

    for y in ys:
        outputs.append(y[sample_id, :, :, :, :])

    return outputs


def interpret_label(labels: [np.ndarray], feature_maps: [FeatureMap]) -> [BoundingBox]:
    THRESHOLD = 0.5

    bounding_boxes = []

    for label, fm in zip(labels, feature_maps):
        for x in range(fm.width):
            for y in range(fm.height):
                for ar_id in range(len(fm.aspect_ratios)):
                    prob, bb = calculate_bounding_box(label, x, y, ar_id, fm)

                    if prob > THRESHOLD:
                        bounding_boxes.append(bb)

    return bounding_boxes




