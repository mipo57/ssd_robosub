from structures.BoundingBox import BoundingBox
import numpy as np
from maths.IOU import intersection_over_union
from structures.FeatureMaps import AspectRatio, FeatureMap
from neural_network.network_constants import FEATURE_MAPS


def generate_label(target_box: BoundingBox, feature_map: FeatureMap):
    output = np.zeros((feature_map.width, feature_map.height, len(feature_map.aspect_ratios), 5))

    for x in range(feature_map.width):
        for y in range(feature_map.height):
            for ar_id, ar in enumerate(feature_map.aspect_ratios):
                tile_box = BoundingBox(
                    (x + 0.5) / feature_map.width,
                    (y + 0.5) / feature_map.height,
                    ar.scale * np.sqrt(ar.ratio),
                    ar.scale / np.sqrt(ar.ratio))

                iou = intersection_over_union(target_box, tile_box)

                if iou >= 0.5:
                    width_scale = target_box.w / tile_box.w - 1.0
                    height_scale = target_box.h / tile_box.h - 1.0
                    delta_x = feature_map.width * target_box.x - x - tile_box.x
                    delta_y = feature_map.height * target_box.y - y - tile_box.y

                    output[x, y, ar_id, :] = [1.0, delta_x, delta_y, width_scale, height_scale]

    return output


def generate_ssd_labels(target_box, feature_maps=FEATURE_MAPS):
    fm_labels = []

    for fm in feature_maps:
        label = generate_label(target_box, fm)
        fm_labels.append(label)

    return fm_labels




