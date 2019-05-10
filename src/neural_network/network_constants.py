from structures.FeatureMaps import FeatureMap, AspectRatio
import numpy as np

S_MIN = 0.2
S_MAX = 0.9


def scale(x):
    return S_MIN + (S_MAX - S_MIN) * x / 5


def scale_spec(x):
    return np.sqrt(scale(x) + scale(x + 1))


FEATURE_MAPS = [
    FeatureMap(38, 38,
               [AspectRatio(scale_spec(0), 1.0),
                AspectRatio(scale(0), 1.0),
                AspectRatio(scale(0), 2.0),
                AspectRatio(scale(0), 0.5)]),
    FeatureMap(19, 19,
               [AspectRatio(scale_spec(1), 1),
                AspectRatio(scale(1), 1),
                AspectRatio(scale(1), 2),
                AspectRatio(scale(1), 3),
                AspectRatio(scale(1), 1/2),
                AspectRatio(scale(1), 1/3)]),
    FeatureMap(10, 10,
               [AspectRatio(scale_spec(2), 1),
                AspectRatio(scale(2), 1),
                AspectRatio(scale(2), 2),
                AspectRatio(scale(2), 3),
                AspectRatio(scale(2), 1/2),
                AspectRatio(scale(2), 1/3)]),
    FeatureMap(5, 5,
               [AspectRatio(scale_spec(3), 1),
                AspectRatio(scale(3), 1),
                AspectRatio(scale(3), 2),
                AspectRatio(scale(3), 3),
                AspectRatio(scale(3), 1/2),
                AspectRatio(scale(3), 1/3)]),
    FeatureMap(3, 3,
               [AspectRatio(scale_spec(4), 1.0),
                AspectRatio(scale(4), 1.0),
                AspectRatio(scale(4), 2.0),
                AspectRatio(scale(4), 0.5)]),
    FeatureMap(1, 1,
               [AspectRatio(scale(5), 1.0),
                AspectRatio(scale(5), 2.0),
                AspectRatio(scale(5), 0.5)]),
]
