from postprocessing.interpret_labels import interpret_label
from neural_network.network_constants import FEATURE_MAPS
from structures.BoundingBox import BoundingBox
import cv2


def visualize_prediction(input_image, labels):
    bboxes = interpret_label(labels, FEATURE_MAPS)
    img = input_image

    for bbox in bboxes:
        p1 = (int(300*bbox.x1), int(300 * bbox.y1))
        p2 = (int(300*bbox.x2), int(300 * bbox.y2))

        img = cv2.rectangle(img, p1, p2, (255, 0, 255))

    cv2.imshow("Results", img)
    cv2.waitKey(0)