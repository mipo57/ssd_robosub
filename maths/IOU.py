from structures.BoundingBox import BoundingBox


def intersection_over_union(box_1: BoundingBox, box_2: BoundingBox):
    xA = max(box_1.x1, box_2.x1)
    yA = max(box_1.y1, box_2.y1)
    xB = min(box_1.x2, box_2.x2)
    yB = min(box_1.y2, box_2.y2)

    intersecton_area = max(0.0, xB - xA) * max(0.0, yB - yA)

    box_1_area = box_1.w * box_1.h
    box_2_area = box_2.w * box_2.h

    union_area = box_1_area + box_2_area - intersecton_area

    return intersecton_area / union_area
