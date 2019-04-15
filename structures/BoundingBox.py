class BoundingBox:
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.x1 = x - w/2
        self.x2 = x + w/2
        self.y1 = y - h/2
        self.y2 = y + h/2

        self.p1 = (self.x1, self.y1)
        self.p2 = (self.x2, self.y2)

    @staticmethod
    def from_points(x1, y1, x2, y2):
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        return BoundingBox(x, y, w, h)

    def normalize(self, img_widh, img_height):
        return BoundingBox(
            self.x / img_widh,
            self.y / img_height,
            self.w / img_widh,
            self.h / img_height)
