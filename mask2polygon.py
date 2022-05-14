import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon

# trimap = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)
trimap = cv2.imread("D:\\Downloads\\annotations (1).tar\\annotations\\trimaps\yorkshire_terrier_89.png",
                    cv2.IMREAD_GRAYSCALE)
trimap[trimap == 2] = 0
trimap[trimap == 1] = 255
trimap[trimap == 3] = 255
ret, thresh = cv2.threshold(trimap, 127, 255, 0)
contours, hierarchy = cv2.findContours(trimap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
testcontour = contours[1]

contours = map(np.squeeze, contours)  # removing redundant dimensions
polygons = list(map(Polygon, contours))  # converting to Polygons
a = polygons[0]


def str2xy(string):
    segs = string.strip(" ").split(" ")
    return float(segs[0]), float(segs[1])


xys = list(map(str2xy, a.wkt.strip("POLYGON ()").split(",")))
print(a.bounds)
