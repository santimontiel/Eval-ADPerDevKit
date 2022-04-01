import numpy as np
from shapely.geometry import polygon

def iou3d(box1, box2):

    Z_AXIS = 1

    # Extract z-coordinates
    z1 = box1[:, Z_AXIS]
    z2 = box2[:, Z_AXIS]

    # Calculate the minimum and maximum z-coordinates
    z1_min = float(np.min(z1))
    z1_max = float(np.max(z1))
    z2_min = float(np.min(z2))
    z2_max = float(np.max(z2))

    # print(z1_min)

    # Calculate the overlap of the z-coordinates
    z_overlap = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))

    # Delete z-coordinate
    box1 = np.delete(box1, Z_AXIS, axis=0)
    box2 = np.delete(box2, Z_AXIS, axis=0)

    # Create a Polygon with first 4 points
    poly1 = polygon.Polygon(box1[:4])
    poly2 = polygon.Polygon(box2[:4])

    # Calculate the intersection area
    intersection = poly1.intersection(poly2).area
    if intersection == 0:
        return 0
    
    # Calculate the intersection 3D
    intersection3D = intersection * z_overlap

    # height of the first box
    h1 = float(np.max(box1[:, Z_AXIS])) - float(np.min(box1[:, Z_AXIS]))
    h2 = float(np.max(box2[:, Z_AXIS])) - float(np.min(box2[:, Z_AXIS]))

    iou_3d = abs(intersection3D / (poly1.area * h1 + poly2.area * h2 - intersection3D))
    return iou_3d