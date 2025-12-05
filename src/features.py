# src/features.py
import numpy as np
import cv2

def extract_features(binary_img):
    """
    Input: binary_img with foreground=255 (uint8)
    Output: 1D numpy array of features
    """
    h, w = binary_img.shape
    feats = []

    # 1. Pixel density
    pixel_density = binary_img.sum() / (255.0 * h * w)
    feats.append(pixel_density)

    # 2. Bounding box & aspect ratio
    coords = cv2.findNonZero(binary_img)
    if coords is None:
        return np.zeros(13, dtype=float)
    x,y,ww,hh = cv2.boundingRect(coords)
    feats.append((ww*hh) / (h*w))
    feats.append(ww / (hh + 1e-6))

    # 3. Hu moments (log)
    moments = cv2.moments(binary_img)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-9)
    feats.extend(hu_log.tolist())

    # 4. Contour count
    contours, _ = cv2.findContours(binary_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    feats.append(len(contours))

    # 5. Skeleton-like score (distance transform)
    dist = cv2.distanceTransform(255 - binary_img, cv2.DIST_L2, 5)
    feats.append(dist.sum() / (h * w))

    # 6. Projection variance
    hor = (binary_img.sum(axis=1) / 255.0)
    ver = (binary_img.sum(axis=0) / 255.0)
    feats.append(np.var(hor))
    feats.append(np.var(ver))

    return np.array(feats, dtype=float)
