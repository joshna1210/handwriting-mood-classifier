# src/preprocess.py
import cv2
import numpy as np

def preprocess_image(gray_img, size=(128,128)):
    # Resize
    img = cv2.resize(gray_img, size, interpolation=cv2.INTER_AREA)
    # Blur + adaptive threshold (invert so ink=255)
    blur = cv2.GaussianBlur(img, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)
    # Morph open to remove noise
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return opening

# quick CLI test
if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    if not p:
        print("Usage: python src/preprocess.py /path/to/image")
        raise SystemExit(1)
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Cannot read image:", p)
        raise SystemExit(1)
    out = preprocess_image(img)
    out_path = p.replace(".", "_proc.")
    cv2.imwrite(out_path, out)
    print("Wrote:", out_path)
