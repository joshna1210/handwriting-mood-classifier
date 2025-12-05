# src/train.py
import os
import argparse
from glob import glob
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# local imports
from preprocess import preprocess_image
from features import extract_features

def load_data(data_dir, size=(128,128)):
    X, Y = [], []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for cl in classes:
        folder = os.path.join(data_dir, cl)
        for f in glob(os.path.join(folder, '*')):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            proc = preprocess_image(img, size=size)
            feat = extract_features(proc)
            X.append(feat)
            Y.append(cl)
    if len(X) == 0:
        raise RuntimeError("No images found in data_dir: " + data_dir)
    return np.vstack(X), np.array(Y), classes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw', help='Path to raw data')
    parser.add_argument('--model_out', default='models/handwriting_mood_rf.pkl')
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()

    X, y, classes = load_data(args.data_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    print("Train acc:", clf.score(X_train, y_train))
    print("Val acc:", clf.score(X_val, y_val))

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump({'model': clf, 'classes': classes}, args.model_out)
    print("Saved model to", args.model_out)
