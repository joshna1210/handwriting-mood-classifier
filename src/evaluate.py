# src/evaluate.py
import argparse
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from train import load_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/raw')
    parser.add_argument('--model', default='models/handwriting_mood_rf.pkl')
    args = parser.parse_args()

    obj = joblib.load(args.model)
    clf = obj['model']

    X, y, _ = load_data(args.data_dir)
    y_pred = clf.predict(X)
    print(classification_report(y, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))
