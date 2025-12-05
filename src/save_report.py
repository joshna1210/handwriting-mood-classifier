# src/save_report.py
import os
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from train import load_data

def main():
    os.makedirs("reports", exist_ok=True)

    model_path = "models/handwriting_mood_rf.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train first with src/train.py")

    # load trained model (we saved a dict {'model': clf, 'classes': classes})
    obj = joblib.load(model_path)
    clf = obj.get('model')
    if clf is None:
        raise ValueError("Loaded object does not contain 'model' key. Check how model was saved.")

    # load dataset (same loader used for training)
    X, y, classes = load_data("data/raw")

    # predict and produce reports
    y_pred = clf.predict(X)
    rep = classification_report(y, y_pred, digits=4, target_names=classes)
    cm = confusion_matrix(y, y_pred, labels=classes)

    # print to console
    print(rep)
    print("Confusion matrix:")
    print(cm)

    # save textual report
    with open("reports/classification_report.txt", "w", encoding="utf-8") as f:
        f.write("Classification report (labels: " + ", ".join(classes) + ")\n\n")
        f.write(rep)

    # save confusion matrix as CSV (rows = true, cols = pred)
    import numpy as np
    np.savetxt("reports/confusion_matrix.csv", cm.astype(int), fmt="%d", delimiter=",")

    print("Saved reports/classification_report.txt and reports/confusion_matrix.csv")

if __name__ == "__main__":
    main()
