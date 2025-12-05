# src/plot_confusion.py
import os, joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from train import load_data

os.makedirs("reports", exist_ok=True)
obj = joblib.load("models/handwriting_mood_rf.pkl")
clf = obj['model']
X, y, classes = load_data("data/raw")
y_pred = clf.predict(X)
cm = confusion_matrix(y, y_pred, labels=classes)

plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = range(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(int(cm[i, j]), 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
out = "reports/confusion_matrix.png"
plt.savefig(out)
print("Saved", out)
