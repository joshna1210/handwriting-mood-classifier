# src/cv_scores.py
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from train import load_data

X, y, _ = load_data("data/raw")
clf = RandomForestClassifier(n_estimators=200, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print("5-fold CV accuracy scores:", scores)
print("Mean accuracy: {:.4f}, Std: {:.4f}".format(scores.mean(), scores.std()))
