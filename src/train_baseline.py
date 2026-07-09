import sys
sys.path.append('.')
from load_data import load_UT_HAR_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report

data = load_UT_HAR_data('../data')

def extract_features(X):
    # X shape: (samples, 1, 250, 90) -> flatten time-window into simple stats per sample
    X = X.numpy().reshape(X.shape[0], 250, 90)
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    max_change = np.abs(np.diff(X, axis=1)).max(axis=1)
    features = np.concatenate([mean, std, max_change], axis=1)
    return features

X_train = extract_features(data['X_train'])
X_test = extract_features(data['X_test'])

y_train = (data['y_train'].numpy() == 1).astype(int)  # 1 = fall, 0 = everything else
y_test = (data['y_test'].numpy() == 1).astype(int)

clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall on fall class:", recall_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['not_fall', 'fall']))