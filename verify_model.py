import pickle
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import os

MODEL_PATH = 'model.p'
DATA_PICKLE = 'data.pickle'

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PICKLE):
    print("Files missing.")
    exit()

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)['model']

with open(DATA_PICKLE, 'rb') as f:
    data_dict = pickle.load(f)
    X = np.asarray(data_dict['data'])
    y = np.asarray(data_dict['labels'])

y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)
print(f"Overall accuracy on the ENTIRE dataset: {acc*100:.2f}%")

# Check if any class has 0% accuracy
classes = model.classes_
cm = confusion_matrix(y, y_pred, labels=classes)

for i, cls in enumerate(classes):
    cls_total = np.sum(cm[i, :])
    cls_correct = cm[i, i]
    if cls_total > 0:
        cls_acc = cls_correct / cls_total
        if cls_acc < 0.8:
            print(f"!! CRITICAL: Class '{cls}' has low accuracy: {cls_acc*100:.2f}% ({cls_correct}/{cls_total})")
    else:
        print(f"!! WARNING: Class '{cls}' has NO samples in testing?")

# Check for labels with very few samples
unique, counts = np.unique(y, return_counts=True)
sample_counts = dict(zip(unique, counts))
for cls, count in sample_counts.items():
    if count < 50:
        print(f"!! WARNING: Class '{cls}' only has {count} samples.")
