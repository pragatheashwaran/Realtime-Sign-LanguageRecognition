import pickle
import os

MODEL_PATH = 'model.p'
DATA_DIR = './data'

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
    model_classes = set(model.classes_)
    print(f"Model classes: {sorted(list(model_classes))}")
else:
    print("Model file not found.")

if os.path.exists(DATA_DIR):
    data_folders = set([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    print(f"Data folders: {sorted(list(data_folders))}")
    
    missing_in_model = data_folders - model_classes
    extra_in_model = model_classes - data_folders
    
    if missing_in_model:
        print(f"Missing in model (need training): {missing_in_model}")
    if extra_in_model:
        print(f"Extra in model (refers to non-existent data): {extra_in_model}")
    if not missing_in_model and not extra_in_model:
        print("Model classes match data folders perfectly.")
else:
    print("Data directory not found.")
