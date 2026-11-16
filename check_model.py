import pickle
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']
if hasattr(model, 'classes_'):
    print("Classes in model:", model.classes_)
else:
    print("Model classes not found directly")
