import pickle
try:
    data_dict = pickle.load(open('data.pickle', 'rb'))
    data = data_dict['data']
    lengths = set(len(item) for item in data)
    print(f"Number of samples: {len(data)}")
    print(f"Feature lengths found: {lengths}")
except Exception as e:
    print(f"Error: {e}")
