import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))
except FileNotFoundError:
    print("Error: data.pickle not found. Please run 'python create_dataset.py' first.")
    exit()

if not data_dict['data'] or len(data_dict['data']) == 0:
    print("Error: No data found in data.pickle. Please make sure you have collected images in 'data/' and then run 'create_dataset.py'.")
    exit()

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Dataset loaded: {len(data)} samples with {len(data[0])} features.")

# Split data
try:
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True)
except Exception as e:
    print(f"Error during data split: {e}")
    exit()

# Train model
print("Training classifier... please wait.")
# Increase n_estimators for better performance and stability
# Use class_weight='balanced' in case of any slight imbalances
model = RandomForestClassifier(n_estimators=200, min_samples_split=2, random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print('Training Complete! {:.2f}% of samples were classified correctly.'.format(score * 100))

# Detailed assessment of the signs the user mentioned as "confusing"
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_test, y_predict))

# Save model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
print("Model saved to 'model.p'. You can now run the app.")
print("Tip: Use '.\\venv\\Scripts\\python.exe app.py' to ensure the correct NumPy version is used.")
