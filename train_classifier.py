import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

with open('hand_detective_data.pickle', 'rb') as f:
    data, labels = pickle.load(f)

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

score = accuracy_score(y_test, y_pred)
print(f'{score * 100:.2f}% of samples were classified correctly!')

with open('hand_detective_model.p', 'wb') as f:
    pickle.dump(model, f)
