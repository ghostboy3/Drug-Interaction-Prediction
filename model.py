import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = pd.read_csv('DDI_data.csv')

# Preprocess data
le_name = LabelEncoder()
le_interaction = LabelEncoder()

data['drug1_id'] = le_name.fit_transform(data['drug1_name'])
data['drug2_id'] = le_name.fit_transform(data['drug2_name'])
data['interaction_type'] = le_interaction.fit_transform(data['interaction_type'])

# Split data
X = data[['drug1_id', 'drug2_id']]
y = data['interaction_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = Sequential()
model.add(Dense(128, input_dim=2, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model
y_pred = (model.predict(X_test) > 0.5).astype("int32")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the model
model.save('drug_interaction_model.h5')

# Save the LabelEncoders
import pickle
with open('le_name.pkl', 'wb') as file:
    pickle.dump(le_name, file)
with open('le_interaction.pkl', 'wb') as file:
    pickle.dump(le_interaction, file)

