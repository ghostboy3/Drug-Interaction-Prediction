import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load dataset
data = pd.read_csv('DDI_data.csv')

# Check for missing values
print(data.isnull().sum())

# Visualize the interaction types
sns.countplot(data['interaction_type'])
plt.show()

# Print the class distribution
print(data['interaction_type'].value_counts())

# Preprocess data
le_name = LabelEncoder()
le_interaction = LabelEncoder()

# Fit LabelEncoders on the entire dataset
data['drug1_id'] = le_name.fit_transform(data['drug1_name'])
data['drug2_id'] = le_name.fit_transform(data['drug2_name'])
data['interaction_type'] = le_interaction.fit_transform(data['interaction_type'])

# Additional features
data['id_diff'] = abs(data['drug1_id'] - data['drug2_id'])
data['id_sum'] = data['drug1_id'] + data['drug2_id']

# Separate majority and minority classes
majority = data[data['interaction_type'] == 0]
minority = data[data['interaction_type'] == 1]

# Upsample minority class
minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

# Combine majority class with upsampled minority class
data_balanced = pd.concat([majority, minority_upsampled])

# Split data
X = data_balanced[['drug1_id', 'drug2_id', 'id_diff', 'id_sum']]
y = data_balanced['interaction_type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(classification_report(y_test, y_pred))

# Save the model
with open('drug_interaction_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the LabelEncoders
with open('le_name.pkl', 'wb') as file:
    pickle.dump(le_name, file)
with open('le_interaction.pkl', 'wb') as file:
    pickle.dump(le_interaction, file)

# Test the model with 10 examples from the test set
test_samples = X_test.sample(10, random_state=42)
test_indices = test_samples.index  # Get the indices of the sampled test data
test_labels = y_test.loc[test_indices]  # Select labels based on these indices
test_predictions = model.predict(test_samples)

# Debug: Print lengths of arrays to ensure they are the same
# print(f"Length of test_samples: {len(test_samples)}")
# print(f"Length of test_labels: {len(test_labels)}")
# print(f"Length of test_predictions: {len(test_predictions)}")

# # Ensure all arrays are of the same length
# assert len(test_samples) == len(test_labels) == len(test_predictions)

# # Create a DataFrame to compare predictions with actual values
# comparison_df = pd.DataFrame({
#     'drug1_id': test_samples['drug1_id'],
#     'drug2_id': test_samples['drug2_id'],
#     'actual_interaction': le_interaction.inverse_transform(test_labels),
#     'predicted_interaction': le_interaction.inverse_transform(test_predictions)
# })

# # Add drug names for better understanding
# comparison_df['drug1_name'] = le_name.inverse_transform(comparison_df['drug1_id'])
# comparison_df['drug2_name'] = le_name.inverse_transform(comparison_df['drug2_id'])

# print("Comparison of predictions and actual values for 10 samples:")
# print(comparison_df[['drug1_name', 'drug2_name', 'actual_interaction', 'predicted_interaction']])

# # Add drug names for better understanding
# comparison_df['drug1_name'] = le_name.inverse_transform(comparison_df['drug1_id'])
# comparison_df['drug2_name'] = le_name.inverse_transform(comparison_df['drug2_id'])

# print("Comparison of predictions and actual values for 10 samples:")
# print(comparison_df[['drug1_name', 'drug2_name', 'actual_interaction', 'predicted_interaction']])


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import resample
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle

# # Load dataset
# data = pd.read_csv('DDI_data.csv')

# # Check for missing values
# print(data.isnull().sum())

# # Visualize the interaction types
# sns.countplot(data['interaction_type'])
# plt.show()

# # Print the class distribution
# print(data['interaction_type'].value_counts())

# # Preprocess data
# le_name = LabelEncoder()
# le_interaction = LabelEncoder()

# # Fit LabelEncoders on the entire dataset
# data['drug1_id'] = le_name.fit_transform(data['drug1_name'])
# data['drug2_id'] = le_name.fit_transform(data['drug2_name'])
# data['interaction_type'] = le_interaction.fit_transform(data['interaction_type'])

# # Additional features
# data['id_diff'] = abs(data['drug1_id'] - data['drug2_id'])
# data['id_sum'] = data['drug1_id'] + data['drug2_id']

# # Separate majority and minority classes
# majority = data[data['interaction_type'] == 0]
# minority = data[data['interaction_type'] == 1]

# # Upsample minority class
# minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

# # Combine majority class with upsampled minority class
# data_balanced = pd.concat([majority, minority_upsampled])

# # Split data
# X = data_balanced[['drug1_id', 'drug2_id', 'id_diff', 'id_sum']]
# y = data_balanced['interaction_type']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the Random Forest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Evaluate the model
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
# print(f"F1 Score: {f1}")
# print(classification_report(y_test, y_pred))

# # Save the model
# with open('drug_interaction_model.pkl', 'wb') as file:
#     pickle.dump(model, file)

# # Save the LabelEncoders
# with open('le_name.pkl', 'wb') as file:
#     pickle.dump(le_name, file)
# with open('le_interaction.pkl', 'wb') as file:
#     pickle.dump(le_interaction, file)
#----------------------------------------------

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.utils import resample
# from sklearn.metrics import accuracy_score, f1_score, classification_report
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# # import tensorflow_decision_forests as tfdf
# from tensorflow.keras.layers import Dense, Dropout
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
# from sklearn.ensemble import RandomForestClassifier


# # Load dataset
# data = pd.read_csv('DDI_data.csv')

# # Check for missing values
# print(data.isnull().sum())

# # Visualize the interaction types
# sns.countplot(data['interaction_type'])
# plt.show()

# # Print the class distribution
# print(data['interaction_type'].value_counts())

# # Preprocess data
# le_name = LabelEncoder()
# le_interaction = LabelEncoder()

# # Fit LabelEncoders on the entire dataset
# data['drug1_id'] = le_name.fit_transform(data['drug1_name'])
# data['drug2_id'] = le_name.fit_transform(data['drug2_name'])
# data['interaction_type'] = le_interaction.fit_transform(data['interaction_type'])

# # Additional features
# data['id_diff'] = abs(data['drug1_id'] - data['drug2_id'])
# data['id_sum'] = data['drug1_id'] + data['drug2_id']

# # Separate majority and minority classes
# majority = data[data['interaction_type'] == 0]
# minority = data[data['interaction_type'] == 1]

# # Upsample minority class
# minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)

# # Combine majority class with upsampled minority class
# data_balanced = pd.concat([majority, minority_upsampled])

# # Split data
# X = data_balanced[['drug1_id', 'drug2_id', 'id_diff', 'id_sum']]
# y = data_balanced['interaction_type']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model
# model = Sequential()
# model.add(Dense(256, input_dim=X.shape[1], activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# # Evaluate the model
# y_pred = (model.predict(X_test) > 0.5).astype("int32")
# accuracy = accuracy_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")
# print(f"F1 Score: {f1}")
# print(classification_report(y_test, y_pred))

# # Plot training history
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')
# plt.show()

# # Save the model
# model.save('drug_interaction_model.h5')

# # Save the LabelEncoders
# with open('le_name.pkl', 'wb') as file:
#     pickle.dump(le_name, file)
# with open('le_interaction.pkl', 'wb') as file:
#     pickle.dump(le_interaction, file)
