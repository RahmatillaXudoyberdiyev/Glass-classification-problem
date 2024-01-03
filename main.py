import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle

# Read the dataset
glass = pd.read_csv('glass.csv')

# Split the dataset into features (X) and target variable (y)
X = glass.drop('Type', axis=1)
y = glass['Type']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=33)
clf.fit(X_train, y_train)
accuracy_default = clf.score(X_test, y_test)

# Train a Random Forest classifier with specified parameters
clf = RandomForestClassifier(
    min_samples_leaf=2,
    criterion='entropy',
    class_weight='balanced_subsample',
    random_state=33
)
clf.fit(X_train, y_train)
accuracy_tuned = clf.score(X_test, y_test)

# Make predictions on the test set
y_preds = clf.predict(X_test)

# Evaluate the model using various metrics
accuracy = accuracy_score(y_test, y_preds)
recall = recall_score(y_test, y_preds, average='macro')
precision = precision_score(y_test, y_preds, average='macro')
f1 = f1_score(y_test, y_preds, average='macro')

print(f'Accuracy (Default): {accuracy_default}')
print(f'Accuracy (Tuned): {accuracy_tuned}')
print(f'Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}')

# Plot the scores
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(['Accuracy', 'Recall', 'Precision', 'F1 Score'], [accuracy, recall, precision, f1])
ax.set_ylabel('Score')
ax.set_title('Metrics')
plt.show()

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_preds)
print('Confusion Matrix:')
print(conf_matrix)

# Plot confusion matrix
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.show()

# Display classification report
print('Classification Report:')
print(classification_report(y_test, y_preds))

# Save the model using joblib
joblib.dump(clf, 'glass_clf.joblib')

# Save the model using pickle
with open('glass_clf.pickle', 'wb') as file:
    pickle.dump(clf, file)
