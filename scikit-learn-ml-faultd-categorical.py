## Load in Data
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Billybobstuff/Machine-Learning-Model-Practice/refs/heads/main/Line_fault_dataset.csv')
df.head()
print(df);

## **Data Preparation**
#preprocess the data
# Create Fault_Type categorical column from A, B, C, G
def get_fault_type(row):
    faults = []
    if row['A']: faults.append('A')
    if row['B']: faults.append('B')
    if row['C']: faults.append('C')
    if row['G']: faults.append('G')
    return 'Normal' if not faults else ''.join(faults)

df['Fault_Type'] = df.apply(get_fault_type, axis=1)

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['Fault_Type'])
X = df.drop(columns=['Fault_Type']).values

## Data splitting
import sklearn;
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

#X_train will have 80% of the data. This is used to build the model
#X_test will have 20% of the data. This is used as random data for traing to use.

## Model Building
## Random Forest
## Training the model
from sklearn.ensemble import RandomForestClassifier # if y is quantitative build regression model. If categorical build classification model

rf = RandomForestClassifier(max_depth=2, random_state=100)
rf.fit(X_train, y_train)

y_rf_train_pred = rf.predict(X_train)
y_rf_test_pred = rf.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random Forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
rf_results.columns = ['Method', 'Training MSE', 'Test MSE', 'Train R2', 'Test R2']
print(rf_results);


#ML Evaluation Metrics
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
# Evaluate the model
fault_classes = le.classes_
accuracy = accuracy_score(y_test, y_rf_test_pred)
precision = precision_score(y_test, y_rf_test_pred, average='weighted')
recall = recall_score(y_test, y_rf_test_pred, average='weighted')
f1 = f1_score(y_test, y_rf_test_pred, average='weighted')
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Confusion Matrix
cm = confusion_matrix(y_test, y_rf_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=fault_classes)
disp.plot()
plt.title('Confusion Matrix - Random Forest')
plt.xticks(rotation=45)
plt.show()
# Classification Report
print('\nClassification Report:')
print(classification_report(y_test, y_rf_test_pred, target_names=fault_classes))


