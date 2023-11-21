import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Function to evaluate a model on test data and return performance metrics
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # ROC curve for binary classification
    if len(set(y_test)) == 2:
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc_score = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()

        return accuracy, report, cm, auc_score

    return accuracy, report, cm, None

# Load and preprocess the test data
test_data_folder = '~/test_data/'
test_files = os.listdir(test_data_folder)

all_y_test_rf = []
all_y_pred_rf = []

all_y_test_lr = []
all_y_pred_lr = []

# Specify the expected column names
expected_columns = ['name', 'taxID', 'taxRank', 'genomeSize', 'numReads', 'numUniqueReads',
                    'abundance', 'genus', 'presence', 'sim_abundance']

# Evaluate RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
for test_file_name in test_files:
    if test_file_name.endswith('.txt'):
        test_file_path = os.path.join(test_data_folder, test_file_name)

        # Load test data
        test_data = pd.read_csv(test_file_path, names=expected_columns, header=0)
        test_data.dropna(inplace=True)

        # Drop non-numeric columns
        test_data_numeric = test_data.select_dtypes(include='number')

        # Split the test data into features (X_test) and labels (y_test)
        X_test_rf = test_data_numeric.drop(['presence', 'sim_abundance'], axis=1)
        y_test_rf = test_data_numeric['presence']

        # Model evaluation on the test data
        y_pred_rf = rf_model.fit(X, y).predict(X_test_rf)
        all_y_test_rf.extend(y_test_rf)
        all_y_pred_rf.extend(y_pred_rf)

# Evaluate LogisticRegression
lr_model = LogisticRegression(random_state=42)
for test_file_name in test_files:
    if test_file_name.endswith('.txt'):
        test_file_path = os.path.join(test_data_folder, test_file_name)

        # Load test data
        test_data = pd.read_csv(test_file_path, names=expected_columns, header=0)
        test_data.dropna(inplace=True)

        # Drop non-numeric columns
        test_data_numeric = test_data.select_dtypes(include='number')

        # Split the test data into features (X_test) and labels (y_test)
        X_test_lr = test_data_numeric.drop(['presence', 'sim_abundance'], axis=1)
        y_test_lr = test_data_numeric['presence']

        # Model evaluation on the test data
        y_pred_lr = lr_model.fit(X, y).predict(X_test_lr)
        all_y_test_lr.extend(y_test_lr)
        all_y_pred_lr.extend(y_pred_lr)

# Calculate overall performance metrics for RandomForestClassifier
overall_accuracy_rf = accuracy_score(all_y_test_rf, all_y_pred_rf)
overall_report_rf = classification_report(all_y_test_rf, all_y_pred_rf)
overall_cm_rf = confusion_matrix(all_y_test_rf, all_y_pred_rf)

# Calculate overall performance metrics for LogisticRegression
overall_accuracy_lr = accuracy_score(all_y_test_lr, all_y_pred_lr)
overall_report_lr = classification_report(all_y_test_lr, all_y_pred_lr)
overall_cm_lr = confusion_matrix(all_y_test_lr, all_y_pred_lr)

# Print overall performance metrics
print("\nOverall Performance Metrics for RandomForestClassifier:")
print(f"Overall Accuracy: {overall_accuracy_rf}")
print(f"Overall Classification Report:\n{overall_report_rf}")
print(f"Overall Confusion Matrix:\n{overall_cm_rf}")

# Print overall performance metrics
print("\nOverall Performance Metrics for LogisticRegression:")
print(f"Overall Accuracy: {overall_accuracy_lr}")
print(f"Overall Classification Report:\n{overall_report_lr}")
print(f"Overall Confusion Matrix:\n{overall_cm_lr}")

# Compare ROC curves for both models
evaluate_model(rf_model, X_test_rf, y_test_rf)
evaluate_model(lr_model, X_test_lr, y_test_lr)
