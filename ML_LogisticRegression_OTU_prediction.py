import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load and preprocess the data for each population (Training Data)
data_folder = '~/training_data/'
all_files = os.listdir(data_folder)

# Initialize empty lists to store data from all populations
all_data = []

# Specify the expected column names
expected_columns = ['name', 'taxID', 'taxRank', 'genomeSize', 'numReads', 'numUniqueReads',
                    'abundance', 'genus', 'presence', 'sim_abundance']

all_y_test = []
all_y_pred = []

for file_name in all_files:
    if file_name.endswith('.txt'):
        file_path = os.path.join(data_folder, file_name)

        # Load data for the current population
        print(f'Processing file: {file_path}')
        try:
            # Read the comma-separated file with expected column names
            data = pd.read_csv(file_path, names=expected_columns, header=0)
        except pd.errors.ParserError as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        # Handle missing values if necessary
        data.dropna(inplace=True)

        # Drop non-numeric columns
        data_numeric = data.select_dtypes(include='number')

        # Append data from the current population to the list
        all_data.append(data_numeric)

# Combine data from all populations for training
if all_data:
    combined_data = pd.concat(all_data, axis=0)

    if not combined_data.empty:
        # Step 3: Model selection
        model = LogisticRegression(random_state=42)

        # Split the combined data into features (X) and labels (y)
        X = combined_data.drop(['presence', 'sim_abundance'], axis=1)
        y = combined_data['presence']

        # Step 4: Model training
        model.fit(X, y)

        # Step 5: Model evaluation on the test data
        test_data_folder = '~/test_data/'
        test_files = os.listdir(test_data_folder)

        for test_file_name in test_files:
            if test_file_name.endswith('.txt'):
                test_file_path = os.path.join(test_data_folder, test_file_name)

                # Load test data
                test_data = pd.read_csv(test_file_path, names=expected_columns, header=0)
                test_data.dropna(inplace=True)

                # Drop non-numeric columns
                test_data_numeric = test_data.select_dtypes(include='number')

                # Split the test data into features (X_test) and labels (y_test)
                X_test = test_data_numeric.drop(['presence', 'sim_abundance'], axis=1)
                y_test = test_data_numeric['presence']

                # Step 6: Model evaluation on the test data
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                # Accumulate true labels and predicted labels for overall metrics
                all_y_test.extend(y_test)
                all_y_pred.extend(y_pred)

                # Print evaluation metrics for each test file
                print(f"\nPerformance Metrics for file {test_file_path}:")
                print(f"Accuracy: {accuracy}")
                print(f"Classification Report:\n{report}")
                print(f"Confusion Matrix:\n{cm}")

        # Calculate overall performance metrics
        overall_accuracy = accuracy_score(all_y_test, all_y_pred)
        overall_report = classification_report(all_y_test, all_y_pred)
        overall_cm = confusion_matrix(all_y_test, all_y_pred)

        # Print overall performance metrics
        print("\nOverall Performance Metrics:")
        print(f"Overall Accuracy: {overall_accuracy}")
        print(f"Overall Classification Report:\n{overall_report}")
        print(f"Overall Confusion Matrix:\n{overall_cm}")

        # Plot overall confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(overall_cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"])
        plt.title("Overall Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()

    else:
        print("No valid data after preprocessing. Check your data format and preprocessing steps.")
else:
    print("No valid TXT files found in the training data folder.")
