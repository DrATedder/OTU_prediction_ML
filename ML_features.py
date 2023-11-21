### Which features are driving prediction performance?

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import os
import matplotlib.pyplot as plt

# Load and preprocess the training data
data_folder = '~/training_data/'
all_files = os.listdir(data_folder)

expected_columns = ['name', 'taxID', 'taxRank', 'genomeSize', 'numReads', 'numUniqueReads',
                    'abundance', 'genus', 'presence', 'sim_abundance']

all_data = []

for file_name in all_files:
    if file_name.endswith('.txt'):
        file_path = os.path.join(data_folder, file_name)

        try:
            data = pd.read_csv(file_path, names=expected_columns, header=0)
        except pd.errors.ParserError as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        data.dropna(inplace=True)
        data_numeric = data.select_dtypes(include='number')
        all_data.append(data_numeric)

if all_data:
    combined_data = pd.concat(all_data, axis=0)

    if not combined_data.empty:
        # Model selection and training
        model = RandomForestClassifier(random_state=42)
        X = combined_data.drop(['presence', 'sim_abundance'], axis=1)
        y = combined_data['presence']
        model.fit(X, y)

        # Extract feature importances
        feature_importances = model.feature_importances_

        # Create a DataFrame to display feature importances
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title('Feature Importances')
        plt.show()

else:
    print("No valid data after preprocessing. Check your data format and preprocessing steps.")
