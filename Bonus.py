# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.cluster import KMeans
# from sklearn.metrics import confusion_matrix, f1_score, classification_report
# import seaborn as sns
# import matplotlib.pyplot as plt
# #
# # # Load the dataset
# # file_path = 'Datasets/MCSDatasetNEXTCONLab.csv'   # Replace with correct file path
# # data = pd.read_csv(file_path)
# #
# # # Define feature columns and target
# # features = ['Latitude', 'Longitude', 'Day', 'Hour', 'Minute', 'Duration',
# #             'RemainingTime', 'Resources', 'Coverage', 'OnPeakHours', 'GridNumber']
# # target = 'Ligitimacy'
# #
# # # Extract features and target
# # X = data[features]
# # y = data[target]
# #
# # # Apply K-means clustering to label legitimate and fake clusters
# # kmeans = KMeans(n_clusters=5, random_state=42)
# # data['Cluster'] = kmeans.fit_predict(X)
# #
# # # Identify legitimate clusters (assume these are clusters where the majority class is legitimate)
# # cluster_legitimacy = data.groupby('Cluster')[target].mean()
# # legitimate_clusters = cluster_legitimacy[cluster_legitimacy > 0.5].index
# #
# # # Filter dataset to include only legitimate clusters
# # legitimate_data = data[data['Cluster'].isin(legitimate_clusters)]
# #
# # # Train-test split for supervised and hybrid approaches
# # X_train_legitimate, X_test_legitimate, y_train_legitimate, y_test_legitimate = train_test_split(
# #     legitimate_data[features], legitimate_data[target], test_size=0.25, random_state=42)
# #
# # X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.25, random_state=42)
# #
# # # Train a supervised model on the full dataset (purely supervised approach)
# # rf_supervised = RandomForestClassifier(random_state=42)
# # rf_supervised.fit(X_train_full, y_train_full)
# # y_pred_supervised = rf_supervised.predict(X_test_full)
# #
# # # Train a supervised model on the legitimate dataset (hybrid approach)
# # rf_hybrid = RandomForestClassifier(random_state=42)
# # rf_hybrid.fit(X_train_legitimate, y_train_legitimate)
# # y_pred_hybrid = rf_hybrid.predict(X_test_legitimate)
# #
# # # Evaluate the purely supervised model
# # conf_matrix_supervised = confusion_matrix(y_test_full, y_pred_supervised)
# # f1_supervised = f1_score(y_test_full, y_pred_supervised, average='weighted')
# # print("Purely Supervised Approach")
# # print("\nConfusion Matrix:\n", conf_matrix_supervised)
# # print("\nClassification Report:\n", classification_report(y_test_full, y_pred_supervised))
# #
# # # Plot the confusion matrix for the supervised approach
# # plt.figure(figsize=(6, 5))
# # sns.heatmap(conf_matrix_supervised, annot=True, fmt='d', cmap='Blues', cbar=False)
# # plt.title(f"Supervised Confusion Matrix\nF1 Score: {f1_supervised:.3f}")
# # plt.xlabel("Predicted Label")
# # plt.ylabel("True Label")
# # plt.show()
# #
# # # Evaluate the hybrid model
# # conf_matrix_hybrid = confusion_matrix(y_test_legitimate, y_pred_hybrid)
# # f1_hybrid = f1_score(y_test_legitimate, y_pred_hybrid, average='weighted')
# # print("Hybrid Approach (Supervised + Legitimate Clusters)")
# # print("\nConfusion Matrix:\n", conf_matrix_hybrid)
# # print("\nClassification Report:\n", classification_report(y_test_legitimate, y_pred_hybrid))
# #
# # # Plot the confusion matrix for the hybrid approach
# # plt.figure(figsize=(6, 5))
# # sns.heatmap(conf_matrix_hybrid, annot=True, fmt='d', cmap='Greens', cbar=False)
# # plt.title(f"Hybrid Confusion Matrix\nF1 Score: {f1_hybrid:.3f}")
# # plt.xlabel("Predicted Label")
# # plt.ylabel("True Label")
# # plt.show()
# #
# # # Print comparison of F1 scores
# # print(f"Purely Supervised F1 Score: {f1_supervised:.3f}")
# # print(f"Hybrid Approach F1 Score: {f1_hybrid:.3f}")
# # from sklearn.metrics import precision_score, recall_score, accuracy_score
# #
# # # Purely Supervised Metrics
# # precision_supervised = precision_score(y_test_full, y_pred_supervised, average='weighted')
# # recall_supervised = recall_score(y_test_full, y_pred_supervised, average='weighted')
# # accuracy_supervised = accuracy_score(y_test_full, y_pred_supervised)
# #
# # # Hybrid Metrics
# # precision_hybrid = precision_score(y_test_legitimate, y_pred_hybrid, average='weighted')
# # recall_hybrid = recall_score(y_test_legitimate, y_pred_hybrid, average='weighted')
# # accuracy_hybrid = accuracy_score(y_test_legitimate, y_pred_hybrid)
# #
# # # Print the metrics for comparison
# # print("=== Purely Supervised Metrics ===")
# # print(f"Precision: {precision_supervised:.3f}")
# # print(f"Recall: {recall_supervised:.3f}")
# # print(f"Accuracy: {accuracy_supervised:.3f}")
# # print(f"F1 Score: {f1_supervised:.3f}\n")
# #
# # print("=== Hybrid Approach Metrics ===")
# # print(f"Precision: {precision_hybrid:.3f}")
# # print(f"Recall: {recall_hybrid:.3f}")
# # print(f"Accuracy: {accuracy_hybrid:.3f}")
# # print(f"F1 Score: {f1_hybrid:.3f}\n")
# #
# # # Visualize the comparison
# # labels = ['Precision', 'Recall', 'Accuracy', 'F1 Score']
# # supervised_scores = [precision_supervised, recall_supervised, accuracy_supervised, f1_supervised]
# # hybrid_scores = [precision_hybrid, recall_hybrid, accuracy_hybrid, f1_hybrid]
# #
# # x = np.arange(len(labels))
# # width = 0.35
# #
# # plt.figure(figsize=(10, 6))
# # plt.bar(x - width/2, supervised_scores, width, label='Purely Supervised', color='blue')
# # plt.bar(x + width/2, hybrid_scores, width, label='Hybrid Approach', color='green')
# # plt.xticks(x, labels)
# # plt.ylabel('Scores')
# # plt.title('Comparison of Metrics: Purely Supervised vs Hybrid Approach')
# # plt.legend()
# # plt.show()
#
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, precision_recall_curve, auc, confusion_matrix
# from imblearn.over_sampling import SMOTE
# from sklearn.cluster import KMeans
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load dataset
#  # Replace with correct file path
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, precision_recall_curve, auc, confusion_matrix
# from imblearn.over_sampling import SMOTE
# from sklearn.cluster import KMeans
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, precision_recall_curve, auc, confusion_matrix
# from imblearn.over_sampling import SMOTE
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Load dataset
# file_path = 'Datasets/MCSDatasetNEXTCONLab.csv'  # Replace with your correct file path
# data = pd.read_csv(file_path)
#
# # Step 1: Filter the dataset for legitimate=0
# data_filtered = data[data['Ligitimacy'] == 0]
#
# # Check if the filtered dataset has only one class
# if data_filtered['Ligitimacy'].nunique() == 1:
#     # Include a small subset of legitimate=1 data to avoid single-class errors
#     legitimate_1_subset = data[data['Ligitimacy'] == 1].sample(frac=0.1, random_state=42)  # Use 10% of legitimate=1
#     data_filtered = pd.concat([data_filtered, legitimate_1_subset])
#
# # Verify class distribution
# print("Filtered Dataset Class Distribution:")
# print(data_filtered['Ligitimacy'].value_counts())
#
# # Step 2: Define features and target
# features = ['Latitude', 'Longitude', 'Day', 'Hour', 'Minute', 'Duration',
#             'RemainingTime', 'Resources', 'Coverage', 'OnPeakHours', 'GridNumber']
# X = data_filtered[features]
# y = data_filtered['Ligitimacy']
#
# # Step 3: Purely Supervised Model with SMOTE
# # Apply SMOTE to balance the dataset
# smote = SMOTE(random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)
#
# # Split the resampled data into train-test sets
# X_train_supervised, X_test_supervised, y_train_supervised, y_test_supervised = train_test_split(
#     X_resampled, y_resampled, test_size=0.25, random_state=42
# )
#
# # Train the purely supervised model
# rf_supervised = RandomForestClassifier(random_state=42)
# rf_supervised.fit(X_train_supervised, y_train_supervised)
# y_pred_supervised = rf_supervised.predict(X_test_supervised)
#
# # Step 4: Hybrid Approach with Clustering
# # Apply K-means clustering to find legitimate=0 clusters
# kmeans = KMeans(n_clusters=3, random_state=42)
# data_filtered['Cluster'] = kmeans.fit_predict(X)
#
# # Focus on legitimate clusters
# cluster_sizes = data_filtered['Cluster'].value_counts()
# print("\nCluster sizes:")
# print(cluster_sizes)
#
# # Use clusters for training the hybrid model
# X_hybrid = data_filtered[features]
# y_hybrid = data_filtered['Ligitimacy']
# X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = train_test_split(
#     X_hybrid, y_hybrid, test_size=0.25, random_state=42
# )
#
# rf_hybrid = RandomForestClassifier(random_state=42)
# rf_hybrid.fit(X_train_hybrid, y_train_hybrid)
# y_pred_hybrid = rf_hybrid.predict(X_test_hybrid)
#
# # Step 5: Evaluation
# # Compute Precision-Recall curves and AUC for both approaches
# precision_supervised, recall_supervised, _ = precision_recall_curve(
#     y_test_supervised, rf_supervised.predict_proba(X_test_supervised)[:, 1]
# )
# precision_hybrid, recall_hybrid, _ = precision_recall_curve(
#     y_test_hybrid, rf_hybrid.predict_proba(X_test_hybrid)[:, 1]
# )
#
# auc_supervised = auc(recall_supervised, precision_supervised)
# auc_hybrid = auc(recall_hybrid, precision_hybrid)
#
# # Plot Precision-Recall curves
# plt.figure(figsize=(10, 6))
# plt.plot(recall_supervised, precision_supervised, label=f'Supervised (PR AUC={auc_supervised:.3f})', color='blue')
# plt.plot(recall_hybrid, precision_hybrid, label=f'Hybrid (PR AUC={auc_hybrid:.3f})', color='green')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve for Legitimate=0 Class')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # Confusion Matrices
# cm_supervised = confusion_matrix(y_test_supervised, y_pred_supervised)
# cm_hybrid = confusion_matrix(y_test_hybrid, y_pred_hybrid)
#
# print("=== Purely Supervised Confusion Matrix ===")
# print(cm_supervised)
# print("\n=== Hybrid Confusion Matrix ===")
# print(cm_hybrid)
#
# # Classification Reports
# print("\n=== Purely Supervised Classification Report ===")
# print(classification_report(y_test_supervised, y_pred_supervised, target_names=['Fake (0)', 'Legitimate (1)']))
#
# print("\n=== Hybrid Classification Report ===")
# print(classification_report(y_test_hybrid, y_pred_hybrid, target_names=['Fake (0)', 'Legitimate (1)']))


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = 'Datasets/MCSDatasetNEXTCONLab.csv'  # Update with correct path
data = pd.read_csv(file_path)

# Define features and target
features = ['Latitude', 'Longitude', 'Day', 'Hour', 'Minute', 'Duration',
            'RemainingTime', 'Resources', 'Coverage', 'OnPeakHours', 'GridNumber']
target = 'Ligitimacy'

X = data[features]
y = data[target]

# Step 1: Clustering Step
# Apply K-Means to separate legitimate and fake tasks into clusters
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Latitude', y='Longitude', hue='Cluster', palette='viridis', alpha=0.7)
plt.title("Clustering of Tasks")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
plt.legend(title="Cluster")
plt.show()

# Separate legitimate and fake clusters for Hybrid Approach
legitimate_clusters = data[data['Ligitimacy'] == 1]
fake_clusters = data[data['Ligitimacy'] == 0]

# Balance the dataset by using fake and legitimate clusters
balanced_data = pd.concat([legitimate_clusters, fake_clusters])
X_balanced = balanced_data[features]
y_balanced = balanced_data[target]

# Step 2: Hybrid Approach
# Split balanced dataset into train-test sets
X_train_hybrid, X_test_hybrid, y_train_hybrid, y_test_hybrid = train_test_split(
    X_balanced, y_balanced, test_size=0.25, random_state=42
)

# Train a supervised model on the balanced data
rf_hybrid = RandomForestClassifier(random_state=42)
rf_hybrid.fit(X_train_hybrid, y_train_hybrid)
y_pred_hybrid = rf_hybrid.predict(X_test_hybrid)

# Step 3: Purely Supervised Model
# Apply SMOTE to balance the original dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split resampled dataset into train-test sets
X_train_supervised, X_test_supervised, y_train_supervised, y_test_supervised = train_test_split(
    X_resampled, y_resampled, test_size=0.25, random_state=42
)

# Train a supervised model on the resampled data
rf_supervised = RandomForestClassifier(random_state=42)
rf_supervised.fit(X_train_supervised, y_train_supervised)
y_pred_supervised = rf_supervised.predict(X_test_supervised)

# Step 4: Evaluation
# Confusion matrices
cm_hybrid = confusion_matrix(y_test_hybrid, y_pred_hybrid)
cm_supervised = confusion_matrix(y_test_supervised, y_pred_supervised)

# Print confusion matrices
print("=== Hybrid Approach Confusion Matrix ===")
print(cm_hybrid)

print("\n=== Purely Supervised Confusion Matrix ===")
print(cm_supervised)

# Classification reports
print("\n=== Hybrid Approach Classification Report ===")
print(classification_report(y_test_hybrid, y_pred_hybrid, target_names=['Fake (0)', 'Legitimate (1)']))

print("\n=== Purely Supervised Classification Report ===")
print(classification_report(y_test_supervised, y_pred_supervised, target_names=['Fake (0)', 'Legitimate (1)']))
print('1- Reduced FN and FP in the Hybrid approach.\n 2- Better precision and recall for legitimate=0 and legitimate=1 tasks.')

# Plot confusion matrices
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.heatmap(cm_hybrid, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title("Hybrid Approach Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(1, 2, 2)
sns.heatmap(cm_supervised, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Purely Supervised Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()
