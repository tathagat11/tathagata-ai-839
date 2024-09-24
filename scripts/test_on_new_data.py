import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.inspection import PartialDependenceDisplay
import mlflow.sklearn
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from src.tathagata_ai_839.pipelines.data_processing.nodes import preprocess_data

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    return preprocess_data(data)

def evaluate_model(model, X, y, dataset_name):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary', zero_division=1)
    recall = recall_score(y, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y, y_pred, average='binary', zero_division=1)
    
    print(f"{dataset_name} Data Scores: ---------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return y_pred

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'data/09_data_exploration/confusion_matrix_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_feature_importance(model, X, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances - {title}")
    plt.bar(range(X.shape[1]), importances[indices])
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f'data/09_data_exploration/feature_importance_{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_partial_dependence(model, X, title):
    features_to_plot = range(min(5, X.shape[1]))  # Plot top 5 important features
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(model, X, features_to_plot, ax=ax)
    plt.title(f'Partial Dependence Plots - {title}')
    plt.tight_layout()
    plt.savefig(f'data/09_data_exploration/partial_dependence_{title.lower().replace(" ", "_")}.png')
    plt.close()

def compare_feature_distributions(X_original, X_new):
    n_features = X_original.shape[1]
    n_cols = 3
    n_rows = (n_features - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()
    
    for i, feature in enumerate(X_original.columns):
        sns.kdeplot(X_original[feature], ax=axes[i], label='Original', shade=True)
        sns.kdeplot(X_new[feature], ax=axes[i], label='New', shade=True)
        axes[i].set_title(feature)
        axes[i].legend()
    
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('data/09_data_exploration/feature_distribution_comparison.png')
    plt.close()

def main():
    # Load and preprocess data
    preprocessed_data_new = load_and_preprocess_data('data/01_raw/dataset_id_T01_V3_96.csv')
    preprocessed_data_original = load_and_preprocess_data('data/01_raw/dataset_id_96.csv')

    X_new = preprocessed_data_new.drop('y', axis=1)
    y_new = preprocessed_data_new['y']
    X_original = preprocessed_data_original.drop('y', axis=1)
    y_original = preprocessed_data_original['y']

    # Load the model
    model_a = mlflow.sklearn.load_model("models:/Model/20")

    # Evaluate model on both datasets
    y_pred_original = evaluate_model(model_a, X_original, y_original, "Original")
    y_pred_new = evaluate_model(model_a, X_new, y_new, "New")

    # Plot confusion matrices
    plot_confusion_matrix(y_original, y_pred_original, "Original Data")
    plot_confusion_matrix(y_new, y_pred_new, "New Data")

    # Plot feature importance
    plot_feature_importance(model_a, X_original, "Original Data")
    plot_feature_importance(model_a, X_new, "New Data")

    # Plot partial dependence
    plot_partial_dependence(model_a, X_original, "Original Data")
    plot_partial_dependence(model_a, X_new, "New Data")

    # Compare feature distributions
    compare_feature_distributions(X_original, X_new)

    # Generate data drift report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=X_original, current_data=X_new)
    drift_report.save_html("data/08_reporting/data_drift_report.html")
    print("Data drift report saved at data/08_reporting/data_drift_report.html")

    # Print classification reports
    print("\nClassification Report - Original Data:")
    print(classification_report(y_original, y_pred_original))
    print("\nClassification Report - New Data:")
    print(classification_report(y_new, y_pred_new))

if __name__ == "__main__":
    main()