"""
Evaluates model performance and generates reports and plots.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report)
import os
import config
from utils import timing_decorator

def get_label_names():
    """Returns appropriate label names based on selected dataset."""
    if config.SELECTED_DATASET == 'cic_ids':
        return ['BENIGN', 'DDoS']
    else:
        return ['Normal', 'Attack']

@timing_decorator
def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluates a single model and returns a dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else 'N/A'
    }
    
    # Get appropriate label names
    label_names = get_label_names()
    
    # Save classification report
    report = classification_report(y_test, y_pred, target_names=label_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    save_report(report_df, f'{model_name}_classification_report')
    
    # Save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name, label_names)
    
    return metrics

def save_report(report_df, report_name):
    """Saves a pandas DataFrame report to a CSV file."""
    # Get current output directories
    output_dirs = config.get_output_dirs()
    results_dir = output_dirs['RESULTS_DIR']
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    report_path = os.path.join(results_dir, f'{report_name}.csv')
    report_df.to_csv(report_path)

def plot_confusion_matrix(cm, model_name, label_names=None):
    """Plots and saves the confusion matrix."""
    # Get current output directories
    output_dirs = config.get_output_dirs()
    plots_dir = output_dirs['PLOTS_DIR']
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    if label_names is None:
        label_names = get_label_names()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix for {model_name} ({config.SELECTED_DATASET.upper()})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plot_path = os.path.join(plots_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()

if __name__ == '__main__':
    from data_processor import load_dataset
    from models import load_model
    
    print("Loading data and models for evaluation test...")
    X_train, X_test, y_train, y_test = load_dataset()
    
    if X_train is not None:
        for model_name in config.SELECTED_MODELS:
            model = load_model(model_name)
            if model:
                evaluate_model(model, X_test, y_test, model_name)
            else:
                print(f"Skipping evaluation for {model_name} as model is not loaded.") 