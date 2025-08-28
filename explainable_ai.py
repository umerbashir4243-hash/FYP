"""
Integrates Explainable AI (ExAI) techniques like SHAP and LIME.
"""
import shap
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import config
import lime
import lime.lime_tabular
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
import warnings

# Suppress SHAP verbose logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def explain_with_shap(model, X_train, X_test, model_name, y_train=None):
    """
    Generates and saves SHAP summary plots for a model.
    """
    feature_names = X_train.columns.tolist()
    X_train_np = X_train.values
    
    background_data = shap.sample(X_train_np, min(100, X_train_np.shape[0]))
    
    # Handle models without predict_proba (like LinearSVC)
    if hasattr(model, 'predict_proba'):
        predict_fn = model.predict_proba
    else:
        # Use CalibratedClassifierCV to add probability support for LinearSVC
        if y_train is None:
            predict_fn = lambda x: np.column_stack([1 - model.decision_function(x), model.decision_function(x)])
        else:
            calibrated_model = CalibratedClassifierCV(model, cv=3)
            calibrated_model.fit(X_train_np, y_train)
            predict_fn = calibrated_model.predict_proba
    
    # Suppress SHAP logging
    import logging
    logging.getLogger('shap').setLevel(logging.ERROR)
    
    explainer = shap.KernelExplainer(predict_fn, background_data)
    
    X_test_sample = X_test.sample(n=min(100, len(X_test)), random_state=42)
    X_test_sample_np = X_test_sample.values

    # Add progress bar for SHAP calculations
    with tqdm(total=len(X_test_sample_np), desc=f"SHAP {model_name}", leave=False) as pbar:
        shap_values = explainer.shap_values(X_test_sample_np)
        pbar.update(len(X_test_sample_np))

    # Get current output directories
    output_dirs = config.get_output_dirs()
    plots_dir = output_dirs['PLOTS_DIR']
    
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Robustly handle the structure of shap_values
    if isinstance(shap_values, list):
        # Standard case for binary classifiers from scikit-learn
        shap_values_for_plot = shap_values[1]
    else:
        # Handle cases where shap_values is a numpy array
        if shap_values.ndim == 3:
            # If 3D, it's likely (samples, features, classes), take values for class 1
            shap_values_for_plot = shap_values[:, :, 1]
        else:
            # If 2D, use as is
            shap_values_for_plot = shap_values

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_for_plot, X_test_sample_np, feature_names=feature_names, show=False)
    plt.title(f'SHAP Summary Plot for {model_name} ({config.SELECTED_DATASET.upper()})')
    plot_path = os.path.join(plots_dir, f'{model_name}_shap_summary.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()


def explain_with_lime(model, X_train, X_test, y_train, model_name, instance_index=0):
    """
    Generates and saves a LIME explanation for a single instance.
    """
    # Get appropriate class names based on dataset
    if config.SELECTED_DATASET == 'cic_ids':
        class_names = ['BENIGN', 'DDoS']
    else:
        class_names = ['Normal', 'Attack']

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        mode='classification'
    )

    instance = X_test.iloc[instance_index].values
    
    # Handle models without predict_proba (like LinearSVC)
    if hasattr(model, 'predict_proba'):
        predict_fn = model.predict_proba
    else:
        # Use CalibratedClassifierCV to add probability support for LinearSVC
        calibrated_model = CalibratedClassifierCV(model, cv=3)
        calibrated_model.fit(X_train.values, y_train)
        predict_fn = calibrated_model.predict_proba
    
    explanation = explainer.explain_instance(
        instance,
        predict_fn,
        num_features=10
    )

    # Get current output directories
    output_dirs = config.get_output_dirs()
    results_dir = output_dirs['RESULTS_DIR']
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    output_file = os.path.join(results_dir, f'{model_name}_lime_explanation.html')
    explanation.save_to_file(output_file)


if __name__ == '__main__':
    from data_processor import load_dataset
    from models import load_model

    print("Loading data and models for ExAI test...")
    X_train, X_test, y_train, y_test = load_dataset()

    if X_train is not None:
        for model_name in config.SELECTED_MODELS:
            model = load_model(model_name)
            if model:
                # Due to performance intensity, you might want to run these selectively
                explain_with_shap(model, X_train, X_test, model_name, y_train)
                explain_with_lime(model, X_train, X_test, y_train, model_name)
            else:
                print(f"Skipping ExAI for {model_name} as model is not loaded.") 