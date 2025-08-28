"""
Main entry point for the Intrusion Detection System application.
Orchestrates the workflow: data loading, feature selection, training, evaluation, and explanation.
"""
import os
import logging
import pandas as pd
import config
import data_processor
import feature_selector
import models
import evaluator
import explainable_ai
from utils import timing_decorator
import time
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_logging():
    """Sets up logging to file and console."""
    # Get fresh output directories based on current dataset selection
    output_dirs = config.get_output_dirs()
    
    # Create all output directories
    for dir_path in [output_dirs['OUTPUT_DIR'], output_dirs['MODEL_DIR'], 
                     output_dirs['RESULTS_DIR'], output_dirs['PLOTS_DIR']]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    # Suppress SHAP and other verbose logging
    import logging
    logging.getLogger('shap').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('seaborn').setLevel(logging.ERROR)
        
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dirs['LOG_FILE']),
            logging.StreamHandler()
        ],
        force=True  # Allow re-configuration of logging
    )

@timing_decorator
def main(data=None):
    """Main function to run the IDS pipeline."""
    setup_logging()
    logging.info(f"Starting IDS pipeline - Dataset: {config.SELECTED_DATASET.upper()}, Mode: {config.EXECUTION_MODE}")

    # Determine whether to run grid search based on execution mode
    run_grid_search = (config.EXECUTION_MODE == 'prod')

    # 1. Load Data
    if data:
        X_train, X_test, y_train, y_test = data
    else:
        with tqdm(total=1, desc="Loading Data", leave=False) as pbar:
            X_train, X_test, y_train, y_test = data_processor.load_dataset()
            pbar.update(1)
        if X_train is None:
            logging.error("Data loading failed. Exiting.")
            return

    # 2. Feature Selection
    X_train_selected, X_test_selected = feature_selector.select_features(
        X_train, y_train, X_test
    )

    # 3. Train and Evaluate Models
    ml_models = models.get_models()
    all_metrics = {}

    for name, model in ml_models.items():
        logging.info(f"Processing {name}...")
        
        # Train model
        trained_model = models.train_model(model, X_train_selected, y_train, use_grid_search=run_grid_search)
        models.save_model(trained_model, name)
        
        # Evaluate model
        with tqdm(total=1, desc=f"Evaluating {name}", leave=False) as pbar:
            metrics = evaluator.evaluate_model(trained_model, X_test_selected, y_test, name)
            pbar.update(1)
        all_metrics[name] = metrics
        
        # Explain model predictions
        if config.ENABLE_XAI:
            if config.EXPLAINER_METHOD == 'SHAP':
                explainable_ai.explain_with_shap(trained_model, X_train_selected, X_test_selected, name, y_train)
            elif config.EXPLAINER_METHOD == 'LIME':
                explainable_ai.explain_with_lime(trained_model, X_train_selected, X_test_selected, y_train, name)

    # 4. Final Report
    logging.info("Pipeline completed successfully.")
    metrics_df = pd.DataFrame(all_metrics).transpose()
    evaluator.save_report(metrics_df, 'overall_performance_summary')

    logging.shutdown() # Properly close all logging handlers

if __name__ == '__main__':
    main() 