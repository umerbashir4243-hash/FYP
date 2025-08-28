"""
Configuration settings for the Intrusion Detection System.
"""
import os
import logging

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# Selected dataset: 'nsl_kdd' or 'cic_ids'
SELECTED_DATASET = 'nsl_kdd'

# NSL-KDD Dataset Files
TRAIN_FILE = 'KDDTrain+_20Percent.txt'  # Training subset (fast)
TEST_FILE = 'KDDTest-21.txt'             # Test set (excludes difficulty 21)

# CIC-IDS Dataset Files
CIC_IDS_FILE = 'cic-ids/cic_ids_subset.csv'          # CIC-IDS 2017 dataset

# Data directory
DATA_DIR = 'data'

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Selected models to train and evaluate
SELECTED_MODELS = ['RandomForest', 'SVM']  # Available: 'RandomForest', 'SVM', 'SGD'

# =============================================================================
# FEATURE SELECTION CONFIGURATION
# =============================================================================

# Feature selection method: 'RFE', 'PCA', or None
FEATURE_SELECTION_METHOD = 'RFE'

# Number of features for RFE
NUM_FEATURES_RFE = 20

# CIC-IDS specific RFE features (different dataset has different optimal features)
CIC_IDS_NUM_FEATURES_RFE = 15

# PCA variance to retain
PCA_VARIANCE = 0.95

# =============================================================================
# EXECUTION CONFIGURATION
# =============================================================================

# Execution mode: 'test' for development, 'prod' for production
EXECUTION_MODE = 'test'

# Logging level
LOG_LEVEL = logging.INFO

# =============================================================================
# EXPLAINABLE AI CONFIGURATION
# =============================================================================

# Enable Explainable AI features
ENABLE_XAI = True

# Explainer method: 'SHAP' or 'LIME'
EXPLAINER_METHOD = 'SHAP'

# =============================================================================
# SYNTHETIC DATA CONFIGURATION (for testing)
# =============================================================================

# Number of synthetic samples to generate for testing
SYNTHETIC_DATA_SAMPLES = 1000

# =============================================================================
# OUTPUT DIRECTORY CONFIGURATION
# =============================================================================

def get_output_dirs():
    """
    Returns dictionary of output directories based on selected dataset.
    """
    base_output = f'output/{SELECTED_DATASET.upper()}'
    
    return {
        'OUTPUT_DIR': base_output,
        'MODEL_DIR': f'{base_output}/models',
        'RESULTS_DIR': f'{base_output}/results',
        'PLOTS_DIR': f'{base_output}/plots',
        'LOG_FILE': f'{base_output}/app.log'
    }

# Legacy output directories (for backward compatibility)
OUTPUT_DIR = get_output_dirs()['OUTPUT_DIR']
MODEL_DIR = get_output_dirs()['MODEL_DIR']
RESULTS_DIR = get_output_dirs()['RESULTS_DIR']
PLOTS_DIR = get_output_dirs()['PLOTS_DIR']
LOG_FILE = get_output_dirs()['LOG_FILE'] 