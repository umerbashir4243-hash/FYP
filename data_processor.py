"""
Handles data loading, cleaning, and preprocessing for the IDS datasets.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import config
import os
import logging
from utils import timing_decorator
import numpy as np

# NSL-KDD dataset column names (files don't include headers)
NSL_KDD_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
]

@timing_decorator
def load_nsl_kdd():
    """
    Loads and preprocesses the NSL-KDD dataset using file paths from config.
    """
    train_path = os.path.join(config.DATA_DIR, config.TRAIN_FILE)
    test_path = os.path.join(config.DATA_DIR, config.TEST_FILE)

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        logging.error(f"Dataset not found at {train_path} or {test_path}")
        logging.error("Please ensure NSL-KDD files are in the data directory.")
        return None, None, None, None

    # Load NSL-KDD data (no headers in original files)
    logging.info(f"Loading training data from {train_path}")
    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    
    logging.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)
    
    logging.info(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")

    # Combine for preprocessing
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Map attack types to binary classification (normal=0, attack=1)
    logging.info("Mapping attack types to binary classification...")
    df['label'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
    
    # Log attack type distribution
    attack_counts = df['attack_type'].value_counts()
    logging.info(f"Attack type distribution: Normal={attack_counts.get('normal', 0)}, "
                f"Attacks={len(df) - attack_counts.get('normal', 0)}")
    
    # Drop attack_type and difficulty columns (keep only features and binary label)
    df = df.drop(['attack_type', 'difficulty'], axis=1)

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'label']
    
    logging.info(f"Encoding categorical features: {list(categorical_cols)}")
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop('label', axis=1)
    y = df['label']

    # Feature Scaling
    logging.info("Applying standard scaling to features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    logging.info(f"Final dataset shape: {X.shape}")
    logging.info(f"Feature names: {list(X.columns)}")

    # Split back into original train/test sets
    n_train = len(train_df)
    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]
    
    logging.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test

@timing_decorator
def load_cic_ids_2017():
    """
    Loads and preprocesses the CIC-IDS dataset.
    """
    cic_ids_path = os.path.join(config.DATA_DIR, config.CIC_IDS_FILE)
    
    if not os.path.exists(cic_ids_path):
        logging.error(f"CIC-IDS dataset not found at {cic_ids_path}")
        return None, None, None, None
    
    # Load CIC-IDS data (has headers)
    df = pd.read_csv(cic_ids_path)
    
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        df = df.fillna(0)
    
    # Map labels to binary classification (BENIGN=0, DDoS=1)
    df['label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Drop the original Label column and keep only features and binary label
    df = df.drop('Label', axis=1)
    
    # Check for infinite values and replace with large finite numbers
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split into train/test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

@timing_decorator
def load_dataset():
    """
    Loads the selected dataset based on configuration.
    """
    if config.SELECTED_DATASET == 'nsl_kdd':
        logging.info("Loading NSL-KDD dataset...")
        return load_nsl_kdd()
    elif config.SELECTED_DATASET == 'cic_ids':
        logging.info("Loading CIC-IDS dataset...")
        return load_cic_ids_2017()
    else:
        logging.error(f"Unknown dataset: {config.SELECTED_DATASET}")
        logging.error("Available options: 'nsl_kdd', 'cic_ids'")
        return None, None, None, None

if __name__ == '__main__':
    print("Loading NSL-KDD dataset...")
    X_train, X_test, y_train, y_test = load_nsl_kdd()
    if X_train is not None:
        print("NSL-KDD Data loaded successfully:")
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("X_test shape:", X_test.shape)
        print("y_test shape:", y_test.shape)

    print("\nLoading CIC-IDS dataset...")
    X_train_cic, X_test_cic, y_train_cic, y_test_cic = load_cic_ids_2017()
    if X_train_cic is not None:
        print("CIC-IDS Data loaded successfully:")
        print("X_train shape:", X_train_cic.shape)
        print("y_train shape:", y_train_cic.shape)
        print("X_test shape:", X_test_cic.shape)
        print("y_test shape:", y_test_cic.shape) 