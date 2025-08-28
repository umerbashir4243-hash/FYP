"""
Implements feature selection techniques like RFE and PCA.
"""
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import config
from utils import timing_decorator
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

@timing_decorator
def select_features(X_train, y_train, X_test, method=config.FEATURE_SELECTION_METHOD):
    """
    Selects features using the specified method.
    """
    if method == 'RFE':
        return apply_rfe(X_train, y_train, X_test)
    elif method == 'PCA':
        return apply_pca(X_train, X_test)
    else:
        return X_train, X_test

def apply_rfe(X_train, y_train, X_test):
    """
    Applies Recursive Feature Elimination (RFE) to select the best features.
    """
    # Choose number of features based on selected dataset
    if config.SELECTED_DATASET == 'cic_ids':
        n_features = config.CIC_IDS_NUM_FEATURES_RFE
    else:
        n_features = config.NUM_FEATURES_RFE
    
    # Using a simpler RandomForest for RFE to speed up the process.
    # A smaller number of estimators is sufficient for feature ranking.
    estimator = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1, verbose=0)
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    
    # Add progress bar for RFE
    with tqdm(total=1, desc="RFE Selection", leave=False) as pbar:
        selector = selector.fit(X_train, y_train)
        pbar.update(1)
    
    selected_features = X_train.columns[selector.support_]
    
    X_train_rfe = X_train[selected_features]
    X_test_rfe = X_test[selected_features]
    
    return X_train_rfe, X_test_rfe

def apply_pca(X_train, X_test):
    """
    Applies Principal Component Analysis (PCA) for dimensionality reduction.
    """
    pca = PCA(n_components=config.PCA_VARIANCE)
    
    # Add progress bar for PCA
    with tqdm(total=1, desc="PCA", leave=False) as pbar:
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        pbar.update(1)
    
    # Convert back to DataFrame for consistency, though column names are lost
    X_train_pca = pd.DataFrame(X_train_pca, index=X_train.index)
    X_test_pca = pd.DataFrame(X_test_pca, index=X_test.index)

    return X_train_pca, X_test_pca

if __name__ == '__main__':
    from data_processor import load_dataset
    
    print("Loading data for feature selection testing...")
    X_train, X_test, y_train, y_test = load_dataset()

    if X_train is not None:
        # Test RFE
        config.FEATURE_SELECTION_METHOD = 'RFE'
        X_train_rfe, X_test_rfe = select_features(X_train, y_train, X_test)
        print("RFE output shapes:", X_train_rfe.shape, X_test_rfe.shape)

        # Test PCA
        config.FEATURE_SELECTION_METHOD = 'PCA'
        X_train_pca, X_test_pca = select_features(X_train, y_train, X_test)
        print("PCA output shapes:", X_train_pca.shape, X_test_pca.shape) 