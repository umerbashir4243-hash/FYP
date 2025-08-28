"""
Defines, trains, and saves the machine learning models.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import os
import config
from utils import timing_decorator
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def get_models():
    """
    Returns a dictionary of ML models to be trained.
    """
    models = {
        'RandomForest': RandomForestClassifier(random_state=42, verbose=0),
        'SVM': LinearSVC(random_state=42, max_iter=2000, verbose=0),
        'SGD': SGDClassifier(random_state=42, max_iter=2000, verbose=0)
    }
    return {name: model for name, model in models.items() if name in config.SELECTED_MODELS}

@timing_decorator
def train_model(model, X_train, y_train, use_grid_search=False):
    """
    Trains a single model.
    Optionally performs hyperparameter tuning using GridSearchCV.
    """
    if use_grid_search:
        param_grid = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [10, 20, None]
            },
            'LinearSVC': {
                'C': [0.1, 1, 10],
                'loss': ['hinge', 'squared_hinge']
            },
            'SGDClassifier': {
                'alpha': [0.0001, 0.001, 0.01],
                'loss': ['hinge', 'log_loss', 'modified_huber']
            }
        }
        model_name = type(model).__name__
        if model_name in param_grid:
            # Suppress GridSearchCV verbose output
            grid = GridSearchCV(model, param_grid[model_name], refit=True, verbose=0, cv=3)
            grid.fit(X_train, y_train)
            return grid.best_estimator_
        else:
            model.fit(X_train, y_train)
            return model
    else:
        # Add progress bar for model training
        with tqdm(total=1, desc="Training", leave=False) as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)
        return model

def save_model(model, model_name):
    """
    Saves the trained model to a file.
    """
    # Get current output directories
    output_dirs = config.get_output_dirs()
    model_dir = output_dirs['MODEL_DIR']
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, f'{model_name}.joblib')
    joblib.dump(model, model_path)

def load_model(model_name):
    """
    Loads a trained model from a file.
    """
    # Get current output directories
    output_dirs = config.get_output_dirs()
    model_dir = output_dirs['MODEL_DIR']
    
    model_path = os.path.join(model_dir, f'{model_name}.joblib')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

if __name__ == '__main__':
    from data_processor import load_dataset
    
    print("Loading data for model training test...")
    X_train, X_test, y_train, y_test = load_dataset()

    if X_train is not None:
        models = get_models()
        for name, model in models.items():
            print(f"\n--- Training {name} ---")
            trained_model = train_model(model, X_train, y_train, use_grid_search=False)
            save_model(trained_model, name)
            loaded_model = load_model(name)
            if loaded_model:
                print(f"{name} model loaded successfully.")
                # Simple prediction test
                predictions = loaded_model.predict(X_test)
                print(f"Made {len(predictions)} predictions with loaded model.") 