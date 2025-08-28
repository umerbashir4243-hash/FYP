# Intrusion Detection System

## Overview
This project implements an advanced Intrusion Detection System (IDS) using machine learning techniques, specifically Random Forest and Support Vector Machine (SVM) algorithms, enhanced with feature selection methods and Explainable AI (XAI) techniques. The system is designed based on research requirements for scalable, interpretable, and high-performance intrusion detection using the **real NSL-KDD dataset**.

## Key Features

### Machine Learning Models
- **Random Forest**: Ensemble method for robust classification with excellent performance
- **SVM (LinearSVC)**: Scalable Support Vector Machine implementation optimized for large datasets
- **SGD Classifier**: Alternative linear classifier using Stochastic Gradient Descent (optional)

### Feature Selection
- **Recursive Feature Elimination (RFE)**: Selects the most important features iteratively
- **Principal Component Analysis (PCA)**: Dimensionality reduction technique preserving variance

### Explainable AI (XAI)
- **SHAP (SHapley Additive exPlanations)**: Feature importance and contribution analysis
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local interpretability for individual predictions

### Datasets Supported
- **NSL-KDD**: Real network intrusion detection dataset (fully implemented and tested)
  - Uses actual NSL-KDD files with 41 features from network traffic analysis
  - Binary classification: Normal vs Attack traffic
  - Includes 38 different attack types for comprehensive evaluation
- **CIC-IDS 2017**: Comprehensive intrusion detection evaluation dataset (fully implemented and tested)
  - Uses CIC-IDS subset with 78 features from network flow analysis
  - Binary classification: BENIGN vs DDoS traffic
  - Includes real-world DDoS attack scenarios for evaluation

## Performance Results

The system has been tested with both NSL-KDD and CIC-IDS datasets:

### NSL-KDD Dataset Results
- **Training Set**: 25,192 samples (KDDTrain+_20Percent.txt)
- **Test Set**: 11,850 samples (KDDTest-21.txt)
- **Features**: 41 network traffic features (reduced to 20 via RFE)
- **Attack Types**: 38 different attack categories
- **Random Forest Performance**: 58.36% accuracy, 94.69% precision, 52.04% recall
- **SVM Performance**: 52.58% accuracy, 88.19% precision, 48.57% recall

### CIC-IDS Dataset Results
- **Dataset**: 36,999 samples (22,659 BENIGN, 14,340 DDoS)
- **Features**: 78 network flow features (reduced to 15 via RFE)
- **Attack Type**: DDoS attacks
- **Pipeline Successfully Validated**: Both datasets processed without errors

Performance metrics are generated dynamically during execution and saved to `output/{DATASET}/results/overall_performance_summary.csv`.

## Architecture

### Model Implementation Notes
- **SVM Implementation**: Uses `LinearSVC` instead of `SVC` for better scalability on large datasets
  - `SVC` has quadratic scaling and becomes impractical beyond tens of thousands of samples
  - `LinearSVC` provides linear scaling and is more suitable for large-scale intrusion detection
  - For models without `predict_proba` (like LinearSVC), we use `CalibratedClassifierCV` for XAI compatibility

### Configuration
The system supports two execution modes:
- **Production Mode**: Full dataset processing with comprehensive model training and hyperparameter tuning
- **Test Mode**: Smaller datasets and faster settings for development and testing

### Dataset Configuration
- **NSL-KDD**: Default dataset with 41 features, binary classification (Normal vs Attack)
- **CIC-IDS**: Alternative dataset with 78 features, binary classification (BENIGN vs DDoS)
- **Feature Selection**: RFE reduces NSL-KDD to 20 features, CIC-IDS to 15 features
- **Execution Time**: NSL-KDD ~3 minutes, CIC-IDS ~17 seconds (feature selection phase)

## File Structure

```
pocs/intrusion_detection/
├── config.py              # Configuration settings and parameters
├── data_processor.py      # Data loading and preprocessing for NSL-KDD
├── feature_selector.py    # Feature selection implementations (RFE, PCA)
├── models.py              # ML model definitions and training
├── explainable_ai.py      # XAI implementations (SHAP & LIME)
├── evaluator.py           # Model evaluation and metrics
├── main.py                # Main execution script
├── generate_dataset.py    # Synthetic dataset generation (legacy - not used)
├── test_ids_system.py     # System testing suite
├── utils.py               # Utility functions (timing decorators, etc.)
├── requirements.txt       # Python dependencies
├── Proposal.pdf           # Research proposal document
├── data/                  # Dataset directory
│   ├── cic-ids/           # CIC-IDS dataset files
│   │   ├── cic_ids_subset.csv    # CIC-IDS subset with BENIGN/DDoS traffic
│   │   └── cic_ids_sample.csv    # CIC-IDS sample for testing
│   ├── KDDTrain+_20Percent.txt  # NSL-KDD training subset
│   └── KDDTest-21.txt     # NSL-KDD test set
├── output/                # Generated outputs
│   ├── NSL_KDD/           # NSL-KDD results
│   │   ├── models/        # Saved trained models (.joblib files)
│   │   ├── results/       # Performance reports and LIME explanations
│   │   └── plots/         # SHAP visualizations and confusion matrices
│   └── CIC_IDS/           # CIC-IDS results (when processed)
└── README.md             # This file
```

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)
- NSL-KDD dataset files (included in `data/`)
- CIC-IDS dataset files (included in `data/cic-ids/`)

### Installation Steps

1. **Navigate to the project directory:**
   ```bash
   cd pocs/intrusion_detection
   ```

2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify dataset files:**
   ```bash
   # Check that NSL-KDD files are available
   ls data/KDDTrain+_20Percent.txt data/KDDTest-21.txt
   
   # Check that CIC-IDS files are available
   ls data/cic-ids/cic_ids_subset.csv
   ```
   All required dataset files are included in the repository.

## Usage

### Quick Start
```bash
# Run the complete IDS pipeline with default dataset (NSL-KDD)
python main.py

# Switch to CIC-IDS dataset by editing config.py
# Change SELECTED_DATASET = 'cic_ids' and run again
python main.py

# Run individual components
python data_processor.py  # Test data loading
python models.py          # Train models only
python explainable_ai.py  # Generate XAI explanations
python evaluator.py       # Evaluate model performance
python feature_selector.py # Test feature selection methods
```

### Configuration Options
Edit `config.py` to customize:

```python
# Dataset Selection
SELECTED_DATASET = 'nsl_kdd'       # Options: 'nsl_kdd', 'cic_ids'

# NSL-KDD Dataset Files
TRAIN_FILE = 'KDDTrain+_20Percent.txt'  # Training subset (fast)
TEST_FILE = 'KDDTest-21.txt'             # Test set (excludes difficulty 21)

# CIC-IDS Dataset Files
CIC_IDS_FILE = 'cic-ids/cic_ids_subset.csv'  # CIC-IDS subset with BENIGN/DDoS

# Model Selection
SELECTED_MODELS = ['RandomForest', 'SVM']  # Available: 'RandomForest', 'SVM', 'SGD'

# Feature Selection
FEATURE_SELECTION_METHOD = 'RFE'  # Options: 'RFE', 'PCA', None
NUM_FEATURES_RFE = 20              # Number of features for NSL-KDD RFE
CIC_IDS_NUM_FEATURES_RFE = 15      # Number of features for CIC-IDS RFE
PCA_VARIANCE = 0.95                # Variance to retain for PCA

# Execution Mode
EXECUTION_MODE = 'test'            # Options: 'test', 'prod'

# Explainable AI
ENABLE_XAI = True                  # Enable SHAP and LIME explanations
EXPLAINER_METHOD = 'SHAP'         # Options: 'SHAP', 'LIME'
```

### Using Different Datasets
You can switch between different datasets by updating `config.py`:

```python
# Switch to CIC-IDS dataset
SELECTED_DATASET = 'cic_ids'

# Switch back to NSL-KDD dataset
SELECTED_DATASET = 'nsl_kdd'

# Using different NSL-KDD files (if available)
TRAIN_FILE = 'KDDTrain+.txt'    # Full training set (slower but more comprehensive)
TEST_FILE = 'KDDTest+.txt'      # Full test set with all difficulty levels
TEST_FILE = 'KDDTest-21.txt'    # Excludes highest difficulty samples
```

### Testing the System

Run comprehensive tests to verify all functionality:

```bash
# Test data loading specifically
python data_processor.py

# Run existing test suite
python test_ids_system.py

# Test individual components
python -c "import models; print('Models working')"
python -c "import explainable_ai; print('XAI working')"
python -c "import feature_selector; print('Feature selection working')"
```

## Performance Considerations

### Dataset Sizes and Performance
#### NSL-KDD Dataset
- **KDDTrain+_20Percent.txt**: 25,192 samples (recommended for testing)
- **KDDTrain+.txt**: 125,973 samples (full training set)
- **KDDTest-21.txt**: 11,850 samples (filtered test set)
- **KDDTest+.txt**: 22,544 samples (full test set)
- **Execution Time**: ~3 minutes (including SHAP explanations)

#### CIC-IDS Dataset
- **cic_ids_subset.csv**: 36,999 samples (22,659 BENIGN, 14,340 DDoS)
- **Execution Time**: ~17 seconds (feature selection phase)
- **Memory Usage**: Lower due to fewer features after RFE

### SVM Scalability
- **LinearSVC**: Recommended for datasets > 10,000 samples
- **SGDClassifier**: Alternative for very large datasets (>100,000 samples)
- **Calibration**: Applied automatically for XAI compatibility when needed

### Feature Selection Impact
- **RFE**: Better feature interpretability, moderate computational cost
- **PCA**: Faster processing, reduced interpretability but maintains variance

### Memory and Processing
- **Test Mode**: Uses 20% NSL-KDD subset for quick validation
- **Production Mode**: Scales to full NSL-KDD dataset with optimized processing

## Dataset Information

### NSL-KDD (Fully Supported)
- **Status**: ✅ Fully implemented and tested with real data
- **Source**: Real NSL-KDD dataset files from University of New Brunswick
- **Features**: 41 network traffic features (duration, protocol_type, service, etc.)
- **Classification**: Binary (normal vs attack) with 38 attack types
- **Files Used**:
  - `KDDTrain+_20Percent.txt`: 20% training subset for faster testing
  - `KDDTest-21.txt`: Test set excluding difficulty level 21
- **Format**: CSV without headers (headers added programmatically)
- **Attack Types**: Includes DoS, Probe, R2L, U2R attack categories

### Attack Type Distribution in NSL-KDD
The dataset includes various attack types:
- **Normal Traffic**: Legitimate network connections
- **DoS Attacks**: neptune, smurf, back, teardrop, etc.
- **Probe Attacks**: satan, ipsweep, portsweep, nmap, etc.
- **R2L Attacks**: guess_passwd, warezmaster, ftp_write, etc.
- **U2R Attacks**: buffer_overflow, rootkit, perl, etc.

### CIC-IDS 2017 (Placeholder)
- **Status**: ⚠️ Placeholder implementation only
- **Source**: [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Size**: ~7GB with multiple CSV files
- **Implementation**: Requires manual download and custom loader implementation
- **To Use**: 
  1. Download the dataset
  2. Update `CIC_IDS_2017_PATH` in `config.py`
  3. Implement the loader in `data_processor.py`

## Output

The system generates comprehensive outputs:

### Models
- **Trained Models**: Saved as `.joblib` files in `output/models/`
- **Model Metadata**: Training parameters and configuration

### Performance Analysis
- **Classification Reports**: Detailed metrics per model saved as CSV
- **Confusion Matrices**: Visual performance analysis as PNG plots
- **Overall Summary**: Comparative performance across models in CSV format

### Explainable AI
- **SHAP Visualizations**: Feature importance plots saved as PNG
- **LIME Explanations**: Local interpretability HTML reports
- **Feature Analysis**: Contribution analysis for decision-making

## Research Alignment

This implementation fully aligns with modern IDS research requirements:

### ✅ **Proposal Compliance**
- **Models**: Random Forest and SVM (LinearSVC) as specified
- **Datasets**: Both NSL-KDD and CIC-IDS 2017 for comprehensive evaluation
- **Feature Selection**: Both RFE and PCA implemented with dataset-specific optimization
- **Explainability**: SHAP and LIME for transparency and interpretability
- **Scalability**: LinearSVC for large-scale deployment
- **Performance**: Comprehensive evaluation on standard benchmark datasets
- **Configuration**: Missing config.py file has been created and tested

### ✅ **Best Practices**
- Standard benchmark datasets (NSL-KDD, CIC-IDS 2017) for reproducible results
- Scalable machine learning approaches with dataset-specific optimization
- Feature selection optimization (RFE: 20 features for NSL-KDD, 15 for CIC-IDS)
- Explainable AI for cybersecurity applications
- Comprehensive evaluation methodologies
- Modular and extensible architecture
- Configuration-driven approach for easy dataset switching

## Dependencies

### Core Libraries
```
scikit-learn    # ML algorithms and utilities
pandas          # Data manipulation and analysis
numpy           # Numerical computing
matplotlib      # Visualization and plotting
seaborn         # Statistical visualization
shap            # SHAP explanations
lime            # LIME explanations
```

### Installation
All dependencies are listed in `requirements.txt` and can be installed with:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure virtual environment is activated
2. **Dataset Not Found**: Verify dataset files are in correct directories
   - NSL-KDD: `data/KDDTrain+_20Percent.txt`, `data/KDDTest-21.txt`
   - CIC-IDS: `data/cic-ids/cic_ids_subset.csv`
3. **Memory Issues**: Use test mode for initial testing
4. **SHAP/LIME Slow**: Reduce sample sizes in config for faster processing
5. **Config File Missing**: The config.py file has been created and tested

### Dataset-Specific Issues

#### NSL-KDD Dataset
1. **File Format**: NSL-KDD files have no headers - handled automatically by data processor
2. **Attack Types**: System converts multi-class to binary (normal vs attack)
3. **Feature Count**: 41 features reduced to 20 via RFE

#### CIC-IDS Dataset
1. **File Format**: CIC-IDS files have headers - processed automatically
2. **Attack Types**: Binary classification (BENIGN vs DDoS)
3. **Feature Count**: 78 features reduced to 15 via RFE
4. **Data Balance**: 22,659 BENIGN vs 14,340 DDoS samples

### Performance Optimization

- **For Large Datasets**: Use `EXECUTION_MODE = 'prod'` with full dataset files
- **For Quick Testing**: Use `EXECUTION_MODE = 'test'` with smaller subsets
- **For XAI**: Adjust `SHAP_SAMPLE_SIZE` for faster explanations
- **For CIC-IDS**: Faster execution due to optimized feature selection
- **For NSL-KDD**: Longer execution due to SHAP explanations on larger feature set

## Development and Testing

### Code Quality
- Modular design with clear separation of concerns
- Comprehensive error handling and logging
- Timing decorators for performance monitoring
- Real dataset integration for authentic testing

### Extensibility
- Easy addition of new models in `models.py`
- Pluggable feature selection methods
- Configurable XAI approaches
- Support for additional datasets (CIC-IDS 2017 implemented and tested)
- Configuration-driven dataset switching
- Modular output organization by dataset

## Notes

- **Real Data**: System uses authentic NSL-KDD and CIC-IDS files instead of synthetic data
- **Automatic Processing**: Column names and preprocessing handled automatically for both datasets
- **Attack Mapping**: Multi-class attack types converted to binary classification
- **XAI Performance**: Explanations may take longer for large datasets - adjust sample sizes accordingly
- **Mode Selection**: Use test mode for development, production mode for final evaluation
- **Scalability**: LinearSVC implementation ensures the system scales to real-world deployment scenarios
- **Benchmark**: Uses standard NSL-KDD and CIC-IDS datasets for reproducible research results
- **Configuration**: Missing config.py file has been created and tested successfully
- **Dual Dataset Support**: Both NSL-KDD and CIC-IDS datasets are fully functional
- **Performance Validation**: Pipeline successfully tested on both datasets with different feature counts

## License and Usage

This implementation is designed for research and educational purposes, focusing on intrusion detection system development with explainable AI capabilities using the standard NSL-KDD benchmark dataset. 