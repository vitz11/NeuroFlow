"""
Configuration file for AutoML system
Contains all constants and hyperparameter search spaces
"""

import os

# Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models', 'exported')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Data Processing Constants
MAX_FILE_SIZE_MB = 500
MISSING_THRESHOLD = 0.5  # Drop columns with >50% missing values
VARIANCE_THRESHOLD = 0.01  # Feature selection threshold
CORRELATION_THRESHOLD = 0.95  # Remove highly correlated features
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Problem Type Detection
CLASSIFICATION_THRESHOLD = 20  # Max unique values for classification
REGRESSION_MIN_UNIQUE = 10  # Min unique values for regression

# Model Training
MAX_ITER = 1000  # For iterative models
N_JOBS = -1  # Use all available CPU cores

# Hyperparameter Search Spaces
CLASSIFICATION_MODELS = {
    'Logistic Regression': {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [MAX_ITER],
        'solver': ['lbfgs']
    },
    'Decision Tree': {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'SVM': {
        'C': [0.1, 1.0, 10.0],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
}

REGRESSION_MODELS = {
    'Linear Regression': {},  # No hyperparameters
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    },
    'Lasso': {
        'alpha': [0.1, 1.0, 10.0, 100.0]
    },
    'Decision Tree Regressor': {
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest Regressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'XGBoost Regressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0]
    }
}

# Evaluation Metrics
CLASSIFICATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
REGRESSION_METRICS = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']