"""
Model Trainer Module
Handles model training, hyperparameter tuning, and evaluation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from utils.config import (
    CLASSIFICATION_MODELS, REGRESSION_MODELS,
    TEST_SIZE, RANDOM_STATE, CV_FOLDS, N_JOBS, MAX_ITER
)


class ModelTrainer:
    """
    Trains and evaluates multiple ML models with hyperparameter tuning
    """
    
    def __init__(self, problem_type: str):
        """
        Initialize ModelTrainer
        
        Args:
            problem_type: 'classification' or 'regression'
        """
        self.problem_type = problem_type
        self.models = {}
        self.trained_models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models based on problem type"""
        if self.problem_type == 'classification':
            self.models = {
                'Logistic Regression': (LogisticRegression, CLASSIFICATION_MODELS['Logistic Regression']),
                'Decision Tree': (DecisionTreeClassifier, CLASSIFICATION_MODELS['Decision Tree']),
                'Random Forest': (RandomForestClassifier, CLASSIFICATION_MODELS['Random Forest']),
                'SVM': (SVC, CLASSIFICATION_MODELS['SVM']),
                'XGBoost': (XGBClassifier, CLASSIFICATION_MODELS['XGBoost'])
            }
        else:  # regression
            self.models = {
                'Linear Regression': (LinearRegression, REGRESSION_MODELS['Linear Regression']),
                'Ridge': (Ridge, REGRESSION_MODELS['Ridge']),
                'Lasso': (Lasso, REGRESSION_MODELS['Lasso']),
                'Decision Tree Regressor': (DecisionTreeRegressor, REGRESSION_MODELS['Decision Tree Regressor']),
                'Random Forest Regressor': (RandomForestRegressor, REGRESSION_MODELS['Random Forest Regressor']),
                'XGBoost Regressor': (XGBRegressor, REGRESSION_MODELS['XGBoost Regressor'])
            }
    
    def train_and_evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train and evaluate all models
        
        Args:
            X: Feature matrix
            y: Target vector
        
        Returns:
            Dictionary with results for all models
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        print(f"\nTraining {len(self.models)} models...")
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train each model
        for model_name, (ModelClass, param_grid) in self.models.items():
            print(f"\nTraining {model_name}...")
            
            try:
                # Train with hyperparameter tuning
                best_model, cv_scores = self._train_with_tuning(
                    ModelClass, param_grid, X_train, y_train
                )
                
                # Evaluate on test set
                test_metrics = self._evaluate_model(best_model, X_test, y_test)
                
                # Store results
                self.trained_models[model_name] = best_model
                self.results[model_name] = {
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'test_metrics': test_metrics,
                    'best_params': best_model.get_params() if hasattr(best_model, 'get_params') else {}
                }
                
                print(f"✓ {model_name} - CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
            
            except Exception as e:
                print(f"✗ {model_name} failed: {str(e)}")
                self.results[model_name] = {'error': str(e)}
        
        # Select best model
        self._select_best_model()
        
        return self.results
    
    def _train_with_tuning(self, ModelClass, param_grid: dict, X_train: np.ndarray, 
                          y_train: np.ndarray) -> Tuple[Any, np.ndarray]:
        """
        Train model with hyperparameter tuning
        
        Args:
            ModelClass: Model class
            param_grid: Hyperparameter grid
            X_train: Training features
            y_train: Training target
        
        Returns:
            Tuple of (best_model, cv_scores)
        """
        # Determine scoring metric
        if self.problem_type == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'r2'
        
        # If no hyperparameters, train directly
        if not param_grid:
            model = ModelClass()
            model.fit(X_train, y_train)
            cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring=scoring)
            return model, cv_scores
        
        # Grid search with cross-validation
        model = ModelClass()
        
        # For models that need specific parameters
        if 'XGB' in str(ModelClass):
            model = ModelClass(random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=0)
        elif 'Random Forest' in str(ModelClass):
            model = ModelClass(random_state=RANDOM_STATE, n_jobs=N_JOBS)
        elif hasattr(model, 'random_state'):
            model = ModelClass(random_state=RANDOM_STATE)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=CV_FOLDS,
            scoring=scoring,
            n_jobs=N_JOBS,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_score_ * np.ones(CV_FOLDS)
    
    def _evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        if self.problem_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # Add ROC-AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                    else:
                        metrics['roc_auc'] = None
                except:
                    metrics['roc_auc'] = None
        
        else:  # regression
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
        
        return metrics
    
    def _select_best_model(self):
        """Select the best model based on cross-validation scores"""
        valid_results = {
            name: result for name, result in self.results.items() 
            if 'error' not in result
        }
        
        if not valid_results:
            print("\n⚠ No valid models trained!")
            return
        
        # Find model with best CV score
        best_name = max(valid_results.items(), key=lambda x: x[1]['cv_mean'])[0]
        
        self.best_model_name = best_name
        self.best_model = self.trained_models[best_name]
        self.best_score = valid_results[best_name]['cv_mean']
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"CV Score: {self.best_score:.4f}")
        print(f"{'='*60}")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a formatted dataframe
        
        Returns:
            DataFrame with model comparison
        """
        rows = []
        
        for model_name, result in self.results.items():
            if 'error' in result:
                continue
            
            row = {
                'Model': model_name,
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            }
            
            # Add test metrics
            for metric_name, metric_value in result['test_metrics'].items():
                if metric_value is not None:
                    row[metric_name.upper()] = metric_value
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by CV Mean (descending)
        if len(df) > 0:
            df = df.sort_values('CV Mean', ascending=False).reset_index(drop=True)
        
        return df
    
    def get_best_model_info(self) -> Dict[str, Any]:
        """Get information about the best model"""
        if not self.best_model:
            return {}
        
        info = {
            'name': self.best_model_name,
            'cv_score': self.best_score,
            'test_metrics': self.results[self.best_model_name]['test_metrics'],
            'parameters': self.results[self.best_model_name]['best_params']
        }
        
        return info