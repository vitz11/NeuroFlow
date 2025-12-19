"""
Model Selector Module
Handles final model selection and pipeline serialization
"""

import joblib
import os
from sklearn.pipeline import Pipeline
from typing import Any, Dict
from datetime import datetime


class ModelSelector:
    """
    Selects and serializes the best model with preprocessing pipeline
    """
    
    def __init__(self, preprocessor, model, model_name: str, 
                 problem_type: str, model_info: Dict):
        """
        Initialize ModelSelector
        
        Args:
            preprocessor: Fitted preprocessing pipeline
            model: Trained model
            model_name: Name of the model
            problem_type: 'classification' or 'regression'
            model_info: Dictionary with model information
        """
        self.preprocessor = preprocessor
        self.model = model
        self.model_name = model_name
        self.problem_type = problem_type
        self.model_info = model_info
        self.full_pipeline = None
        self._create_full_pipeline()
    
    def _create_full_pipeline(self):
        """Create a full pipeline combining preprocessing and model"""
        # The full pipeline includes both preprocessing and the model
        # This ensures that new data goes through the same transformations
        self.full_pipeline = {
            'preprocessor': self.preprocessor,
            'model': self.model,
            'metadata': {
                'model_name': self.model_name,
                'problem_type': self.problem_type,
                'model_info': self.model_info,
                'timestamp': datetime.now().isoformat(),
                'preprocessing_info': self.preprocessor.get_preprocessing_info()
            }
        }
    
    def save_model(self, output_path: str) -> str:
        """
        Save the full pipeline to disk
        
        Args:
            output_path: Path to save the model
        
        Returns:
            Path to saved model
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save using joblib (efficient for sklearn objects)
        joblib.dump(self.full_pipeline, output_path)
        
        print(f"\n✓ Model saved successfully to: {output_path}")
        print(f"  Model: {self.model_name}")
        print(f"  Problem Type: {self.problem_type}")
        
        return output_path
    
    @staticmethod
    def load_model(model_path: str) -> Dict[str, Any]:
        """
        Load a saved model pipeline
        
        Args:
            model_path: Path to the saved model
        
        Returns:
            Dictionary containing the full pipeline
        """
        pipeline = joblib.load(model_path)
        
        print(f"\n✓ Model loaded successfully from: {model_path}")
        print(f"  Model: {pipeline['metadata']['model_name']}")
        print(f"  Problem Type: {pipeline['metadata']['problem_type']}")
        
        return pipeline
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Feature dataframe
        
        Returns:
            Predictions
        """
        # Preprocess the data
        X_transformed, _ = self.preprocessor.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_transformed)
        
        # Inverse transform target if needed
        if self.preprocessor.target_encoder:
            predictions = self.preprocessor.target_encoder.inverse_transform(predictions)
        
        return predictions
    
    def get_pipeline_summary(self) -> str:
        """
        Get a summary of the pipeline
        
        Returns:
            String summary
        """
        summary = []
        summary.append("=" * 60)
        summary.append("MODEL PIPELINE SUMMARY")
        summary.append("=" * 60)
        
        summary.append(f"\nModel: {self.model_name}")
        summary.append(f"Problem Type: {self.problem_type}")
        
        # Preprocessing info
        prep_info = self.preprocessor.get_preprocessing_info()
        summary.append(f"\nFeatures:")
        summary.append(f"  Numerical: {len(prep_info['numerical_features'])}")
        summary.append(f"  Categorical: {len(prep_info['categorical_features'])}")
        summary.append(f"  Dropped: {len(prep_info['features_dropped'])}")
        summary.append(f"  Total Input: {prep_info['n_features_in']}")
        summary.append(f"  Total Output: {prep_info['n_features_out']}")
        
        # Model info
        if 'cv_score' in self.model_info:
            summary.append(f"\nModel Performance:")
            summary.append(f"  CV Score: {self.model_info['cv_score']:.4f}")
            
            if 'test_metrics' in self.model_info:
                summary.append(f"\n  Test Metrics:")
                for metric, value in self.model_info['test_metrics'].items():
                    if value is not None:
                        summary.append(f"    {metric.upper()}: {value:.4f}")
        
        summary.append("\n" + "=" * 60)
        
        return "\n".join(summary)
    
    def print_summary(self):
        """Print the pipeline summary"""
        print(self.get_pipeline_summary())