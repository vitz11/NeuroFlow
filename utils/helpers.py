"""
Utility helper functions for AutoML system
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import os


def validate_file(file_path: str, max_size_mb: int = 500) -> Tuple[bool, str]:
    """
    Validate uploaded file
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum allowed file size in MB
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file size
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File size ({file_size_mb:.2f}MB) exceeds maximum ({max_size_mb}MB)"
    
    # Check file extension
    valid_extensions = ['.csv', '.xlsx', '.xls']
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in valid_extensions:
        return False, f"Invalid file type. Supported: {', '.join(valid_extensions)}"
    
    return True, "Valid"


def load_dataset(file_path: str) -> Tuple[pd.DataFrame, str]:
    """
    Load dataset from file
    
    Args:
        file_path: Path to the dataset file
    
    Returns:
        Tuple of (dataframe, error_message)
    """
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            return None, f"Unsupported file format: {file_ext}"
        
        if df.empty:
            return None, "Dataset is empty"
        
        return df, None
    
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Detect numerical and categorical columns
    
    Args:
        df: Input dataframe
    
    Returns:
        Dictionary with 'numerical' and 'categorical' column lists
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        'numerical': numerical_cols,
        'categorical': categorical_cols
    }


def calculate_missing_percentage(df: pd.DataFrame) -> pd.Series:
    """
    Calculate missing value percentage for each column
    
    Args:
        df: Input dataframe
    
    Returns:
        Series with missing percentages
    """
    return (df.isnull().sum() / len(df) * 100).round(2)


def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic statistics about the dataset
    
    Args:
        df: Input dataframe
    
    Returns:
        Dictionary containing basic statistics
    """
    stats = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'n_duplicates': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'missing_cells': df.isnull().sum().sum(),
        'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
    }
    
    return stats


def safe_eval_metric(y_true, y_pred, metric_func) -> float:
    """
    Safely evaluate a metric, returning 0 if error occurs
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_func: Metric function to evaluate
    
    Returns:
        Metric value or 0 if error
    """
    try:
        return metric_func(y_true, y_pred)
    except Exception:
        return 0.0


def format_model_name(model_name: str) -> str:
    """
    Format model name for display
    
    Args:
        model_name: Original model name
    
    Returns:
        Formatted model name
    """
    return model_name.replace('_', ' ').title()


def create_download_link(file_path: str, link_text: str) -> str:
    """
    Create a download link for Streamlit
    
    Args:
        file_path: Path to the file
        link_text: Text to display for the link
    
    Returns:
        HTML string for download link
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    
    import base64
    b64 = base64.b64encode(data).decode()
    
    file_name = os.path.basename(file_path)
    href = f'<a href="data:file/joblib;base64,{b64}" download="{file_name}">{link_text}</a>'
    
    return href