"""
Data Loader Module
Handles Kaggle API integration and local file uploads
"""

import os
import pandas as pd
from typing import Optional, List, Dict, Tuple
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import shutil

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Load from Streamlit secrets if available (for Streamlit Cloud)
try:
    import streamlit as st
    if hasattr(st, 'secrets'):
        if 'kaggle' in st.secrets:
            os.environ['KAGGLE_USERNAME'] = st.secrets['kaggle']['username']
            os.environ['KAGGLE_KEY'] = st.secrets['kaggle']['key']
except Exception:
    pass  # Streamlit not available or secrets not set


class DataLoader:
    """
    Handles data ingestion from multiple sources
    """
    
    def __init__(self, download_dir: str):
        """
        Initialize DataLoader
        
        Args:
            download_dir: Directory to download datasets
        """
        self.download_dir = download_dir
        os.makedirs(self.download_dir, exist_ok=True)
        self.kaggle_api = None
    
    def initialize_kaggle_api(self) -> Tuple[bool, str]:
        """
        Initialize Kaggle API with credentials
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Set User-Agent to avoid None header error
            os.environ['USER_AGENT'] = 'AutoML-System'
            
            self.kaggle_api = KaggleApi()
            
            # Manually set User-Agent if not set
            if hasattr(self.kaggle_api, 'user_agent'):
                self.kaggle_api.user_agent = 'AutoML-System'
            
            self.kaggle_api.authenticate()
            return True, "Kaggle API initialized successfully"
        except Exception as e:
            error_msg = str(e)
            if 'NoneType' in error_msg or 'User-Agent' in error_msg:
                return False, "Kaggle API authentication issue. Please check your kaggle.json credentials."
            return False, f"Failed to initialize Kaggle API: {error_msg}"
    
    def search_kaggle_datasets(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search for datasets on Kaggle
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
        
        Returns:
            List of dataset dictionaries
        """
        if not self.kaggle_api:
            success, msg = self.initialize_kaggle_api()
            if not success:
                print(f"Warning: Could not initialize Kaggle API: {msg}")
                return []
        
        try:
            # Add User-Agent header to avoid None error
            import requests
            requests.utils.default_user_agent = lambda: 'AutoML-System/1.0'
            
            datasets = self.kaggle_api.dataset_list(search=query)
            
            results = []
            for dataset in datasets[:max_results]:
                results.append({
                    'ref': dataset.ref,
                    'title': dataset.title,
                    'size': dataset.size,
                    'download_count': dataset.downloadCount,
                    'vote_count': dataset.voteCount,
                    'usability_rating': dataset.usabilityRating
                })
            
            return results
        
        except Exception as e:
            error_msg = str(e)
            if 'NoneType' in error_msg or 'User-Agent' in error_msg:
                print(f"Warning: Kaggle API User-Agent error. Continuing without search...")
                return []
            print(f"Error searching datasets: {error_msg}")
            return []
    
    def download_kaggle_dataset(self, dataset_ref: str) -> Tuple[Optional[str], str]:
        """
        Download dataset from Kaggle
        
        Args:
            dataset_ref: Dataset reference (e.g., 'username/dataset-name')
        
        Returns:
            Tuple of (file_path, message)
        """
        if not self.kaggle_api:
            success, msg = self.initialize_kaggle_api()
            if not success:
                return None, msg
        
        try:
            # Create a subdirectory for this dataset
            dataset_name = dataset_ref.split('/')[-1]
            dataset_dir = os.path.join(self.download_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Download the dataset
            self.kaggle_api.dataset_download_files(
                dataset_ref,
                path=dataset_dir,
                unzip=True
            )
            
            # Find the first CSV or Excel file
            file_path = self._find_data_file(dataset_dir)
            
            if file_path:
                return file_path, "Dataset downloaded successfully"
            else:
                return None, "No CSV or Excel file found in dataset"
        
        except Exception as e:
            return None, f"Error downloading dataset: {str(e)}"
    
    def _find_data_file(self, directory: str) -> Optional[str]:
        """
        Find the first CSV or Excel file in a directory
        
        Args:
            directory: Directory to search
        
        Returns:
            Path to the first data file found, or None
        """
        valid_extensions = ['.csv', '.xlsx', '.xls']
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in valid_extensions):
                    return os.path.join(root, file)
        
        return None
    
    def load_from_upload(self, uploaded_file, save_path: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Load dataset from Streamlit uploaded file
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            save_path: Path to save the uploaded file
        
        Returns:
            Tuple of (dataframe, message)
        """
        try:
            # Save uploaded file
            with open(save_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(save_path)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(save_path)
            else:
                return None, "Unsupported file format"
            
            if df.empty:
                return None, "Dataset is empty"
            
            return df, "Dataset loaded successfully"
        
        except Exception as e:
            return None, f"Error loading file: {str(e)}"
    
    def load_from_path(self, file_path: str) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Load dataset from file path
        
        Args:
            file_path: Path to the dataset file
        
        Returns:
            Tuple of (dataframe, message)
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                return None, "Unsupported file format"
            
            if df.empty:
                return None, "Dataset is empty"
            
            return df, "Dataset loaded successfully"
        
        except Exception as e:
            return None, f"Error loading file: {str(e)}"