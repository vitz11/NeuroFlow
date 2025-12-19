"""
AutoML System - Streamlit Main Application
Complete end-to-end automated machine learning system
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.data_loader import DataLoader
from pipelines.data_profiler import DataProfiler
from pipelines.problem_detector import ProblemDetector
from pipelines.preprocessor import AutoPreprocessor
from pipelines.model_trainer import ModelTrainer
from pipelines.model_selector import ModelSelector
from utils.config import RAW_DATA_DIR, MODELS_DIR


# Page configuration
st.set_page_config(
    page_title="AutoML System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'profile' not in st.session_state:
        st.session_state.profile = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'model_path' not in st.session_state:
        st.session_state.model_path = None


def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🤖 AutoML System</p>', unsafe_allow_html=True)
    st.markdown("**Automated Machine Learning Pipeline** - From Data to Deployment")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        data_source = st.radio(
            "Select Data Source",
            ["Upload File", "Kaggle Dataset"],
            help="Choose how to load your dataset"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This AutoML system automatically:
        - ✅ Loads and profiles data
        - ✅ Detects problem type
        - ✅ Preprocesses features
        - ✅ Trains multiple models
        - ✅ Selects the best model
        - ✅ Exports trained pipeline
        """)
    
    # Main content area
    if data_source == "Upload File":
        handle_file_upload()
    else:
        handle_kaggle_search()
    
    # Display data profile if loaded
    if st.session_state.data_loaded:
        display_data_profile()
        
        # Model training section
        handle_model_training()


def handle_file_upload():
    """Handle file upload interface"""
    st.markdown('<p class="sub-header">📁 Upload Dataset</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Maximum file size: 500MB"
    )
    
    if uploaded_file:
        with st.spinner("Loading dataset..."):
            # Initialize data loader
            data_loader = DataLoader(RAW_DATA_DIR)
            
            # Save and load file
            file_path = os.path.join(RAW_DATA_DIR, uploaded_file.name)
            df, message = data_loader.load_from_upload(uploaded_file, file_path)
            
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.success(f"✓ {message}")
                
                # Generate profile
                profiler = DataProfiler(df)
                st.session_state.profile = profiler.generate_profile()
            else:
                st.error(f"✗ {message}")


def handle_kaggle_search():
    """Handle Kaggle dataset search interface"""
    st.markdown('<p class="sub-header">🔍 Search Kaggle Datasets</p>', unsafe_allow_html=True)
    
    st.info("ℹ️ Make sure you have set up your Kaggle API credentials (~/.kaggle/kaggle.json)")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search for datasets",
            placeholder="e.g., housing prices, iris, titanic"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    if search_button and search_query:
        with st.spinner("Searching Kaggle..."):
            data_loader = DataLoader(RAW_DATA_DIR)
            datasets = data_loader.search_kaggle_datasets(search_query, max_results=10)
            
            if datasets:
                st.success(f"Found {len(datasets)} datasets")
                
                # Display datasets
                for i, dataset in enumerate(datasets):
                    with st.expander(f"{i+1}. {dataset['title']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Size", dataset['size'])
                        with col2:
                            st.metric("Downloads", f"{dataset['download_count']:,}")
                        with col3:
                            st.metric("Usability", f"{dataset['usability_rating']:.2f}")
                        
                        if st.button(f"Download", key=f"download_{i}"):
                            with st.spinner(f"Downloading {dataset['title']}..."):
                                file_path, message = data_loader.download_kaggle_dataset(dataset['ref'])
                                
                                if file_path:
                                    df, load_msg = data_loader.load_from_path(file_path)
                                    
                                    if df is not None:
                                        st.session_state.df = df
                                        st.session_state.data_loaded = True
                                        st.success(f"✓ Dataset loaded successfully!")
                                        
                                        # Generate profile
                                        profiler = DataProfiler(df)
                                        st.session_state.profile = profiler.generate_profile()
                                        st.rerun()
                                    else:
                                        st.error(f"✗ {load_msg}")
                                else:
                                    st.error(f"✗ {message}")
            else:
                st.warning("No datasets found. Try a different search query.")


def display_data_profile():
    """Display dataset profile and statistics"""
    df = st.session_state.df
    profile = st.session_state.profile
    
    st.markdown('<p class="sub-header">📊 Dataset Overview</p>', unsafe_allow_html=True)
    
    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    
    basic_stats = profile['basic_stats']
    
    with col1:
        st.metric("Rows", f"{basic_stats['n_rows']:,}")
    with col2:
        st.metric("Columns", basic_stats['n_columns'])
    with col3:
        st.metric("Missing Cells", f"{basic_stats['missing_cells']:,}")
    with col4:
        st.metric("Quality Score", f"{profile['data_quality_score']:.1f}/100")
    
    # Data preview
    with st.expander("📋 Data Preview", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Column information
    with st.expander("📈 Column Information"):
        col_types = profile['column_types']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numerical Columns**")
            if col_types['numerical']:
                for col in col_types['numerical']:
                    st.text(f"• {col}")
            else:
                st.text("None")
        
        with col2:
            st.markdown("**Categorical Columns**")
            if col_types['categorical']:
                for col in col_types['categorical']:
                    st.text(f"• {col}")
            else:
                st.text("None")
    
    # Missing values
    if profile['missing_values']['total_missing'] > 0:
        with st.expander("⚠️ Missing Values"):
            missing_df = pd.DataFrame({
                'Column': profile['missing_values']['columns_with_missing'].keys(),
                'Missing %': profile['missing_values']['columns_with_missing'].values()
            }).sort_values('Missing %', ascending=False)
            
            st.dataframe(missing_df, use_container_width=True)


def handle_model_training():
    """Handle model training interface"""
    st.markdown('<p class="sub-header">🎯 Model Training</p>', unsafe_allow_html=True)
    
    df = st.session_state.df
    
    # Target column selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target_column = st.selectbox(
            "Select Target Column",
            options=df.columns.tolist(),
            help="Choose the column you want to predict"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        train_button = st.button("🚀 Train Models", use_container_width=True, type="primary")
    
    if target_column:
        # Display target info
        profiler = DataProfiler(df)
        target_info = profiler.get_target_column_info(target_column)
        
        with st.expander("🎯 Target Column Information"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Data Type", target_info['dtype'])
            with col2:
                st.metric("Unique Values", target_info['n_unique'])
            with col3:
                st.metric("Missing", f"{target_info['missing_percentage']:.1f}%")
    
    if train_button and target_column:
        train_models(df, target_column)


def train_models(df: pd.DataFrame, target_column: str):
    """Train and evaluate models"""
    
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Detect problem type
    with st.spinner("Detecting problem type..."):
        detector = ProblemDetector(y)
        problem_type, confidence, details = detector.detect_problem_type()
        
        st.info(f"🔍 Detected Problem Type: **{problem_type.upper()}** (Confidence: {confidence*100:.1f}%)")
    
    # Validate target
    is_valid, message = detector.validate_target()
    if not is_valid:
        st.error(f"✗ Invalid target: {message}")
        return
    
    # Preprocessing
    with st.spinner("Preprocessing data..."):
        preprocessor = AutoPreprocessor(problem_type)
        X_transformed, y_transformed = preprocessor.fit_transform(X, y)
        
        prep_info = preprocessor.get_preprocessing_info()
        st.success(f"✓ Preprocessing complete: {prep_info['n_features_in']} → {prep_info['n_features_out']} features")
    
    # Model training
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Training models..."):
        status_text.text("Training multiple models with hyperparameter tuning...")
        
        trainer = ModelTrainer(problem_type)
        results = trainer.train_and_evaluate(X_transformed, y_transformed)
        
        progress_bar.progress(100)
        status_text.text("Training complete!")
    
    # Display results
    st.success("✓ Model training completed!")
    
    # Results table
    st.markdown("### 📊 Model Comparison")
    results_df = trainer.get_results_dataframe()
    st.dataframe(results_df, use_container_width=True)
    
    # Best model info
    best_model_info = trainer.get_best_model_info()
    
    st.markdown("### 🏆 Best Model")
    st.markdown(f"**{best_model_info['name']}**")
    
    col1, col2, col3 = st.columns(3)
    
    metrics = best_model_info['test_metrics']
    metric_items = list(metrics.items())
    
    for i, (metric, value) in enumerate(metric_items[:3]):
        with [col1, col2, col3][i]:
            if value is not None:
                st.metric(metric.upper(), f"{value:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{problem_type}_{best_model_info['name'].replace(' ', '_')}_{timestamp}.joblib"
    model_path = os.path.join(MODELS_DIR, model_filename)
    
    with st.spinner("Saving model..."):
        selector = ModelSelector(
            preprocessor=preprocessor,
            model=trainer.best_model,
            model_name=best_model_info['name'],
            problem_type=problem_type,
            model_info=best_model_info
        )
        
        saved_path = selector.save_model(model_path)
        st.session_state.model_path = saved_path
    
    st.success(f"✓ Model saved: {model_filename}")
    
    # Download button
    with open(model_path, 'rb') as f:
        st.download_button(
            label="⬇️ Download Trained Model",
            data=f,
            file_name=model_filename,
            mime='application/octet-stream',
            use_container_width=True
        )
    
    st.session_state.trained = True
    st.session_state.results = results


if __name__ == "__main__":
    main()