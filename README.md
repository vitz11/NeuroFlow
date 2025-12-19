# � NeuroFlow - Intelligent Automated Machine Learning

An intelligent, production-ready automated machine learning system that handles the entire ML pipeline from data ingestion to model deployment. Built with Python, Streamlit, scikit-learn, and XGBoost.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Version](https://img.shields.io/badge/Version-1.0.0-brightgreen.svg)

## 🌟 Features

### Core Capabilities
- **🔍 Kaggle Integration**: Search and download datasets directly from Kaggle's repository
- **📁 File Upload**: Support for CSV and Excel files up to 500MB
- **📊 Automatic Data Profiling**: Comprehensive dataset analysis and quality scoring
- **🎯 Problem Detection**: Automatically detects classification vs regression tasks
- **⚙️ Smart Preprocessing**: Handles missing values, encoding, scaling, and feature selection
- **🤖 Multi-Model Training**: Trains 5-6 algorithms with hyperparameter tuning
- **📈 Model Evaluation**: Comprehensive metrics for both classification and regression
- **💾 Pipeline Export**: Saves complete preprocessing + model pipeline
- **📥 One-Click Download**: Export trained models for production use

### Technical Features
- Automated missing value imputation (median/mode strategies)
- One-hot encoding for categorical variables
- Standard scaling for numerical features
- Feature selection (variance threshold, correlation analysis)
- Cross-validation for robust evaluation
- Grid search hyperparameter tuning
- Complete pipeline serialization with Joblib

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT FRONTEND                       │
│   Data Upload │ Kaggle Search │ Configuration │ Results     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 DATA INGESTION LAYER                        │
│   Kaggle API Handler │ File Validator │ Data Loader        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             DATA PROFILING & ANALYSIS                       │
│   Statistical Analysis │ Quality Scoring │ Type Detection  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           AUTOMATED PREPROCESSING PIPELINE                  │
│   Missing Values → Encoding → Scaling → Feature Selection  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              MODEL TRAINING & TUNING                        │
│   Multiple Models │ Hyperparameter Tuning │ CV Evaluation  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            BEST MODEL SELECTION & EXPORT                    │
│   Model Comparison │ Pipeline Serialization │ Download     │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Technology Stack

| Technology | Purpose | Why |
|-----------|---------|-----|
| **Python 3.8+** | Core Language | Industry standard for ML/AI development |
| **Streamlit** | Web Framework | Rapid UI development, perfect for ML apps |
| **Scikit-learn** | ML Library | Comprehensive, well-documented, production-ready |
| **Pandas** | Data Manipulation | De facto standard for tabular data |
| **NumPy** | Numerical Computing | Fast array operations, foundation of ML stack |
| **XGBoost** | Advanced ML | State-of-the-art gradient boosting |
| **Kaggle API** | Dataset Access | Access to 50,000+ real-world datasets |
| **Joblib** | Serialization | Efficient pickling for sklearn objects |

## 📁 Project Structure

```
automl_system/
│
├── app/
│   ├── __init__.py
│   └── main.py                    # Streamlit application
│
├── pipelines/
│   ├── __init__.py
│   ├── data_loader.py             # Kaggle API + file handling
│   ├── data_profiler.py           # Dataset analysis
│   ├── preprocessor.py            # Automated preprocessing
│   ├── problem_detector.py        # Problem type detection
│   ├── model_trainer.py           # Model training & tuning
│   └── model_selector.py          # Model selection & export
│
├── utils/
│   ├── __init__.py
│   ├── config.py                  # Configuration constants
│   └── helpers.py                 # Utility functions
│
├── data/
│   ├── raw/                       # Raw datasets
│   └── processed/                 # Processed data (optional)
│
├── models/
│   └── exported/                  # Trained model exports
│
├── tests/
│   ├── __init__.py
│   └── test_pipelines.py          # Unit tests
│
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .gitignore                     # Git ignore file
└── kaggle.json.template           # Kaggle API template
```

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager
- Kaggle account (for dataset search feature)

### 2. Clone Repository
```bash
git clone https://github.com/yourusername/automl-system.git
cd automl-system
```

### 3. Create Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Setup Kaggle API (Optional)
To use the Kaggle dataset search feature:

1. Go to https://www.kaggle.com/account
2. Create API token (downloads `kaggle.json`)
3. Place in appropriate directory:
   - **Linux/Mac**: `~/.kaggle/kaggle.json`
   - **Windows**: `C:\Users\<Username>\.kaggle\kaggle.json`
4. Set permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 6. Run the Application
```bash
streamlit run app/main.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### Option 1: Upload Your Own Dataset

1. **Launch Application**: Run `streamlit run app/main.py`
2. **Select "Upload File"** in the sidebar
3. **Choose File**: Upload CSV or Excel file (max 500MB)
4. **Review Profile**: Examine dataset statistics and quality
5. **Select Target**: Choose the column you want to predict
6. **Train Models**: Click "Train Models" button
7. **Review Results**: Compare model performance metrics
8. **Download Model**: Export the best model for production use

### Option 2: Use Kaggle Dataset

1. **Select "Kaggle Dataset"** in the sidebar
2. **Search**: Enter keywords (e.g., "housing prices")
3. **Browse Results**: Review available datasets with metrics
4. **Download**: Click download on your preferred dataset
5. **Continue**: Follow steps 4-7 from Option 1

### Model Deployment

After training, you'll receive a `.joblib` file containing:
- Complete preprocessing pipeline
- Trained model
- Metadata (timestamps, metrics, feature info)

**To use the model in production:**
```python
import joblib
import pandas as pd

# Load the model
pipeline = joblib.load('path/to/model.joblib')

# Load new data
new_data = pd.read_csv('new_data.csv')

# Make predictions
preprocessor = pipeline['preprocessor']
model = pipeline['model']

X_transformed, _ = preprocessor.transform(new_data)
predictions = model.predict(X_transformed)
```

## 🎯 Supported Models

### Classification
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost Classifier

### Regression
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

## 📊 Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: Quality of positive predictions
- **Recall**: Ability to find all positive instances
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (binary only)

### Regression Metrics
- **RMSE** (Root Mean Squared Error): Average prediction error
- **MAE** (Mean Absolute Error): Average absolute error
- **R²** (R-Squared): Proportion of variance explained

## ⚙️ Configuration

Key parameters can be adjusted in `utils/config.py`:

```python
# Data Processing
MAX_FILE_SIZE_MB = 500
MISSING_THRESHOLD = 0.5
TEST_SIZE = 0.2
CV_FOLDS = 5

# Problem Detection
CLASSIFICATION_THRESHOLD = 20
REGRESSION_MIN_UNIQUE = 10

# Model Training
N_JOBS = -1  # Use all CPU cores
MAX_ITER = 1000
```

## 🔍 Example Workflows

### Example 1: Predicting House Prices (Regression)
```
1. Search Kaggle for "housing prices"
2. Download "House Prices - Advanced Regression Techniques"
3. Select target: "SalePrice"
4. System detects: REGRESSION (Confidence: 90%)
5. Best Model: XGBoost Regressor (R² = 0.89)
```

### Example 2: Customer Churn Prediction (Classification)
```
1. Upload customer_data.csv
2. Select target: "Churned" (Yes/No)
3. System detects: CLASSIFICATION (Confidence: 95%)
4. Best Model: Random Forest (Accuracy = 0.94)
```

## 🧪 Testing

Run unit tests:
```bash
python -m pytest tests/
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Kaggle API authentication error
```
Solution: Ensure kaggle.json is in the correct directory with proper permissions
```

**Issue**: Out of memory error with large datasets
```
Solution: Reduce dataset size or increase system RAM
```

**Issue**: Model training takes too long
```
Solution: Adjust hyperparameter search space in config.py
```

## 🚀 Future Enhancements

- [ ] Deep Learning models (Neural Networks)
- [ ] Advanced feature engineering (polynomial features, interaction terms)
- [ ] Automated imbalanced data handling (SMOTE, class weighting)
- [ ] Model interpretability (SHAP, feature importance)
- [ ] Real-time prediction API
- [ ] Cloud deployment (AWS/GCP/Azure)
- [ ] A/B testing framework
- [ ] Model monitoring dashboard
- [ ] Time series support
- [ ] Text and image data support

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

## 🙏 Acknowledgments

- Scikit-learn team for excellent ML library
- Streamlit team for amazing web framework
- Kaggle for dataset API and community
- XGBoost developers for powerful gradient boosting

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com
- Documentation: [Link to docs]

---

**⭐ If you find this project helpful, please give it a star!**