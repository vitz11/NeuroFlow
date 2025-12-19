"""
Unit tests for AutoML pipelines
"""
import unittest
import pandas as pd
from pipelines import DataProfiler, Preprocessor, ProblemDetector


class TestDataProfiler(unittest.TestCase):
    """Test DataProfiler class"""
    
    def setUp(self):
        self.df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })
        self.profiler = DataProfiler(self.df)
    
    def test_basic_info(self):
        """Test basic info extraction"""
        info = self.profiler.get_basic_info()
        self.assertEqual(info['shape'], (5, 3))
        self.assertEqual(len(info['columns']), 3)


class TestPreprocessor(unittest.TestCase):
    """Test Preprocessor class"""
    
    def setUp(self):
        self.df = pd.DataFrame({
            'feature1': [1, 2, None, 4, 5],
            'feature2': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
        self.preprocessor = Preprocessor(self.df, target_column='target')
    
    def test_missing_value_handling(self):
        """Test missing value handling"""
        processed = self.preprocessor.handle_missing_values(strategy='mean')
        self.assertFalse(processed.isnull().any().any())


class TestProblemDetector(unittest.TestCase):
    """Test ProblemDetector class"""
    
    def setUp(self):
        self.df_classification = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })
        self.df_regression = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [1.5, 2.3, 3.8, 4.2, 5.1]
        })
    
    def test_classification_detection(self):
        """Test classification problem detection"""
        detector = ProblemDetector(self.df_classification, 'target')
        problem_type = detector.detect_problem_type()
        self.assertEqual(problem_type.value, 'classification')


if __name__ == '__main__':
    unittest.main()
