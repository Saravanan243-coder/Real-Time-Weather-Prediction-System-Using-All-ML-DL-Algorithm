import os
import json
import joblib

def load_model(model_path):
    if model_path.endswith('.pkl'):
        return joblib.load(model_path)
    return None

def get_model_metrics(model_name, models_dir):
    metrics_path = os.path.join(models_dir, f"{model_name}_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    return None

def get_all_ml_models():
    return [
        {'name': 'logistic_regression', 'display_name': 'Logistic Regression', 'icon': '📈', 'desc': 'A statistical model that uses a logistic function to model a binary dependent variable.'},
        {'name': 'decision_tree', 'display_name': 'Decision Tree', 'icon': '🌲', 'desc': 'A tree-like model of decisions and their possible consequences.'},
        {'name': 'random_forest', 'display_name': 'Random Forest', 'icon': '🌳', 'desc': 'An ensemble learning method operating by constructing a multitude of decision trees.'},
        {'name': 'svm', 'display_name': 'Support Vector Machine', 'icon': '✂️', 'desc': 'Supervised learning models that analyze data for classification and regression analysis.'},
        {'name': 'knn', 'display_name': 'K-Nearest Neighbors', 'icon': '🏘️', 'desc': 'A non-parametric classification method evaluating the k closest training examples.'},
        {'name': 'naive_bayes', 'display_name': 'Naive Bayes', 'icon': '🧮', 'desc': 'Probabilistic classifiers based on applying Bayes theorem with strong independence assumptions.'},
        {'name': 'gradient_boosting', 'display_name': 'Gradient Boosting', 'icon': '🚀', 'desc': 'A machine learning technique for regression and classification problems.'},
        {'name': 'xgboost', 'display_name': 'XGBoost', 'icon': '⚡', 'desc': 'An optimized distributed gradient boosting library designed to be highly efficient.'}
    ]

def get_all_dl_models():
    return [
        {'name': 'ann', 'display_name': 'Artificial Neural Network', 'icon': '🧠', 'desc': 'A basic neural network architecture with hidden layers for pattern recognition.'},
        {'name': 'dnn', 'display_name': 'Deep Neural Network', 'icon': '🏗️', 'desc': 'A neural network with multiple hidden layers for complex feature extraction.'},
        {'name': 'lstm', 'display_name': 'Long Short-Term Memory', 'icon': '⏳', 'desc': 'A recurrent neural network architecture well-suited for sequential or time-series data.'},
        {'name': 'cnn_1d', 'display_name': '1D Convolutional NN', 'icon': '🌊', 'desc': 'A convolutional neural network optimized for extracting spatial features from 1D data.'}
    ]
