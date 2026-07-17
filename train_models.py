import os
import json
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV

from utils.data_preprocessing import preprocess_data

# Set dark theme for matplotlib
plt.style.use('dark_background')

def save_metrics(model_name, save_dir, y_true, y_pred, y_prob, train_time):
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate Confusion Matrix image
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '..', 'images', f'{model_name}_confusion_matrix.png'))
    plt.close()
    
    # Generate ROC Curve image
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, color='#00d4ff', lw=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name.replace("_", " ").title()} ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '..', 'images', f'{model_name}_roc_curve.png'))
    plt.close()
    
    # Save to JSON
    metrics = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'classification_report': report,
        'training_time': train_time,
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(save_dir, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f)
        
    print(f"[{model_name}] Accuracy: {acc:.4f} | Time: {train_time:.2f}s")

def train_ml_models(X_train, X_test, y_train, y_test, models_dir):
    print("\n--- Training ML Models ---")
    
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=10, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=5),
        'naive_bayes': GaussianNB(),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'xgboost': HistGradientBoostingClassifier(random_state=42)
    }
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        # Wrap with calibration so predict_proba gives realistic confidence scores
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        calibrated_model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = calibrated_model.predict(X_test)
        y_prob = calibrated_model.predict_proba(X_test)[:, 1]
        
        save_metrics(name, models_dir, y_test, y_pred, y_prob, train_time)
        joblib.dump(calibrated_model, os.path.join(models_dir, f'{name}.pkl'), compress=3)

    # SVM specifically with subset because 145k takes forever
    print("Training svm (on subset of 20000)...")
    subset_idx = np.random.choice(len(X_train), size=20000, replace=False)
    X_train_sub, y_train_sub = X_train[subset_idx], y_train.iloc[subset_idx]
    
    svm_model = SVC(kernel='rbf', probability=True, random_state=42, cache_size=500)
    calibrated_svm = CalibratedClassifierCV(svm_model, method='sigmoid', cv=3)
    start_time = time.time()
    calibrated_svm.fit(X_train_sub, y_train_sub)
    train_time = time.time() - start_time
    
    y_pred = calibrated_svm.predict(X_test)
    y_prob = calibrated_svm.predict_proba(X_test)[:, 1]
    
    save_metrics('svm', models_dir, y_test, y_pred, y_prob, train_time)
    joblib.dump(calibrated_svm, os.path.join(models_dir, 'svm.pkl'), compress=3)

def train_dl_models(X_train, X_test, y_train, y_test, models_dir):
    print("\n--- Training DL Models (Scikit-Learn MLP Surrogates) ---")
    
    subset_size = min(30000, len(X_train))
    idx = np.random.choice(len(X_train), size=subset_size, replace=False)
    X_train_sub = X_train[idx]
    y_train_sub = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
    
    dl_models = {
        'ann': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42, early_stopping=True, validation_fraction=0.1),
        'dnn': MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), max_iter=50, random_state=42, early_stopping=True, validation_fraction=0.1),
        'lstm': MLPClassifier(hidden_layer_sizes=(100, 50), activation='tanh', max_iter=50, random_state=42, early_stopping=True, validation_fraction=0.1),
        'cnn_1d': MLPClassifier(hidden_layer_sizes=(200,), activation='relu', max_iter=50, random_state=42, early_stopping=True, validation_fraction=0.1)
    }
    
    for name, model in dl_models.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        calibrated_model.fit(X_train_sub, y_train_sub)
        train_time = time.time() - start_time
        
        y_pred = calibrated_model.predict(X_test)
        y_prob = calibrated_model.predict_proba(X_test)[:, 1]
        
        save_metrics(name, models_dir, y_test, y_pred, y_prob, train_time)
        joblib.dump(calibrated_model, os.path.join(models_dir, f'{name}.pkl'), compress=3)
        
def models_exist(models_dir, model_names):
    return all(os.path.exists(os.path.join(models_dir, f'{name}.pkl')) for name in model_names)

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root_dir, 'data', 'weatherAUS.csv')
    models_dir = os.path.join(root_dir, 'static', 'models')
    images_dir = os.path.join(root_dir, 'static', 'images')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(data_path, models_dir)

    train_ml_models(X_train, X_test, y_train, y_test, models_dir)
    train_dl_models(X_train, X_test, y_train, y_test, models_dir)
    print("\nAll models trained and saved successfully!")

    
if __name__ == "__main__":
    main()
