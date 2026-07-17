import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data_path, save_dir):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Drop irrelevant columns
    df.drop(['Date', 'Location'], axis=1, inplace=True, errors='ignore')
    
    # Define categorical and numerical columns
    categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    numerical_cols = [col for col in df.columns if col not in categorical_cols and col != 'RainTomorrow']
    
    print("Handling missing values...")
    # Handle missing values
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
        
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
        
    # Drop rows where target is missing
    df.dropna(subset=['RainTomorrow'], inplace=True)
    
    print("Encoding categorical variables...")
    # Initialize dictionary to save encoders
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    # Encode target
    le_target = LabelEncoder()
    df['RainTomorrow'] = le_target.fit_transform(df['RainTomorrow'])
    encoders['RainTomorrow'] = le_target
    
    # Save feature names
    feature_names = [col for col in df.columns if col != 'RainTomorrow']
    
    # Separate features and target
    X = df[feature_names]
    y = df['RainTomorrow']
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)
    
    print("Saving scaler and encoders...")
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(encoders, os.path.join(save_dir, 'encoders.pkl'))
    joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.pkl'))

    print("Saving feature medians for live-API fallback...")
    medians={col: float(df[col].median()) for col in numerical_cols}
    joblib.dump(medians, os.path.join(save_dir, 'feature_medians.pkl'))
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def get_wind_directions():
    return ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 
            'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

def preprocess_single_input(input_dict, save_dir):
    try:
        scaler = joblib.load(os.path.join(save_dir, 'scaler.pkl'))
        encoders = joblib.load(os.path.join(save_dir, 'encoders.pkl'))
        feature_names = joblib.load(os.path.join(save_dir, 'feature_names.pkl'))
        
        # Create a DataFrame from the input with only one row
        df = pd.DataFrame([input_dict])
        
        # Ensure all columns exist, fill missing with median/mode defaults just in case
        for col in feature_names:
            if col not in df.columns or pd.isna(df[col].iloc[0]) or df[col].iloc[0] == "":
                # We expect the form to provide all values, but just in case
                if col in encoders:
                    df[col] = 'W' # fallback default for wind
                    if col == 'RainToday':
                        df[col] = 'No'
                else:
                    df[col] = 0.0 # fallback for numerical
        
        # Ensure columns are in the exact same order as training
        df = df[feature_names]
        
        # Encode categorical
        categorical_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
        for col in categorical_cols:
            if col in encoders:
                try:
                    # Try to transform, if unseen label, use 0 (or most common)
                    df[col] = encoders[col].transform(df[col].astype(str))
                except ValueError:
                    df[col] = 0
                    
        # Scale
        X_scaled = scaler.transform(df)
        return X_scaled
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None
