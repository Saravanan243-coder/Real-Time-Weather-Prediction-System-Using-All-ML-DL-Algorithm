import os
import joblib
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from datetime import datetime
from utils.model_utils import get_all_ml_models, get_all_dl_models, get_model_metrics, load_model
from utils.data_preprocessing import preprocess_single_input, get_wind_directions
from utils.weather_api import get_weather_bundle, aggregate_daily, map_daily_to_features, map_forecast_entry_to_features

app = Flask(__name__)
app.secret_key = 'weather_secret_key_2024'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'static', 'models')
MEDIANS= joblib.load(os.path.join(MODELS_DIR,'feature_medians.pkl'))

_model_cache = {}
MAX_CACHED_MODELS = 3

def get_cached_model(model_name, model_path):
    if model_name not in _model_cache:
        if len(_model_cache) >= MAX_CACHED_MODELS:
            oldest = next(iter(_model_cache))
            del _model_cache[oldest]
            print(f"Evicted {oldest} from cache")
        _model_cache[model_name] = load_model(model_path)
        print(f"Loaded {model_name} into memory")
    return _model_cache[model_name]

# Dummy credentials
USERNAME = 'admin'
PASSWORD = 'password123'

@app.route('/')
def index():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username == USERNAME and password == PASSWORD:
            session['logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials. Try admin/password123', 'error')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/ml-models')
def ml_models():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    models = get_all_ml_models()
    return render_template('ml_models.html', models=models)

@app.route('/dl-models')
def dl_models():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    models = get_all_dl_models()
    return render_template('dl_models.html', models=models)

@app.route('/result/<model_type>/<model_name>')
def result(model_type, model_name):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
        
    metrics = get_model_metrics(model_name, MODELS_DIR)
    
    if not metrics:
        flash(f"Metrics not found for model: {model_name}. Have you run train_models.py?", "error")
        return redirect(url_for('dashboard'))
        
    if model_type == 'ml':
        model_info = next((m for m in get_all_ml_models() if m['name'] == model_name), None)
    else:
        model_info = next((m for m in get_all_dl_models() if m['name'] == model_name), None)
        
    display_name = model_info['display_name'] if model_info else model_name.title()
    wind_directions = get_wind_directions()
    
    return render_template('result.html', 
                           model_name=model_name, 
                           display_name=display_name, 
                           model_type=model_type,
                           metrics=metrics,
                           wind_directions=wind_directions)

@app.route('/predict/<model_name>', methods=['POST'])
def predict(model_name):
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
        
    try:
        data = request.json
        print(f"Prediction input: {data}")
        
        # Determine model extension and load it
        model_path_pkl = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        print(f"DEBUG - Looking for: {model_path_pkl}")
        print(f"DEBUG - Exists: {os.path.exists(model_path_pkl)}")
        print(f"DEBUG - MODELS_DIR: {MODELS_DIR}")
        if not os.path.exists(model_path_pkl):
            return jsonify({'error': 'Model file not found. Please train models first.'}), 404
        
        model =get_cached_model(model_name, model_path_pkl)
            
        # Preprocess input
        X_input = preprocess_single_input(data, MODELS_DIR)
        
        if X_input is None:
            return jsonify({'error': 'Preprocessing failed. Ensure encoders/scaler exist.'}), 500
            
        # Predict
        pred_class = int(model.predict(X_input)[0])
        if hasattr(model, 'predict_proba'):
            prob = float(model.predict_proba(X_input)[0][1])
        else:
            prob = 1.0 if pred_class == 1 else 0.0
                
        # Format response
        result_text = "Rain Tomorrow!" if pred_class == 1 else "No Rain Tomorrow"
        
        return jsonify({
            'prediction': 'Yes' if pred_class == 1 else 'No',
            'probability': round(prob, 4),
            'message': result_text
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500
    

@app.route('/live')
def live():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    excluded={'knn','decision_tree','naive_bayes'}
    ml_models = [m for m in get_all_ml_models() if m['name'] not in excluded]
    dl_models = get_all_dl_models()
    return render_template('live.html',ml_models=ml_models, dl_models=dl_models)

@app.route('/api/live-weather')
def api_live_weather():
    city = request.args.get('city', 'Chennai')
    bundle = get_weather_bundle(city)
    if not bundle or 'main' not in bundle.get('current', {}):
        return jsonify({'error': 'City not found or weather service unavailable'}), 404

    current = bundle['current']
    daily_agg = aggregate_daily(bundle['forecast_list'])
    dates_available = sorted(daily_agg.keys())

    return jsonify({
        'current': {
            'temp': current['main']['temp'],
            'humidity': current['main']['humidity'],
            'pressure': current['main']['pressure'],
            'wind_speed': round(current['wind']['speed'] * 3.6, 1),
            'clouds': current['clouds']['all'],
            'weather': current['weather'][0]['description'],
        },
        'daily_dates': dates_available
    })

@app.route('/api/predict-date', methods=['POST'])
def predict_date():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.json
        city = data.get('city', 'Chennai')
        model_name = data.get('model_name')
        target_date = data.get('date')

        bundle = get_weather_bundle(city)
        if not bundle:
            return jsonify({'error': 'City not found'}), 404

        daily_agg = aggregate_daily(bundle['forecast_list'])
        if target_date not in daily_agg:
            return jsonify({'error': 'Date must be within the next 5 days (free API limit)'}), 400

        sorted_dates = sorted(daily_agg.keys())
        idx = sorted_dates.index(target_date)
        prev_rain = daily_agg[sorted_dates[idx - 1]]['total_rain'] if idx > 0 else 0

        features = map_daily_to_features(daily_agg[target_date], MEDIANS, prev_day_rain=prev_rain)

        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404
        model = get_cached_model(model_name, model_path)

        X_input = preprocess_single_input(features, MODELS_DIR)
        pred_class = int(model.predict(X_input)[0])
        prob = float(model.predict_proba(X_input)[0][1]) if hasattr(model, 'predict_proba') else float(pred_class)

        return jsonify({
            'date': target_date,
            'prediction': 'Yes' if pred_class == 1 else 'No',
            'probability': round(prob, 4),
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict-next-hour', methods=['POST'])
def predict_next_hour():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.json
        city = data.get('city', 'Chennai')
        model_name = data.get('model_name')

        bundle = get_weather_bundle(city)
        if not bundle or not bundle['forecast_list']:
            return jsonify({'error': 'City not found'}), 404

        next_entry = bundle['forecast_list'][0]  # nearest upcoming 3-hour block
        features = map_forecast_entry_to_features(next_entry, MEDIANS)

        model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404
        model = get_cached_model(model_name, model_path)

        X_input = preprocess_single_input(features, MODELS_DIR)
        pred_class = int(model.predict(X_input)[0])
        prob = float(model.predict_proba(X_input)[0][1]) if hasattr(model, 'predict_proba') else float(pred_class)

        return jsonify({
            'prediction': 'Yes' if pred_class == 1 else 'No',
            'probability': round(prob, 4),
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
