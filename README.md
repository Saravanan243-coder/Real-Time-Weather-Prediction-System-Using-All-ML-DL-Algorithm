# 🌦️ Weather ML Lab

A full-stack Flask web application that predicts rainfall using **9 Machine Learning models** and **4 Deep Learning (PyTorch) models**, powered by live weather data from the OpenWeatherMap API.

---

## 🚀 Features

- **Live Weather Data** — Fetches real-time weather data via OpenWeatherMap API for any city (with autocomplete search)
- **13 Prediction Models** — Compare predictions across 9 classic ML algorithms and 4 PyTorch deep learning models
- **JSON-based Authentication** — Simple, lightweight user login/signup system
- **Corporate Dashboard UI** — Clean, professional interface with city autocomplete search
- **Plain-Language Result Cards** — Predictions explained in simple, human-readable language (not just raw numbers)
- **Dynamic Weather-Mood Background** — Background gradient changes based on current weather conditions
- **Trained on Real Data** — Models trained on the Australian Weather Dataset (rain prediction)

---

## 📸 Screenshots

### Login Page
![Login Page](screenshots/login.png)

### Dashboard
![Dashboard](screenshots/dashboard.png)

### ML Models
![ML Models](screenshots/ml_models.png)

### DL Models
![DL Models](screenshots/dl_models.png)

### Metrics
![Metrics](screenshots/metrics.png)

### Predictions
![Predictions](screenshots/predictions.png)

---

## 🧠 Models Used

### Machine Learning (scikit-learn)
| # | Model |
|---|-------|
| 1 | Logistic Regression |
| 2 | Decision Tree |
| 3 | Random Forest |
| 4 | K-Nearest Neighbors (KNN) |
| 5 | Support Vector Machine (SVM) |
| 6 | Naive Bayes |
| 7 | Gradient Boosting |
| 8 | XGBoost |
| 9 | AdaBoost |

### Deep Learning (PyTorch)
| # | Model |
|---|-------|
| 1 | Feedforward Neural Network (ANN) |
| 2 | Deep Neural Network (DNN) |
| 3 | LSTM |
| 4 | Custom PyTorch Architecture |

> **Note:** TensorFlow is not used in this project due to compatibility issues with Python 3.14 — PyTorch is used for all deep learning models instead.

---

## 🏗️ Tech Stack

- **Backend:** Flask (Python)
- **ML/DL:** scikit-learn, PyTorch
- **Data:** Pandas, NumPy
- **Frontend:** HTML, CSS, Jinja2 Templates
- **Authentication:** JSON-based custom auth (`auth.py`)
- **Live Data Source:** OpenWeatherMap API
- **Dataset:** Australian Weather Dataset (historical rain data)

---

## 📁 Project Structure

```
weather_app/
│
├── static/
│   ├── models/            # Trained model files (.pkl, .pth) — generated locally, not in repo
│   └── ...                # CSS, JS, images
│
├── templates/              # Jinja2 HTML templates
│
├── train.py                 # Script to train all ML & DL models
├── predictor.py              # Loads models & runs predictions
├── app.py                    # Main Flask application
├── auth.py                    # JSON-based authentication logic
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Saravanan243-coder/Real-Time-Weather-Prediction-System-Using-All-ML-DL-Algorithm.git
cd Real-Time-Weather-Prediction-System-Using-All-ML-DL-Algorithm
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your OpenWeatherMap API key
Create a `.env` file in the root directory:
```
OPENWEATHER_API_KEY=your_api_key_here
```

### 5. Train the models
Model files (`.pkl`, `.pth`) are **not included in this repo** due to GitHub's file size limits. Generate them locally:
```bash
python train.py
```
This will train all 9 ML models and 4 DL models, saving them to `static/models/`.

### 6. Run the application
```bash
python app.py
```
Visit `http://localhost:5000` in your browser.

---

## 📊 Dataset

This project uses the **Australian Weather Dataset**, which contains historical daily weather observations across multiple Australian cities, used to predict whether it will rain the next day.

---

## 🔮 How Prediction Works

1. User selects a city (with autocomplete search)
2. App fetches live weather data from OpenWeatherMap API
3. Live data is fed into all 13 trained models
4. Each model generates a rain prediction
5. Results are displayed as easy-to-understand cards, with a background gradient that reflects the current weather mood

---

## 🛣️ Future Improvements

- [ ] Docker containerization for easy deployment
- [ ] Model performance comparison dashboard
- [ ] Historical prediction accuracy tracking
- [ ] Deploy to cloud (Render/Railway/AWS)

---

## 👤 Author

**Saravanan**
Aspiring Data Scientist | ML/DL Enthusiast
[GitHub](https://github.com/Saravanan243-coder)

---

## 📝 License

This project is open source and available for educational purposes.
