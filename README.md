
```markdown
# 🌧️ Australian Rain Prediction System (MLOps)

An end-to-end, production-grade Machine Learning Engineering and MLOps project designed to predict whether it will rain tomorrow in Australia. Built using the classic **Rain in Australia** dataset, this project implements professional software engineering principles, modular pipeline architecture, strict logging, custom exception handling, containerization, and automated Continuous Integration / Continuous Deployment (CI/CD) workflows.

---

## 🚀 Project Overview

The primary objective of this project is to build a robust binary classification engine capable of predicting the `RainTomorrow` target variable based on various meteorological metrics collected from numerous weather stations across Australia (such as temperature, humidity, wind speed, atmospheric pressure, and rainfall levels). 

Beyond training a high-performing classification model, this project showcases a comprehensive **MLOps Lifecycle**, ensuring reproducibility, reliability, and ease of deployment.

### 🌟 Key Features
- **Modular Pipeline Architecture:** Separate components for Data Ingestion, Data Transformation, Model Training, and Model Evaluation.
- **Robust Preprocessing & Feature Engineering:** Automated handling of missing values, encoding for categorical variables, scaling for numerical attributes, and handling class imbalances.
- **Production-Grade Web Interface:** A clean Flask application (`application.py`) that serves real-time model predictions via a web-based user interface.
- **Enterprise Utilities:** Centralized custom logger and custom exception handlers to trace and resolve errors rapidly across the entire system.
- **Cross-Platform MLOps Infrastructure:** Architected, migrated, and verified across multiple major SCM platforms and automation engines (GitHub, GitLab, and CircleCI)[cite: 1, 2].
- **Containerization:** A fully optimized Docker setup for consistent multi-environment runtime environments.

---

## 📂 Project Structure & Directory Layout

```text
├── .circleci/
│   └── config.yml               # CircleCI pipeline configuration[cite: 1, 2]
├── .github/
│   └── workflows/
│       └── deploy.yml           # GitHub Actions workflow configuration[cite: 1, 2]
├── .gitlab-ci.yml               # GitLab CI/CD pipeline configuration
├── artifacts/
│   ├── raw/
│   │   └── data.csv             # Raw source dataset (Rain in Australia)
│   └── processed/
│       ├── preprocessor.pkl     # Saved Scikit-Learn preprocessing pipeline
│       ├── X_train_resampled.pkl # Processed and balanced training features[cite: 1, 2]
│       ├── X_test_processed.pkl  # Processed evaluation features[cite: 1, 2]
│       ├── y_train_resampled.pkl # Balanced training labels
│       └── y_test.pkl            # Evaluation labels
├── notebook/
│   └── notebook.ipynb           # Exploratory Data Analysis & experimental models
├── pipeline/
│   └── training_pipeline.py     # Orchestrator to execute the end-to-end training flow
├── src/
│   ├── __init__.py              # Initializer marking src as a Python package
│   ├── custom_exception.py      # Extended custom error catching utilities
│   ├── data_processing.py       # Data validation, cleaning, and preprocessing logic
│   ├── logger.py                # Runtime event logging setup
│   └── model_training.py        # Model definition, hyperparameter tuning & evaluation
├── static/
│   └── style.css                # CSS styles for the Flask web application UI
├── templates/
│   └── index.html               # Main frontend interface for the web application
├── .gitignore                   # Specified files and directories ignored by Git[cite: 1, 2]
├── Dockerfile                   # Docker instruction recipe for container deployment[cite: 1, 2]
├── application.py               # Flask main driver entry-point for web application[cite: 1, 2]
├── requirements.txt             # List of third-party python dependencies
└── setup.py                     # Setup script for packaging the repository as a library

```

---

## 📊 Dataset & Feature Glossary

The model processes historical daily weather observation data from the Australian Bureau of Meteorology.

| Feature Name | Description | Type |
| --- | --- | --- |
| `Date` | The date of the weather observation | Temporal

 |
| `Location` | Name of the weather station/location | Categorical

 |
| `MinTemp` / `MaxTemp` | Minimum and maximum temperatures recorded (°C) | Numerical

 |
| `Rainfall` | Amount of rainfall recorded for the day (mm) | Numerical

 |
| `Evaporation` | Class A pan evaporation (mm) in the 24 hours to 9am | Numerical

 |
| `Sunshine` | Number of hours of bright sunshine in the day | Numerical

 |
| `WindGustDir` / `WindGustSpeed` | Direction and speed (km/h) of the strongest wind gust | Mixed

 |
| `WindSpeed9am` / `WindSpeed3pm` | Wind speed averaged over 10 minutes prior to 9am/3pm | Numerical

 |
| `Humidity9am` / `Humidity3pm` | Relative humidity percentage at 9am/3pm | Numerical

 |
| `Pressure9am` / `Pressure3pm` | Atmospheric pressure reduced to mean sea level at 9am/3pm | Numerical

 |
| `Cloud9am` / `Cloud3pm` | Fraction of sky obscured by cloud at 9am/3pm (in oktas) | Numerical

 |
| `Temp9am` / `Temp3pm` | Dry-bulb temperature at 9am/3pm (°C) | Numerical

 |
| `RainToday` | Boolean indicating if precipitation exceeded 1mm in the 24 hours to 9am | Categorical

 |
| **`RainTomorrow` (Target)** | **Boolean indicating whether it rained the following day (Yes/No)** | **Target Binary**<br> |

---

## ⚙️ Core Modules & Workflow Pipeline

### 1. Data Processing (`src/data_processing.py`)

* **Imputation:** Handles missing values using strategy-based imputation (e.g., median for skewed numerical distributions, mode for categorical records).


* **Encoding:** Converts string values (`Location`, `WindGustDir`, `RainToday`) into numerical forms using Standard or One-Hot Encoding methods.


* **Scaling:** Normalizes features using `StandardScaler` to remove scale bias for gradient-based or distance-based estimators.


* **Resampling:** Addresses structural class imbalances (fewer rainy days than dry days) to maximize model sensitivity toward positive instances (`RainTomorrow = Yes`).



### 2. Model Training (`src/model_training.py`)

* Standardized interfaces to test, tune, and evaluate multiple estimators (such as Logistic Regression, Random Forests, or Gradient Boosting classifiers).


* Serialization of the top-performing model and the preprocessor state artifact into the `artifacts/processed/` repository folder for live production scoring.



### 3. Pipeline Execution (`pipeline/training_pipeline.py`)

* Combines preprocessing and training steps into a single executable automated flow.


* Running this script end-to-end updates your production-ready serialization wrappers instantly.



### 4. Production Web Server (`application.py`)

* Exposes a standard `/predict` endpoint wrapped in an interactive web portal.


* Receives custom feature values typed by end-users, passes inputs to the preprocessor pipeline, and flashes the real-time weather outlook status immediately.



---

## 🛠️ Installation & Local Setup

### Prerequisites

* Python 3.10 or higher installed.


* Git configuration active.



### 1. Clone the Repository

```bash
git clone [https://github.com/himanshub0810/australian_rain_predication-mlops-7-.git](https://github.com/himanshub0810/australian_rain_predication-mlops-7-.git)
cd australian_rain_predication-mlops-7-

```

### 2. Establish a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt

```

### 4. Run the End-to-End Training Pipeline

To ingest data, transform features, tune hyperparameters, and save production artifacts locally:

```bash
python pipeline/training_pipeline.py

```

### 5. Launch the Web Interface Locally

```bash
python application.py

```

Open your preferred web browser and navigate to **`http://127.0.0.1:5000`** to interact with the system application.

---

## 🐳 Docker Deployment

The system features complete container environment configuration files to eliminate host environment mismatches.

### 1. Build the Docker Image

```bash
docker build -t australian-rain-predictor:latest .

```

### 2. Launch the Application Container

```bash
docker run -p 5000:5000 australian-rain-predictor:latest

```

Access the dashboard immediately at `http://localhost:5000`.

---

## 🔄 Infrastructure & DevOps Evolution

This project underwent an iterative evolutionary path across various industry-leading Source Control Management (SCM) systems and DevOps engines, serving as an operational showcase for agile, platform-agnostic MLOps engineering:

1. **Phase 1 (GitHub SCM + GitHub Actions):** The repository was initialized on GitHub, utilizing GitHub Actions (`.github/workflows/deploy.yml`) to orchestrate our baseline automated verification, dependency checking, and environment test cycles.


2. **Phase 2 (CircleCI Integration):** To scale the integration stack and accelerate execution turnaround, the project was wired into CircleCI (`.circleci/config.yml`), utilizing advanced performance layer caching and isolated build environments.


3. **Phase 3 (Migration to GitLab SCM & CI/CD):** Transitioned the entire source control management to GitLab to unify SCM and automation. The modern production infrastructure is fully configured via a native `.gitlab-ci.yml` pipeline file, dynamically running tasks for data version checks, image building, and live deployments.

---

## 🛡️ Enterprise Logging & Exception Framework

* **Logging (`src/logger.py`):** Captures multi-level runtime logs (`INFO`, `WARNING`, `ERROR`) stamped with timing, script source references, and line traces to file logs inside custom-dated folders.


* **Custom Exceptions (`src/custom_exception.py`):** Automatically intercepts internal Python system errors and formats descriptive error blocks showing line numbers and module positions to dramatically accelerate resolution times.



---



