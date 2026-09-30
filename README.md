# Fraud Detection ML Platform

An end-to-end credit card fraud detection system built with Python, scikit-learn, MLflow, Flask, and Docker.

This project explores the complete machine learning lifecycle — from raw transaction data and feature engineering to model training, evaluation, experiment tracking, deployment, and monitoring.

The goal is to build a reproducible, testable, and production-oriented ML system while developing a deep understanding of classical machine learning algorithms.

## Project Objectives

- Build a complete ML pipeline from raw data to production inference.
- Understand data preprocessing, feature engineering, and leakage prevention.
- Implement and evaluate classical ML algorithms using scikit-learn.
- Handle highly imbalanced classification problems.
- Evaluate models using fraud-specific metrics and business costs.
- Track experiments, parameters, metrics, and artifacts with MLflow.
- Deploy a trained model through a Flask service.
- Implement automated testing, model versioning, and monitoring.

## Dataset

**Credit Card Fraud Detection — ULB / Worldline**

Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

The dataset contains 284,807 transactions, including 492 fraudulent transactions.

| Feature | Description |
|---|---|
| Time | Seconds elapsed from the first transaction |
| V1–V28 | Anonymized PCA-transformed features |
| Amount | Transaction amount |
| Class | Target: 0 = legitimate, 1 = fraud |

The dataset is highly imbalanced and covers a limited time period. It is intended for experimentation and learning, not as a complete representation of a production banking system.

Download the dataset from Kaggle and place `creditcard.csv` in `data/raw/`. The dataset is not included in this repository.

## Machine Learning Approach

The project uses classical machine learning — no neural networks or deep learning frameworks.

### Models

- DummyClassifier — baseline
- Logistic Regression — interpretable classification baseline
- Decision Tree — nonlinear decision boundaries
- Random Forest — tree-based ensemble learning
- HistGradientBoostingClassifier — gradient-boosted trees
- Isolation Forest — optional unsupervised anomaly detection experiment

### ML Pipeline

1. Data ingestion and schema validation
2. Exploratory data analysis
3. Data quality checks and duplicate analysis
4. Chronological train, validation, and test splitting
5. Feature engineering and preprocessing
6. Baseline model development
7. Model training and hyperparameter tuning
8. Evaluation and threshold selection
9. Model serialization and versioning
10. API deployment and monitoring

All preprocessing steps are designed to avoid data leakage and to ensure consistent transformations during training and inference.

## Evaluation Metrics

Accuracy alone is insufficient for fraud detection because fraudulent transactions are rare.

The project evaluates models using:

- Precision
- Recall
- F1-score
- Average Precision (AP)
- Precision-Recall curves
- ROC-AUC
- Confusion matrix
- Brier score and probability calibration
- False-positive and false-negative costs

Decision thresholds are selected using validation data and an explicit cost-based objective. Final test data is kept separate from model selection.

## Experiment Tracking with MLflow

MLflow is used to track and compare machine learning experiments.

Tracked information includes:

- Model parameters and hyperparameters
- Dataset and split configuration
- Evaluation metrics
- Training duration
- Model artifacts
- Evaluation plots and reports
- Model versions and metadata

The Model Registry can be used to manage model versions and control which validated model is deployed.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data processing | NumPy, pandas, SciPy |
| Machine learning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Experiment tracking | MLflow |
| API | Flask |
| Testing | pytest |
| Model serialization | joblib |
| Containerization | Docker |

## Project Structure

```text
fraud-detection-ml-platform/
├── configs/
│   ├── train.yaml
│   └── model.yaml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   └── fraud_detection/
│       ├── data/
│       │   ├── ingest.py
│       │   ├── validation.py
│       │   └── split.py
│       ├── features.py
│       ├── train.py
│       ├── tune.py
│       ├── evaluate.py
│       ├── thresholds.py
│       ├── predict.py
│       └── monitoring.py
├── api/
│   └── main.py
├── tests/
├── reports/
├── artifacts/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-ml-platform.git
cd fraud-detection-ml-platform
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download the dataset from Kaggle and place the CSV file at:

```text
data/raw/creditcard.csv
```

### 5. Run the pipeline

The following commands represent the intended workflow. Run them as the corresponding modules are implemented.

```bash
python -m fraud_detection.data.ingest
python -m fraud_detection.train
python -m fraud_detection.evaluate
```

### 6. Launch MLflow

```bash
mlflow server --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to explore experiment runs.

### 7. Run the API

```bash
 python main.py
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the API documentation.

## Testing

Run the automated test suite:

```bash
pytest -v
```

Tests are intended to cover data validation, preprocessing, model predictions, evaluation logic, artifact loading, and API behavior.

## Monitoring and Model Lifecycle

The planned monitoring workflow includes:

- Input schema and data quality checks
- Feature distribution and data drift analysis
- Prediction distribution monitoring
- API latency and error monitoring
- Performance evaluation when ground-truth labels become available
- Model versioning, validation, and controlled retraining

Drift detection is treated as a signal for investigation rather than an automatic reason to deploy a new model.

## Engineering Principles

- Reproducible experiments
- Separation of training and inference
- Leakage-safe preprocessing
- Time-aware model evaluation
- Explicit decision thresholds
- Automated testing
- Versioned model artifacts
- Traceable experiments
- Controlled model promotion

## Limitations

- The benchmark dataset is anonymized and covers a short time period.
- The data does not provide the complete transaction history required to simulate a real bank's online fraud system.
- Model performance on this dataset does not establish performance on live financial transactions.
- Production deployment would require additional security, privacy, latency, and operational controls.

## Learning Outcomes

This project is designed to develop practical skills in:

- Classical ML algorithms and their mathematical foundations
- Imbalanced classification and cost-sensitive evaluation
- Data engineering and feature pipelines
- Experiment tracking and model management
- Model testing and reproducibility
- REST API deployment
- ML monitoring and lifecycle management

---

**Status:** In development

Built to understand machine learning from raw data to production — one pipeline component at a time.
