# 🛡️ FinGuard – Indian Banking Transaction Fraud Detection

### Machine Learning | Python | Scikit-learn | Streamlit

## 📌 Project Overview

FinGuard is a machine learning-based banking transaction fraud detection system designed to identify potentially fraudulent transactions using historical transaction data.

The project analyzes transaction patterns, customer information, account characteristics and transaction behaviour to classify transactions as legitimate or potentially fraudulent.

It also provides an interactive Streamlit dashboard where users can upload transaction datasets and receive fraud-risk predictions.

## 🎯 Project Objectives

- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Identify patterns in fraudulent transactions.
- Apply feature engineering techniques.
- Train and compare multiple machine learning algorithms.
- Evaluate models using classification metrics.
- Develop an interactive fraud detection dashboard.

## 📊 Dataset Information

**Dataset:** Indian Banking Transactions 2019–2024

**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/belbino/indian-banking-transactions-20192024)

- Records: 550,000
- Original features: 20
- Target variable: `is_fraud`
- 0 = Legitimate transaction
- 1 = Fraudulent transaction

The dataset is not included in this repository. Download it from Kaggle and place the CSV file inside the `data/` directory.

## 🧹 Data Processing

The project includes:

- Missing-value analysis and handling.
- Duplicate record detection.
- Data type conversion.
- Invalid-value checks.
- Categorical data processing.
- Feature engineering.
- Target leakage investigation.

## 📈 Exploratory Data Analysis

Visualizations include:

- Fraud vs legitimate transaction distribution.
- Transaction amount analysis.
- Fraud rate by transaction type.
- Fraud rate by transaction hour.
- Fraud rate by channel.
- Monthly transaction trends.

## 🤖 Machine Learning Algorithms

Five classification algorithms were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Extra Trees Classifier
5. SGD Classifier

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

Model selection considers precision-recall performance because fraud detection involves an imbalanced dataset.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computing |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| Jupyter Notebook | Experimentation |
| Streamlit | Interactive dashboard |
| Git & GitHub | Version control |

## 📁 Project Structure

```text
FINTECH/
│
├── data/
│   ├── indian_banking_transactions.csv
│   ├── cleaned_banking_transactions.csv
│   └── engineered_banking_transactions.csv
│
├── notebooks/
│   └── fraud.detection.ipynb
│
├── outputs/
│   ├── final_fraud_model.pkl
│   ├── decision_threshold.json
│   ├── experiment_summary.json
│   └── model_comparison.csv
│
├── visualizations/
│
├── reports/
│
├── src/
│
├── app.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sarthak-singh18/FinGuard-Banking-Fraud-Detection.git
```

### 2. Navigate to the project

```bash
cd FinGuard-Banking-Fraud-Detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows)

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the dataset

Download the CSV from Kaggle and place it in the `data/` directory.

### 7. Run the notebook

Open:

`notebooks/fraud.detection.ipynb`

Execute the notebook to reproduce the data processing and model experiments.

### 8. Run the Streamlit application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

The saved model, threshold and experiment metadata must be available in `outputs/` before running the application.

## 🖥️ Application Features

- CSV transaction upload.
- Automated transaction analysis.
- Fraud probability estimation.
- Legitimate and fraud classification.
- Risk-level categorization.
- Downloadable prediction results.

## ⚠️ Limitations

The dataset may contain synthetic fraud labels influenced by transaction-generation rules. Therefore, model performance on this dataset should not be interpreted as proof of effectiveness on real-world banking fraud.

FinGuard is an educational and portfolio demonstration, not a production banking security system. Predictions require independent validation before any real financial use.

## 👨‍💻 Author

**Sarthak Singh**

B.Tech – Computer Science and Engineering

GitHub: [sarthak-singh18](https://github.com/sarthak-singh18)

---

⭐ If you find this project interesting, consider starring the repository!