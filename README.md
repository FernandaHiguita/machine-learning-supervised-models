# Machine Learning - Supervised Models

This repository contains two introductory supervised machine learning exercises developed using the Titanic dataset and Scikit-Learn.

The project demonstrates the complete workflow of supervised learning, including:

- Data preprocessing
- Feature selection
- Handling categorical variables
- Train/Test split
- Model training
- Predictions
- Model evaluation

---

## Project Structure

```text
.
├── .venv/
├── regression_example.py
├── classification_example.py
├──data/
|   ├── Titanic-Dataset.csv
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Exercise 1 - Regression

### Objective

Predict the ticket fare (`Fare`) using:

- Passenger Class (`Pclass`)
- Sex (`Sex`)
- Age (`Age`)

### Algorithm

- Linear Regression

### Metrics

- Mean Absolute Error (MAE)
- R² Score

---

## Exercise 2 - Classification

### Objective

Predict passenger survival (`Survived`) using:

- Passenger Class (`Pclass`)
- Sex (`Sex`)
- Age (`Age`)
- Fare (`Fare`)

### Algorithm

- Logistic Regression

### Metrics

- Accuracy
- Precision
- Recall
- F1-Score

---

## Technologies

- Python 3
- Pandas
- NumPy
- Scikit-Learn
- UV Package Manager

---

## Environment Setup

This project uses UV for dependency management.

Install UV:

```bash
pip install uv
```

Create the virtual environment:

```bash
uv venv
```

Activate the environment:

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

Or manually:

```bash
uv add pandas numpy scikit-learn
```

---

## Running the Exercises

### Regression

```bash
python regression_example.py
```

### Classification

```bash
python classification_example.py
```

---

## Machine Learning Concepts Covered

- Supervised Learning
- Regression
- Classification
- Feature Engineering
- Train/Test Split
- Linear Regression
- Logistic Regression
- Model Evaluation
- MAE
- R²
- Accuracy
- Precision
- Recall
- F1 Score

---

## Dataset

Titanic Dataset

A classic dataset widely used to introduce supervised machine learning concepts.

---

## Author

Fernanda Higuita