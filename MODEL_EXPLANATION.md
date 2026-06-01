# Supervised Learning Notes
## Regression and Classification using the Titanic Dataset

## Project Overview

This project demonstrates the fundamentals of Supervised Machine Learning using the Titanic dataset.

Two different machine learning problems are solved using the same dataset:

1. Regression
2. Classification

The main objective is to understand how changing the target variable transforms the problem and affects the model selection and evaluation metrics.

---

# What is Supervised Learning?

Supervised Learning is a Machine Learning paradigm where a model learns from labeled data.

Each training example contains:

- Features (input variables)
- Target (expected output)

The model learns patterns from historical data and uses them to make predictions on unseen observations.

## General Workflow

```text
Dataset
 ↓
Features + Target
 ↓
Train/Test Split
 ↓
Model Training
 ↓
Predictions
 ↓
Evaluation
```

---

# Regression Model

## Objective

Predict the ticket price (`Fare`) paid by Titanic passengers.

### Features

- Passenger Class (`Pclass`)
- Age (`Age`)
- Gender (`Sex`)

### Target

```python
Fare
```

Since the target variable is numeric and continuous, this is a Regression problem.

---

## Data Preparation

### Feature Selection

```python
df = df[["Fare", "Pclass", "Sex", "Age"]]
```

Only the variables relevant to the prediction task are selected.

---

### Categorical Encoding

```python
pd.get_dummies(df, columns=["Sex"], drop_first=True)
```

Machine Learning algorithms require numerical data.

The `Sex` column is transformed into:

| Original Value | Sex_male |
|---------------|-----------|
| Female | 0 |
| Male | 1 |

---

### Handling Missing Values

```python
df.dropna()
```

Rows containing missing values are removed before training.

---

## Train-Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Dataset split:

- 80% Training Data
- 20% Testing Data

This allows the model to be evaluated on unseen observations.

---

## Model

```python
LinearRegression()
```

Linear Regression is used because the target variable is continuous.

The model attempts to learn a mathematical relationship between the selected features and the ticket price.

---

## Training

```python
model.fit(X_train, y_train)
```

During training, the model learns patterns from historical examples.

For example:

- First-class passengers tend to have more expensive tickets.
- Third-class passengers tend to have cheaper tickets.

---

## Predictions

```python
y_pred = model.predict(X_test)
```

Example output:

```python
[35.28, 49.76, 50.00, 88.92, -2.14]
```

Each value represents a predicted ticket price.

---

# Regression Results

## Mean Absolute Error (MAE)

```text
25.09
```

### Interpretation

On average, the model makes an error of approximately $25 per prediction.

Lower values indicate better performance.

---

## R² Score

```text
0.194
```

### Interpretation

The model explains approximately:

```text
19.4%
```

of the variance in ticket prices.

This indicates that important explanatory variables are still missing.

---

## Regression Conclusion

The model successfully learned some relationships between passenger attributes and ticket prices.

However, the low R² score suggests limited predictive power and room for improvement through additional features.

---

# Classification Model

## Objective

Predict whether a passenger survived the Titanic disaster.

### Target

```python
Survived
```

Possible values:

```text
0 = Did Not Survive
1 = Survived
```

Since the output belongs to predefined categories, this is a Classification problem.

---

## Features

- Passenger Class (`Pclass`)
- Age (`Age`)
- Gender (`Sex`)
- Fare (`Fare`)

---

## Model

```python
LogisticRegression(max_iter=1000)
```

Logistic Regression is used because the target variable is categorical.

Although the name contains "Regression", it is primarily a classification algorithm.

---

## Why max_iter = 1000?

```python
max_iter=1000
```

Defines the maximum number of optimization iterations allowed during training.

This helps ensure convergence and prevents training from stopping too early.

---

## Training

```python
model.fit(X_train, y_train)
```

The model learns relationships between passenger attributes and survival outcomes.

Examples:

- Women had higher survival rates.
- First-class passengers were more likely to survive.
- Higher ticket prices often correlated with higher survival probability.

---

## Predictions

```python
y_pred = model.predict(X_test)
```

Example output:

```python
[0, 1, 1, 1, 0, 0, 1, 1, 1, 1]
```

Interpretation:

```text
0 = Did Not Survive
1 = Survived
```

---

# Classification Results

## Accuracy

```text
0.755
```

Equivalent to:

```text
75.5%
```

### Interpretation

Out of every 100 passengers:

- 76 were classified correctly.
- 24 were classified incorrectly.

---

## Precision

```text
0.68
```

### Interpretation

When the model predicts that a passenger survived, it is correct approximately 68% of the time.

---

## Recall

```text
0.71
```

### Interpretation

The model correctly identifies approximately 71% of all actual survivors.

---

## F1 Score

```text
0.70
```

### Interpretation

The model maintains a reasonable balance between Precision and Recall.

---

## Classification Conclusion

The model achieved an accuracy of 75.5%, demonstrating a reasonable ability to distinguish between survivors and non-survivors.

Compared to the regression model, the classification model produced stronger overall results.

---

# Key Takeaways

- Supervised Learning requires labeled data.
- The target variable determines the type of problem.
- Continuous targets lead to Regression models.
- Categorical targets lead to Classification models.
- The same dataset can solve different problems depending on the chosen target.
- Evaluation metrics are essential for measuring model performance.
- Model interpretation is as important as model training.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- UV

---

## Dataset

Titanic Dataset

A widely used dataset for learning and demonstrating Machine Learning concepts.