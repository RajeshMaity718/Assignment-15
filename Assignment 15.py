import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression
)

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# Load Dataset
# ==================================================

df = pd.read_csv("dataset.csv")

print("Dataset Shape:")
print(df.shape)

print(df.head())

# ==================================================
# PART 1 : REGRESSION ALGORITHM
# ==================================================

# ==================================================
# Task 1 : Linear Regression
# ==================================================

numerical_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

# Select Target Column
target_regression = numerical_columns[-1]

X_reg = df[numerical_columns[:-1]]

y_reg = df[target_regression]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.2,
        random_state=42
    )
)

linear_model = LinearRegression()

linear_model.fit(
    X_train_reg,
    y_train_reg
)

y_pred_reg = linear_model.predict(
    X_test_reg
)

print("\nLinear Regression Predictions:")
print(y_pred_reg[:10])

# Actual vs Predicted Plot

plt.figure(figsize=(8,5))

plt.scatter(
    y_test_reg,
    y_pred_reg
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")

plt.show()

# ==================================================
# PART 2 : REGRESSION METRICS
# ==================================================

# ==================================================
# Task 2 : Evaluation Metrics
# ==================================================

mae = mean_absolute_error(
    y_test_reg,
    y_pred_reg
)

mse = mean_squared_error(
    y_test_reg,
    y_pred_reg
)

rmse = np.sqrt(mse)

print("\nRegression Metrics")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)

print("""
Interpretation:

MAE  = Average absolute error

MSE  = Average squared error

RMSE = Square root of MSE
Lower values indicate better performance.
""")

# ==================================================
# PART 3 : CLASSIFICATION ALGORITHMS
# ==================================================

# Assume last column is classification target
# Modify according to your dataset

target_classification = df.columns[-1]

X_cls = df.drop(
    columns=[target_classification]
)

# Convert object columns into numbers

X_cls = pd.get_dummies(
    X_cls,
    drop_first=True
)

y_cls = df[target_classification]

# Convert target if categorical

if y_cls.dtype == "object":

    y_cls = pd.factorize(y_cls)[0]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = (
    train_test_split(
        X_cls,
        y_cls,
        test_size=0.2,
        random_state=42
    )
)

# ==================================================
# Task 3 : Logistic Regression
# ==================================================

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_cls,
    y_train_cls
)

logistic_pred = logistic_model.predict(
    X_test_cls
)

print("\nLogistic Regression Completed")

# ==================================================
# Task 4 : Naive Bayes
# ==================================================

nb_model = GaussianNB()

nb_model.fit(
    X_train_cls,
    y_train_cls
)

nb_pred = nb_model.predict(
    X_test_cls
)

print("\nNaive Bayes Completed")

# ==================================================
# Task 5 : KNN Classifier
# ==================================================

k_values = [3, 5, 7]

best_accuracy = 0
best_k = 0

for k in k_values:

    knn = KNeighborsClassifier(
        n_neighbors=k
    )

    knn.fit(
        X_train_cls,
        y_train_cls
    )

    pred = knn.predict(
        X_test_cls
    )

    acc = accuracy_score(
        y_test_cls,
        pred
    )

    print(f"K={k} Accuracy={acc:.4f}")

    if acc > best_accuracy:

        best_accuracy = acc
        best_k = k

print("\nBest K:", best_k)

knn_best = KNeighborsClassifier(
    n_neighbors=best_k
)

knn_best.fit(
    X_train_cls,
    y_train_cls
)

knn_pred = knn_best.predict(
    X_test_cls
)

# ==================================================
# PART 4 : CLASSIFICATION METRICS
# ==================================================

# ==================================================
# Task 6 : Evaluation Metrics
# ==================================================

models = {
    "Logistic Regression": logistic_pred,
    "Naive Bayes": nb_pred,
    "KNN": knn_pred
}

for name, prediction in models.items():

    print(f"\n{name}")

    print(
        "Accuracy:",
        accuracy_score(
            y_test_cls,
            prediction
        )
    )

    print(
        "Precision:",
        precision_score(
            y_test_cls,
            prediction,
            average='weighted'
        )
    )

    print(
        "Recall:",
        recall_score(
            y_test_cls,
            prediction,
            average='weighted'
        )
    )

    print(
        "F1 Score:",
        f1_score(
            y_test_cls,
            prediction,
            average='weighted'
        )
    )

    print("\nClassification Report")

    print(
        classification_report(
            y_test_cls,
            prediction
        )
    )

    print("\nConfusion Matrix")

    print(
        confusion_matrix(
            y_test_cls,
            prediction
        )
    )

# ==================================================
# PART 5 : MODEL BEHAVIOR
# ==================================================

# ==================================================
# Task 7 : Overfitting & Underfitting
# ==================================================

# Underfitting Model

underfit_model = KNeighborsClassifier(
    n_neighbors=50
)

underfit_model.fit(
    X_train_cls,
    y_train_cls
)

train_acc_under = underfit_model.score(
    X_train_cls,
    y_train_cls
)

test_acc_under = underfit_model.score(
    X_test_cls,
    y_test_cls
)

# Overfitting Model

overfit_model = KNeighborsClassifier(
    n_neighbors=1
)

overfit_model.fit(
    X_train_cls,
    y_train_cls
)

train_acc_over = overfit_model.score(
    X_train_cls,
    y_train_cls
)

test_acc_over = overfit_model.score(
    X_test_cls,
    y_test_cls
)

print("\nUnderfitting Model")

print("Training Accuracy:",
      train_acc_under)

print("Testing Accuracy:",
      test_acc_under)

print("\nOverfitting Model")

print("Training Accuracy:",
      train_acc_over)

print("Testing Accuracy:",
      test_acc_over)

print("""

Explanation:

Underfitting:
Model is too simple.
Both training and testing accuracy remain low.

Overfitting:
Model memorizes training data.
Training accuracy becomes very high.
Testing accuracy becomes lower.

""")

# ==================================================
# Task 8 : Bias & Variance
# ==================================================

print("""

1. What is Bias?

Bias is error caused by overly simple
assumptions in the model.

High Bias -> Underfitting

------------------------------------------------

2. What is Variance?

Variance is error caused because the
model is too sensitive to training data.

High Variance -> Overfitting

------------------------------------------------

3. Relationship

High Bias  -> Underfitting

High Variance -> Overfitting

Good models maintain a balance between both.

------------------------------------------------

4. How to Reduce Overfitting?

Use more training data

Feature selection

Regularization

Cross Validation

Simpler models

Increase K in KNN

------------------------------------------------

""")
