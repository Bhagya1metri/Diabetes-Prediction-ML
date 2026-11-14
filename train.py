
import os
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

# =====================================
# 1. LOAD DATASET
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "diabetes.csv")
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# =====================================
# 2. HANDLE MISSING VALUES
# =====================================

zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X = df.drop("Outcome", axis=1).copy()
y = df["Outcome"]

X[zero_columns] = X[zero_columns].replace(0, np.nan)

print("\nMissing values after zero replacement:")
print(X.isnull().sum())

# =====================================
# 3. SPLIT DATASET
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# =====================================
# 4. DEFINE MODELS
# =====================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Random Forest Balanced": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ),

    "SVM": SVC(
        probability=True,
        random_state=42
    )
}

# =====================================
# 5. CROSS VALIDATION
# =====================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_results = []

print("\n================================")
print("5-FOLD CROSS VALIDATION")
print("================================")

for name, classifier in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )

    recall_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="recall"
    )

    cv_results.append({
        "Model": name,
        "CV Accuracy (%)": round(scores.mean() * 100, 2),
        "CV Recall (%)": round(recall_scores.mean() * 100, 2)
    })

    print("\n", name)
    print("Fold Scores:", np.round(scores * 100, 2))
    print("Average Accuracy:", round(scores.mean() * 100, 2), "%")
    print("Average Recall:", round(recall_scores.mean() * 100, 2), "%")

# =====================================
# 6. TRAIN AND EVALUATE ALL MODELS
# =====================================

results = []
trained_models = {}
evaluation_reports = {}
roc_data = {}

for name, classifier in models.items():

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", classifier)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    auc = roc_auc_score(y_test, probabilities)

    trained_models[name] = pipeline

    evaluation_reports[name] = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    roc_data[name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc
    }

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Precision (%)": round(precision * 100, 2),
        "Recall (%)": round(recall * 100, 2),
        "F1-Score (%)": round(f1 * 100, 2),
        "ROC-AUC": round(auc, 4)
    })

    print("\n==============================")
    print(name)
    print("==============================")

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1-Score:", round(f1 * 100, 2), "%")
    print("ROC-AUC:", round(auc, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

# =====================================
# 7. CROSS VALIDATION COMPARISON
# =====================================

cv_results_df = pd.DataFrame(cv_results)

print("\nCROSS VALIDATION COMPARISON")
print(cv_results_df.to_string(index=False))

cv_results_df.to_csv(
    os.path.join(BASE_DIR, "cv_results.csv"),
    index=False
)

# =====================================
# 8. MODEL COMPARISON
# =====================================

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(BASE_DIR, "model_comparison.csv"),
    index=False
)

# =====================================
# 9. SELECT BALANCED RANDOM FOREST
# =====================================

rf_model = trained_models["Random Forest Balanced"]

rf_predictions = rf_model.predict(X_test)
rf_probabilities = rf_model.predict_proba(X_test)[:, 1]

rf_auc = roc_auc_score(y_test, rf_probabilities)

print("\nBalanced Random Forest ROC-AUC:", round(rf_auc, 4))

# =====================================
# 10. CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_test, rf_predictions)

print("\nRandom Forest Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Diabetes", "Diabetes"]
)

display.plot(cmap="Blues")

plt.title("Balanced Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================
# 11. ROC CURVE COMPARISON
# =====================================

plt.figure(figsize=(9, 7))

for name, values in roc_data.items():

    plt.plot(
        values["fpr"],
        values["tpr"],
        linewidth=2,
        label=f"{name} (AUC = {values['auc']:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Model Comparison")

plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nROC curve saved successfully!")

# =====================================
# 12. FEATURE IMPORTANCE
# =====================================

rf_classifier = rf_model.named_steps["classifier"]

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_classifier.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    os.path.join(BASE_DIR, "feature_importance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# =====================================
# 13. SAVE MODEL AND EVALUATION
# =====================================

joblib.dump(
    rf_model,
    os.path.join(BASE_DIR, "diabetes_model.pkl")
)

evaluation_results = {
    "roc_auc": rf_auc,
    "classification_report": evaluation_reports[
        "Random Forest Balanced"
    ],
    "roc_data": roc_data,
    "model_results": results_df.to_dict(orient="records"),
    "confusion_matrix": cm
}

joblib.dump(
    evaluation_results,
    os.path.join(BASE_DIR, "model_evaluation.pkl")
)

print("\nAll files saved successfully!")

print("\n================================")
print("MODEL TRAINING COMPLETED!")
print("================================")
