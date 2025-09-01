import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# ================================
# 1. Load and preprocess Adult dataset
# ================================
adult = fetch_openml(name="adult", version=2, as_frame=True)
X = adult.data
y = (adult.target == ">50K").astype(int)  # binary {0,1}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

categorical_cols = X.select_dtypes(include=["category", "object"]).columns
numeric_cols = X.select_dtypes(exclude=["category", "object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# ================================
# 2. Train base model θ_old
# ================================
clf_old = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=200, solver="lbfgs"))
])
clf_old.fit(X_train, y_train)

# ================================
# 3. Sample Wasserstein perturbations
# ================================
def sample_perturbations(X, eps_x):
    Xp = X.copy()
    noise = np.random.normal(0, 0.1, size=Xp[numeric_cols].shape)
    # Scale to satisfy constraint (1/n Σ ||δ||² ≤ eps_x)
    scale = np.sqrt(eps_x / (np.mean(np.linalg.norm(noise, axis=1)**2) + 1e-12))
    Xp[numeric_cols] += scale * noise
    return Xp

# ================================
# 4. Fine-tune model under perturbed dataset
# ================================
def fine_tune_model(X, y, eps_x):
    X_new = sample_perturbations(X, eps_x)
    clf_new = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=200, solver="lbfgs"))
    ])
    clf_new.fit(X_new, y)
    return clf_new

# ================================
# 5. Generate candidate counterfactual
# ================================
def generate_counterfactual(x, clf_old, target_class=1, max_steps=20):
    x_cf = x.copy()
    for step in range(max_steps):
        prob = clf_old.predict_proba(pd.DataFrame([x_cf]))[0, target_class]
        if prob >= 0.5:
            return x_cf
        # Simple heuristic: try to increase education-num or hours-per-week
        if "education-num" in x_cf:
            x_cf["education-num"] = min(x_cf["education-num"] + 1, 16)
        elif "hours-per-week" in x_cf:
            x_cf["hours-per-week"] = min(x_cf["hours-per-week"] + 5, 99)
    return x_cf

# ================================
# 6. Monte Carlo robustness evaluation
# ================================
def robustness_check(x_cf, X_train, y_train, eps_x, N=50, target_class=1):
    scores = []
    for i in range(N):
        clf_new = fine_tune_model(X_train, y_train, eps_x)
        prob = clf_new.predict_proba(pd.DataFrame([x_cf]))[0, target_class]
        scores.append(prob)
    return np.min(scores), np.mean(scores), np.max(scores), np.mean(np.array(scores) >= 0.5)

# ================================
# 7. Run full workflow on a test sample
# ================================
# Pick one negative example (predicted ≤50K)
x0 = X_test.iloc[0].copy()
y0_pred = clf_old.predict(pd.DataFrame([x0]))[0]

if y0_pred == 0:
    print("Original instance predicted ≤50K")
    x_cf = generate_counterfactual(x0, clf_old, target_class=1)
    min_s, mean_s, max_s, frac_valid = robustness_check(
        x_cf, X_train, y_train, eps_x=0.05, N=30, target_class=1
    )
    print("\nCandidate counterfactual:\n", x_cf)
    print("\nRobustness scores across perturbed models:")
    print("  min =", round(min_s, 3),
          "mean =", round(mean_s, 3),
          "max =", round(max_s, 3))
    print("  fraction valid =", round(frac_valid, 3))
else:
    print("Sample already >50K, pick another test point.")
