
# ─────────────────────────────────────────────
#  Iris Flower Classification
#  Run: python iris_classification.py
# ─────────────────────────────────────────────

# ── 0. Install dependencies (run once in terminal) ──────────────────────────
# pip install scikit-learn pandas matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)


# ── 1. Load dataset ──────────────────────────────────────────────────────────
iris   = load_iris()
df     = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=" * 55)
print("IRIS DATASET — first 5 rows")
print("=" * 55)
print(df.head())
print("\nShape:", df.shape)
print("\nClass distribution:\n", df['species'].value_counts())
print("\nBasic stats:\n", df.describe().round(2))


# ── 2. Visualise ─────────────────────────────────────────────────────────────
# 2a. Pair plot
sns.pairplot(df, hue='species', palette={'setosa':'#534AB7',
             'versicolor':'#0F6E56', 'virginica':'#993C1D'}, plot_kws={'alpha':0.7})
plt.suptitle("Iris — pair plot of all features", y=1.02)
plt.tight_layout()
plt.savefig("iris_pairplot.png", dpi=120, bbox_inches='tight')
plt.show()

# 2b. Correlation heat-map
plt.figure(figsize=(6, 4))
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5)
plt.title("Feature correlation heat-map")
plt.tight_layout()
plt.savefig("iris_heatmap.png", dpi=120)
plt.show()

# 2c. Box-plots per feature
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flatten(), iris.feature_names):
    sns.boxplot(x='species', y=col, data=df, ax=ax,
                palette={'setosa':'#534AB7','versicolor':'#0F6E56','virginica':'#993C1D'})
    ax.set_title(col)
plt.suptitle("Feature distributions by species", fontsize=13)
plt.tight_layout()
plt.savefig("iris_boxplots.png", dpi=120)
plt.show()


# ── 3. Train / test split ────────────────────────────────────────────────────
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Optional: scale features (helps Logistic Regression & KNN)
scaler  = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)

print(f"\nTrain: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")


# ── 4. Define classifiers ────────────────────────────────────────────────────
models = {
    "K-Nearest Neighbors (k=3)": KNeighborsClassifier(n_neighbors=3),
    "Logistic Regression":       LogisticRegression(max_iter=200, random_state=42),
    "Decision Tree (max_depth=4)": DecisionTreeClassifier(max_depth=4, random_state=42),
}

results = {}   # store accuracy for comparison bar chart later


# ── 5. Train, predict, evaluate each model ──────────────────────────────────
for name, model in models.items():
    # Use scaled data for KNN and LR; raw for Decision Tree
    if "Decision Tree" in name:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(Xs_train, y_train)
        y_pred = model.predict(Xs_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("\n" + "=" * 55)
    print(f"  {name}")
    print("=" * 55)
    print(f"  Accuracy : {acc:.2%}")
    print()
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Confusion matrix plot
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=iris.target_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"Confusion matrix — {name}")
    plt.tight_layout()
    fname = "cm_" + name.split()[0].lower() + ".png"
    plt.savefig(fname, dpi=120)
    plt.show()


# ── 6. Accuracy comparison bar chart ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(list(results.keys()), list(results.values()),
               color=['#534AB7', '#0F6E56', '#993C1D'])
ax.set_xlim(0.85, 1.02)
ax.set_xlabel("Accuracy")
ax.set_title("Model accuracy comparison")
for bar, val in zip(bars, results.values()):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.2%}", va='center', fontsize=11)
plt.tight_layout()
plt.savefig("iris_accuracy_comparison.png", dpi=120)
plt.show()


# ── 7. Decision tree visualisation ──────────────────────────────────────────
dt_model = models["Decision Tree (max_depth=4)"]
fig, ax = plt.subplots(figsize=(14, 6))
plot_tree(dt_model, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True,
          rounded=True, fontsize=10, ax=ax)
plt.title("Decision tree — learned rules")
plt.tight_layout()
plt.savefig("iris_decision_tree.png", dpi=120)
plt.show()


# ── 8. Predict a new sample ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PREDICT NEW SAMPLES")
print("=" * 55)

new_samples = [
    [5.1, 3.5, 1.4, 0.2],   # likely Setosa
    [6.0, 2.9, 4.5, 1.5],   # likely Versicolor
    [6.7, 3.0, 5.5, 2.1],   # likely Virginica
]

# Use the best model (KNN)
knn = models["K-Nearest Neighbors (k=3)"]
new_scaled = scaler.transform(new_samples)

for sample, scaled in zip(new_samples, new_scaled):
    pred = knn.predict([scaled])[0]
    prob = knn.predict_proba([scaled])[0]
    print(f"\n  Input  : sepal={sample[:2]}, petal={sample[2:]}")
    print(f"  Predicted : {iris.target_names[pred].upper()}")
    print(f"  Probabilities: " +
          ", ".join(f"{n}={p:.0%}" for n, p in zip(iris.target_names, prob)))

print("\nDone! All plots saved as PNG files in the same folder.")
