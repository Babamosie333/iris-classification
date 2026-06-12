# Iris Flower Classification Pipeline

A complete, self-contained Python script to explore, visualize, and classify the classic Iris dataset using multiple Machine Learning algorithms. 

The script trains three distinct classifiers, evaluates their performance, saves diagnostic plots, and makes predictions on new samples.

---

## 🚀 Features

* **Exploratory Data Analysis (EDA)**: Automatic generation of distribution stats, feature correlations, and pair plots.
* **Feature Scaling**: Integrates `StandardScaler` to optimize performance for distance-based models.
* **Multi-Model Comparison**: Evaluates and benchmarks three separate classifiers:
  * **K-Nearest Neighbors** ($K=3$)
  * **Logistic Regression** (L2 Regularized)
  * **Decision Tree** (Max Depth $= 4$)
* **Automated Asset Generation**: Saves all data visualizations and confusion matrices directly to your workspace as high-resolution `.png` files.

---

## 🛠️ Installation & Setup

Ensure you have Python 3.8+ installed. You can install all required dependencies via `pip`:

```bash
pip install scikit-learn pandas matplotlib seaborn
```

---

## 💻 How to Run

Execute the script directly from your terminal or command prompt:

```bash
python iris_classification.py
```

---

## 📊 Outputs & Generated Artifacts

When you execute the pipeline, the following 7 diagnostic plots are automatically saved to your working directory:

### 1. Data Visualizations
* `iris_pairplot.png`: Pairwise scatter plots colored by target species.
* `iris_heatmap.png`: Pearson correlation matrix across all 4 measurements.
* `iris_boxplots.png`: Box plots showing feature spreads and outliers per class.

### 2. Model Evaluation
* `cm_k-nearest.png`: Confusion matrix for the KNN classifier.
* `cm_logistic.png`: Confusion matrix for the Logistic Regression model.
* `cm_decision.png`: Confusion matrix for the Decision Tree.
* `iris_accuracy_comparison.png`: Horizontal bar chart ranking final model accuracies.
* `iris_decision_tree.png`: Graphical structure of the trained Decision Tree logic boundaries.

---

## 🔍 Model Performance Summary

The script splits the dataset using a **stratified 80/20 train/test split** to maintain class balances. Each model outputs a complete Scikit-Learn `classification_report` containing:
* **Precision**: True positives relative to total predicted positives.
* **Recall**: True positives relative to total actual positives.
* **F1-Score**: Harmonic mean of precision and recall.

---

## 🔮 Inference on New Samples

The script includes a production inference loop using the trained KNN model to classify unseen raw measurements:

| Sepal Length | Sepal Width | Petal Length | Petal Width | Expected Class |
| :--- | :--- | :--- | :--- | :--- |
| 5.1 cm | 3.5 cm | 1.4 cm | 0.2 cm | **Setosa** |
| 6.0 cm | 2.9 cm | 4.5 cm | 1.5 cm | **Versicolor** |
| 6.7 cm | 3.0 cm | 5.5 cm | 2.1 cm | **Virginica** |
