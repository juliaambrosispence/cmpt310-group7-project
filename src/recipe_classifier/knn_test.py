# knn_test.py
# Accuracy test for the KNN classifier, built to plug directly into the
# transformed_X_train / transformed_X_test / y_train / y_test 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def safe_cv(y_train, requested_cv=10):
    """
    Cross-validation needs at least `cv` examples of EVERY class in the
    training set, or sklearn throws a ValueError. This clamps cv down to
    fit whatever data you actually have, so small samples don't crash.
    """
    counts = pd.Series(y_train).value_counts().values
    smallest_class = counts.min()
    safe = max(2, min(requested_cv, int(smallest_class)))
    if safe != requested_cv:
        print(f"NOTE: reduced cv from {requested_cv} to {safe} "
              f"(smallest class in y_train only has {int(smallest_class)} examples)")
    return safe


def find_k(X_train, y_train, k_range=range(1, 101), cv=10):
    # Return whichever k had the best average accuracy.
    cv = safe_cv(y_train, cv)
    mean_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="accuracy")
        mean_scores.append(scores.mean())

    k_list = list(k_range)
    best_k = k_list[int(np.argmax(mean_scores))]
    return best_k, k_list, mean_scores


def plot_k_sweep(k_values, mean_scores, best_k, out_path="k_sweep_knn.png"):
    """
    KNN elbow plot: error rate vs. k, cross-validated on the training set only. 
    Best K point is marked with a star.
    """
    error_rates = [1 - s for s in mean_scores]
    best_error = 1 - mean_scores[k_values.index(best_k)]

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, error_rates, marker="o", linestyle="-", color="steelblue",
              label="CV error rate")
    plt.scatter([best_k], [best_error], color="red", s=140, zorder=5,
                marker="*", label=f"Elbow / best k = {best_k}")
    plt.axvline(best_k, color="red", linestyle="--", alpha=0.5)
    plt.title("KNN Elbow Plot: Error Rate vs. k")
    plt.xlabel("k (n_neighbors)")
    plt.ylabel("Cross-Validated Error Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved k-sweep elbow plot to: {out_path}")


def plot_confusion_matrix(cm, labels, out_path="confusion_matrix_knn.png"):
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("KNN Confusion Matrix (Test Set)")
    plt.tight_layout(h_pad=3.0)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to: {out_path}")


def test_knn_accuracy(
    X_train,
    X_test,
    y_train,
    y_test,
    k=None,
    max_k=101,
    cv=10,
    make_plots=True,
    model_name="recipe",
):
    """
    Runs a full accuracy check for KNN on data that's preprocessed 
    — i.e. transformed_X_train / transformed_X_test / y_train / y_test
    straight out of main.py.

    Returns a dict with the trained model, chosen k, predictions, and
    the accuracy/precision/recall/f1/confusion matrix/report.
    """

    print(f"Training set: {X_train.shape[0]} recipes")
    print(f"Testing set:  {X_test.shape[0]} recipes\n")

    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)
    print("STEP 1: Choosing k")
    print("=" * 60)
    if k is not None:
        best_k = k
        print(f"Using user-specified k = {best_k}\n")
    else:
        best_k, k_values, mean_scores = find_k(
            X_train, y_train, k_range=range(1, max_k + 1), cv=cv
        )
        print(f"Best k found via cross-validation on TRAINING data: k = {best_k}")
        print(f"(Best mean CV accuracy: {max(mean_scores):.4f})\n")
        if make_plots:
            plot_k_sweep(k_values, mean_scores, best_k, out_path=f"k_sweep_knn_{model_name}.png")

    print("=" * 60)
    print("STEP 2: Training final KNN model on training data")
    print("=" * 60)
    model = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    model.fit(X_train, y_train)
    print(f"KNN model trained with k = {best_k}.\n")

    print("=" * 60)
    print("STEP 3: Predicting on TEST features, scoring against the real answer key")
    print("=" * 60)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}\n")
    print("Classification Report:")
    print(report)

    if make_plots:
        labels = sorted(y_train.unique()) if hasattr(y_train, "unique") else sorted(set(y_train))
        plot_confusion_matrix(cm, labels, out_path=f"confusion_matrix_{model_name}.png")

    print("\nDone. The model never saw y_test during training, so this accuracy")
    print("reflects how well it generalizes to unseen recipes.")

    return {
        "model": model,
        "best_k": best_k,
        "y_pred": y_pred,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }