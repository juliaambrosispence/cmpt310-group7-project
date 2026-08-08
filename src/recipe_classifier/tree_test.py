# tree_test.py
# Accuracy test for the Decision Tree classifier, built to plug directly into the
# transformed_X_train / transformed_X_test / y_train / y_test 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def safe_cv(y_train, requested_cv=5):
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


def find_depth(X_train, y_train, depth_range=range(1, 31), cv=5):
    # return whichever max_depth had the best average accuracy.
    cv = safe_cv(y_train, cv)
    mean_scores = []
    for depth in depth_range:
        # class_weight='balanced' for the good recipe bias
        dt = DecisionTreeClassifier(max_depth=depth, class_weight='balanced', random_state=67)
        scores = cross_val_score(dt, X_train, y_train, cv=cv, scoring="balanced_accuracy")
        mean_scores.append(scores.mean())

    depth_list = list(depth_range)
    best_depth = depth_list[int(np.argmax(mean_scores))]
    return best_depth, depth_list, mean_scores


def plot_depth_sweep(depth_values, mean_scores, best_depth, out_path="depth_sweep.png"):
    """
    Decision Tree elbow plot: error rate vs. depth, cross-validated on the training set only. 
    Best depth point is marked with a star.
    """
    error_rates = [1 - s for s in mean_scores]
    best_error = 1 - mean_scores[depth_values.index(best_depth)]

    plt.figure(figsize=(8, 5))
    plt.plot(depth_values, error_rates, marker="o", linestyle="-", color="forestgreen",
              label="CV error rate")
    plt.scatter([best_depth], [best_error], color="red", s=140, zorder=5,
                marker="*", label=f"Best Depth = {best_depth}")
    plt.axvline(best_depth, color="red", linestyle="--", alpha=0.5)
    plt.title("Decision Tree: Error Rate vs. Max Depth")
    plt.xlabel("Max Depth (Number of splits)")
    plt.ylabel("Cross-Validated Error Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved depth sweep plot to: {out_path}")


def plot_confusion_matrix(cm, labels, out_path="dt_confusion_matrix.png"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Decision Tree Confusion Matrix (Test Set)")
    plt.tight_layout(h_pad=3.0)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to: {out_path}")


def test_tree_accuracy(
    X_train,
    X_test,
    y_train,
    y_test,
    depth=None,
    max_depth_test=30,
    cv=5,
    make_plots=True,
    model_name="recipe",
):
    """
    Runs a full accuracy check for Decision Tree on data that's preprocessed 
    — i.e. transformed_X_train / transformed_X_test / y_train / y_test
    straight out of tree.py.

    Returns a dict with the trained model, chosen depth, predictions, and
    the accuracy/precision/recall/f1/confusion matrix/report.
    """

    print(f"Training set: {X_train.shape[0]} recipes")
    print(f"Testing set:  {X_test.shape[0]} recipes\n")

    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)
    print("STEP 1: Choosing max_depth")
    print("=" * 60)
    if depth is not None:
        best_depth = depth
        print(f"Using user-specified depth = {best_depth}\n")
    else:
        best_depth, depth_values, mean_scores = find_depth(
            X_train, y_train, depth_range=range(1, max_depth_test + 1), cv=cv
        )
        print(f"Best depth found via cross-validation on TRAINING data: depth = {best_depth}")
        print(f"(Best mean CV accuracy: {max(mean_scores):.4f})\n")
        if make_plots:
            plot_depth_sweep(depth_values, mean_scores, best_depth, out_path=f"depth_sweep_{model_name}.png")

    print("=" * 60)
    print("STEP 2: Training final Decision Tree model on training data")
    print("=" * 60)
    model = DecisionTreeClassifier(max_depth=best_depth, class_weight='balanced', random_state=67)
    model.fit(X_train, y_train)
    print(f"Decision Tree model trained with max_depth = {best_depth}.\n")

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
        plot_confusion_matrix(cm, labels, out_path=f"dt_confusion_matrix_{model_name}.png")

    print("\nDone. The model never saw y_test during training, so this accuracy")
    print("reflects how well it generalizes to unseen recipes.")

    return {
        "model": model,
        "best_depth": best_depth,
        "y_pred": y_pred,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }
