# rforest_test.py
# Accuracy test for the Random Forest classifier, built to plug directly into the
# transformed_X_train / transformed_X_test / y_train / y_test 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay, balanced_accuracy_score,
)

# Best params for rforest
max_depth = None
max_features = "sqrt"
min_samples_split = 2
min_samples_leaf = 2


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


def find_estimator(X_train, y_train, estimator_range=[1, 100, 200], cv=5, data_balanced=False):
    if data_balanced:
        acc_measure = "accuracy"
        class_weight = None
    else:
        acc_measure = "balanced_accuracy"
        class_weight = "balanced"

    # return whichever max_estimator had the best average accuracy.
    cv = safe_cv(y_train, cv)
    mean_scores = []
    for estimator in estimator_range:
        rf = RandomForestClassifier(n_estimators=estimator, random_state=67,
                                    max_depth=max_depth,
                                    max_features=max_features,
                                    class_weight=class_weight,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    n_jobs=-1)
        scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring=acc_measure)
        mean_scores.append(scores.mean())

    estimator_list = list(estimator_range)
    best_estimator = estimator_list[int(np.argmax(mean_scores))]
    return best_estimator, estimator_list, mean_scores


def plot_estimator_sweep(estimator_values, mean_scores, best_estimator, out_path="estimator_sweep.png"):
    """
    Random Forest elbow plot: error rate vs. estimator, cross-validated on the training set only.
    Best estimator point is marked with a star.
    """
    error_rates = [1 - s for s in mean_scores]
    best_error = 1 - mean_scores[estimator_values.index(best_estimator)]

    plt.figure(figsize=(8, 5))
    plt.plot(estimator_values, error_rates, marker="o", linestyle="-", color="forestgreen",
              label="CV error rate")
    plt.scatter([best_estimator], [best_error], color="red", s=140, zorder=5,
                marker="*", label=f"Best estimator = {best_estimator}")
    plt.axvline(best_estimator, color="red", linestyle="--", alpha=0.5)
    plt.title("Random Forest: Error Rate vs. Max Estimators")
    plt.xlabel("Max Estimators")
    plt.ylabel("Cross-Validated Error Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved estimator sweep plot to: {out_path}")


def plot_confusion_matrix(cm, labels, out_path="dt_confusion_matrix.png"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Random Forest Confusion Matrix (Test Set)")
    plt.tight_layout(h_pad=3.0)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to: {out_path}")


def test_rforest_accuracy(
    X_train,
    X_test,
    y_train,
    y_test,
    estimator=None,
    max_estimator_test=30,
    cv=5,
    make_plots=True,
    model_name="recipe",
    data_balanced=False,
):
    """
    Runs a full accuracy check for Random Forest on data that's preprocessed
    — i.e. transformed_X_train / transformed_X_test / y_train / y_test

    Returns a dict with the trained model, chosen estimator, predictions, and
    the accuracy/precision/recall/f1/confusion matrix/report.
    """

    if data_balanced:
        class_weight = None
    else:
        class_weight = "balanced"

    print(f"Training set: {X_train.shape[0]} recipes")
    print(f"Testing set:  {X_test.shape[0]} recipes\n")

    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)
    print("STEP 1: Choosing max_estimator")
    print("=" * 60)
    if estimator is not None:
        best_estimator = estimator
        print(f"Using user-specified estimator = {best_estimator}\n")
    else:
        best_estimator, estimator_values, mean_scores = find_estimator(
            X_train, y_train, estimator_range=[1, *range(250, max_estimator_test + 1, 250)], cv=cv, data_balanced=data_balanced
        )
        print(f"Best estimator found via cross-validation on TRAINING data: estimator = {best_estimator}")
        print(f"(Best mean CV accuracy: {max(mean_scores):.4f})\n")
        if make_plots:
            plot_estimator_sweep(estimator_values, mean_scores, best_estimator, out_path=f"rforest_sweep_{model_name}.png")

    print("=" * 60)
    print("STEP 2: Training final Random Forest model on training data")
    print("=" * 60)
    model = RandomForestClassifier(n_estimators=best_estimator, random_state=67,
                                   max_depth=None,
                                   max_features=max_features,
                                   class_weight=class_weight,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   n_jobs=-1
                                   )
    model.fit(X_train, y_train)
    print(f"Random Forest model trained with max_estimator = {best_estimator}.\n")

    print("=" * 60)
    print("STEP 3: Predicting on TEST features, scoring against the real answer key")
    print("=" * 60)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"Accuracy:       {acc:.4f}")
    print(f"Bal. Accuracy:  {balanced_acc:.4f}")
    print(f"Precision:      {prec:.4f}")
    print(f"Recall:         {rec:.4f}")
    print(f"F1 Score:       {f1:.4f}\n")
    print("Classification Report:")
    print(report)

    if make_plots:
        labels = sorted(y_train.unique()) if hasattr(y_train, "unique") else sorted(set(y_train))
        plot_confusion_matrix(cm, labels, out_path=f"rforest_confusion_matrix_{model_name}.png")

    print("\nDone. The model never saw y_test during training, so this accuracy")
    print("reflects how well it generalizes to unseen recipes.")

    return {
        "model": model,
        "best_estimator": best_estimator,
        "y_pred": y_pred,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }
