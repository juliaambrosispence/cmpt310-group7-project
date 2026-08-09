# mlp_test.py
# Accuracy test for the Multi-layer Perceptron classifier, built to plug directly into the
# transformed_X_train / transformed_X_test / y_train / y_test 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay, balanced_accuracy_score,
)
from sklearn.neural_network import MLPClassifier


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


def find_params(X_train, y_train, cv=5, data_balanced=False, max_iter=100):
    if data_balanced:
        acc_measure = "accuracy"
    else:
        acc_measure = "balanced_accuracy"


    cv = safe_cv(y_train, cv)
    mean_scores = []

    # Perform RandomizedSearchCV over parameters to find best network
    # These values are simply a combination of the previously-found bests
    params = {
        "hidden_layer_sizes": [
            (70,),
            (128, 64, 32)
        ],
        "early_stopping": [True],
        "activation": ['tanh'],
        "solver": ["lbfgs", "adam"],
        "alpha": [0.0001, np.float64(7.658041870432551e-05)],
        "learning_rate_init": [0.0001, np.float64(0.008353089881517721)],
        "batch_size": [32, 64],
        "learning_rate": ["invscaling"],
    }

    mlp = MLPClassifier(max_iter=max_iter, random_state=67)

    rand_search = RandomizedSearchCV(
        estimator=mlp,
        param_distributions=params,
        n_iter=32,
        cv=cv,
        n_jobs=-1,
        random_state=67,
        scoring=acc_measure,
        verbose=1
    )

    # Return the best model and its accuracy during training
    rand_search.fit(X_train, y_train)
    best_model = rand_search.best_estimator_
    scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring=acc_measure)
    mean_scores.append(np.mean(scores))

    return rand_search, mean_scores#best_estimator, estimator_list, mean_scores


def plot_confusion_matrix(cm, labels, out_path="mlp_confusion_matrix.png"):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("MLP Confusion Matrix (Test Set)")
    plt.tight_layout(h_pad=3.0)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot to: {out_path}")

def print_dict(dict):
    for key, value in dict.items():
        print(f"    {key}: {value}")
    print()

def test_mlp_accuracy(
    X_train,
    X_test,
    y_train,
    y_test,
    params=None,
    cv=5,
    max_iter=10,
    make_plots=True,
    model_name="recipe",
    data_balanced=False,
):
    """
    Runs a full accuracy check for an MLP on data that's preprocessed
    — i.e. transformed_X_train / transformed_X_test / y_train / y_test

    Returns a dict with the trained model, chosen estimator, predictions, and
    the accuracy/precision/recall/f1/confusion matrix/report.
    """

    print(f"Training set: {X_train.shape[0]} recipes")
    print(f"Testing set:  {X_test.shape[0]} recipes\n")

    print("=" * 60)
    print(f"Model: {model_name}")
    print("=" * 60)
    print("STEP 1: Performing Randomized Search CV...")
    print("=" * 60)
    if params is not None:
        print(f"Instead of searching, using user-specified parameters:")
        print_dict(params)
        best_model = MLPClassifier(max_iter=max_iter, random_state=67, **params)
    else:
        search_result, mean_scores = find_params(
            X_train, y_train, cv=cv, data_balanced=data_balanced, max_iter=max_iter,
        )
        best_model = search_result.best_estimator_
        params = search_result.best_params_
        print(f"Best params found on TRAINING data:")
        print_dict(params)
        print(f"(Best mean CV accuracy: {max(mean_scores):.4f})\n")

    print("=" * 60)
    print("STEP 2: Training final MLP on training data")
    print("=" * 60)
    best_model.fit(X_train, y_train)
    print(f"MLP Model trained with parameters:")
    print_dict(params)
    print("=" * 60)
    print("STEP 3: Predicting on TEST features, scoring against the real answer key")
    print("=" * 60)
    y_pred = best_model.predict(X_test)

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
        plot_confusion_matrix(cm, labels, out_path=f"mlp_confusion_matrix_{model_name}.png")

    print("\nDone. The model never saw y_test during training, so this accuracy")
    print("reflects how well it generalizes to unseen recipes.")

    return {
        "model": best_model,
        "params": params,
        "y_pred": y_pred,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "report": report,
    }
