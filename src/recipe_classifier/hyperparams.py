# hyperparams.py
# Contains global variables for tweaking

hyperparams = [
    # KNN Hyperparameters
    # Balance the training dataset to discourage simply guessing 1 every time
    {
        "n_rows" : 25000,
        "rating_threshold" : 5,
        "min_ingredient_freq" : 50,
        "min_keyword_freq" : 60,
        "cv_count" : 10,
        "max_k" : 67,
        "max_depth" : 25,
        "balance_training" : True,
    },
    # Decision Tree Hyperparameters
    {
        "n_rows": 25000,
        "rating_threshold": 5,
        "min_ingredient_freq": 50,
        "min_keyword_freq": 60,
        "cv_count": 10,
        "max_k": 67,
        "max_depth": 25,
        "balance_training": False,
    }
]
