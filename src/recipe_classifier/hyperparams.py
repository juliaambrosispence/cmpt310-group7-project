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
        "max_sweep" : 67,
        "balance_training_recipe" : True,
        "balance_training_cuisine" : False,
        "param1" : None,
        "param2" : None
    },
    # Decision Tree Hyperparameters
    {
        "n_rows": 25000,
        "rating_threshold": 5,
        "min_ingredient_freq": 50,
        "min_keyword_freq": 60,
        "cv_count": 10,
        "max_sweep": 25,
        "balance_training_recipe": False,
        "balance_training_cuisine": False,
        "param1" : None,
        "param2" : None
    },
    # Random Forest Hyperparameters
    {
        "n_rows": 25000,
        "rating_threshold": 5,
        "min_ingredient_freq": 20,
        "min_keyword_freq": 60,
        "cv_count": 10,
        "max_sweep": 1000,
        "balance_training_recipe": False,
        "balance_training_cuisine": False,
        "param1": None,
        "param2": None
    }
]
