# main.py
# Main project file
# Take model selection from user, train and take input recipes to make predictions

import kagglehub
import pandas as pd
import sys
import numpy as np
import pyarrow.parquet as pq
from joblib.externals.loky.process_executor import MAX_DEPTH
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path
import matplotlib.pyplot as plt

from webscraper import parse_url
from utils import *
from enum import Enum
from knn_test import test_knn_accuracy
from tree_test import test_tree_accuracy
from rforest_test import test_rforest_accuracy
from hyperparams import *

USER_MODEL = None

# Dictionary that contains preprocessing method for each column of recipes dataset
# Specify which columns we:
# drop - Delete from dataset
# standard - Subtract mean and divide by std. deviation
# time - Convert to number of minutes and then treat as standard
# one-hot - Treat as a categorical feature, where an example has a 1 if it has that feature, 0 o.w.
# multicat - Use MultiLabelBinarizer to take an array of categories and mark if the example has that feature
feature_settings = {
 'RecipeId' : "drop",
 'Name' : "drop",
 'AuthorId' : "drop",
 'AuthorName' : "drop",
 'CookTime' : "time",
 'PrepTime' : "time",
 'TotalTime' : "drop",
 'DatePublished' : "drop",
 'Description' : "drop",
 'Images' : "drop",
 'RecipeCategory' : "drop",
 'Keywords' : "one-hot",
 'RecipeIngredientQuantities' : "drop",
 'RecipeIngredientParts' : "multicat",
 'AggregatedRating' : "drop",
 'ReviewCount' : "drop",
 'Calories' : "standard",
 'FatContent' : "standard",
 'SaturatedFatContent' : "standard",
 'CholesterolContent' : "standard",
 'SodiumContent' : "standard",
 'CarbohydrateContent' : "standard",
 'FiberContent' : "standard",
 'SugarContent' : "standard",
 'ProteinContent' : "standard",
 'RecipeServings' : "standard",
 'RecipeYield' : "drop",
 'RecipeInstructions' : "drop",
}

# AI Models to choose
class Models(Enum):
    KNN = 1
    DTREE = 2
    RFOREST = 3


# Define a modified MultiLabelBinarizer for use in ColumnTransformer
class ModifiedMultiLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self):
      self.mlb = MultiLabelBinarizer()

    def fit(self, X, y=None):
      # Extract first column as an array of iterable objects
      X_series = pd.Series(X.iloc[:, 0])
      self.mlb.fit(X_series)
      return self

    def transform(self, X):
      X_series = pd.Series(X.iloc[:, 0])
      return self.mlb.transform(X_series)

    def get_feature_names_out(self, input_features=None):
         prefix = input_features[0]
         return [f"{prefix}_{c}" for c in self.mlb.classes_]



# Download the dataset, if it already exists in cache then this returns quickly
dataset_dir = Path(kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews"))
print(f"Dataset located at {dataset_dir}")

# Perform preprocessing on the dataset based on hyperparameters
def data_preprocess(n_rows, rating_threshold, min_ingredient_freq, min_keyword_freq,
                    cv_count, max_sweep, balance_training_recipe, balance_training_cuisine,
                    param1, param2):
    # Parquet files need a different approach to only read a certain number of entries
    parquet_recipes = pd.read_parquet(dataset_dir / "recipes.parquet")#pq.ParquetFile(dataset_dir / "recipes.parquet")
    #filtering out recipes that don't have a rating
    parquet_recipes = parquet_recipes.dropna(subset=['AggregatedRating']).reset_index(drop=True)
    # filter out recipes with less than min review count
    #print(len(parquet_recipes))
    #parquet_recipes = parquet_recipes[parquet_recipes['ReviewCount'] >= 10].reset_index(drop=True)
    #print(len(parquet_recipes))

    #recipes_df = next(parquet_recipes.iter_batches(batch_size=n_rows)).to_pandas()
    recipes_df = parquet_recipes.sample(n=n_rows, random_state=67) #randomly picking n_rows amount from dataset

    #recipes_df['ReviewCount'].value_counts().sort_index().plot(kind='hist')
    # plt.hist(recipes_df['ReviewCount'].value_counts().sort_index(), bins=5000)
    # plt.xlim(0, 100)
    # plt.xlabel('Review Count')
    # plt.ylabel('Number of Recipes')
    # plt.title("Review Count Distribution")
    # plt.show()

    # Next we should do processing on the data

    # Split up all the columns depending on feature settings specified at the top of file.
    drop_columns = []
    standard_columns = []
    time_columns = []
    categorical_columns = []
    multicat_columns = []

    for column, method in feature_settings.items():
     if method == "drop":
      drop_columns.append(column)
     elif method == "time":
      time_columns.append(column)
      standard_columns.append(column)
     elif method == "standard":
      standard_columns.append(column)
     elif method == "one-hot":
      categorical_columns.append(column)
     elif method == "multicat":
      multicat_columns.append(column)

    # Convert ISO 8601 durations in time_columns to minutes
    for column in time_columns:
     recipes_df[column] = pd.to_timedelta(
     recipes_df[column].str.replace("PT", "").str.replace("H", " hours ").str.replace("M", " minutes").str.replace("S", " seconds")).dt.total_seconds() / 60


    # Keywords work like ingredients
    keyword_ings = recipes_df['Keywords'].tolist()

    # Remove any None entries from Keywords column
    for i in range(0, len(keyword_ings)):
        if keyword_ings[i][0] is None:
            keyword_ings[i] = []

    # Only keep keywords that are relevant to cuisine type
    keep_keywords = [
        'Asian',
        'Mexican',  'Canadian',
        'Southwestern U.S.',
        'Australian', 'Indian', 'African', 'Chinese', 'Southwest Asia (middle East)',
        'Greek', 'Caribbean', 'South American', 'Cajun', 'German',
        'Scandinavian', 'Creole', 'Spanish', 'Thai', 'Moroccan', 'Japanese', 'Scottish', 'Portuguese', 'New Zealand', 'Hawaiian',  'Swiss', 'Korean', 'Lebanese', 'South African', 'Hungarian', 'Russian', 'Vietnamese', 'Welsh', 'Swedish', 'Brazilian', 'Austrian',
       'Turkish', 'Indonesian', 'Norwegian', 'Peruvian', 'Native American', 'Polynesian', 'Dutch', 'Polish', 'Danish', 'Belgian', 'Szechuan', 'Pennsylvania Dutch', 'Czech', 'Egyptian', 'Cuban', 'Finnish', 'Filipino', 'Malaysian', 'Venezuelan',
        'Guatemalan', 'Nigerian','Colombian', 'Palestinian', 'Puerto Rican', 'Ethiopian', 'Iraqi', 'Cantonese', 'Cambodian', 'Hunan', 'Chilean', 'Pakistani', 'Icelandic', 'Costa Rican', 'Nepalese', 'Sudanese', 'Honduran', 'Ecuadorean'

    ]

    new_keywords = ["None"] * len(keyword_ings)

    # overwrite the category as it goes through keywords list
    for i in range(0, len(keyword_ings)):
        keywords = keyword_ings[i]
        #keep_words = []
        for keyword in keywords:
            if keyword in keep_keywords:
                #keep_words.append(keyword)

                # Only assign a keyword once, the first one that appears
                if new_keywords[i] == "None":
                    #print("Replacing", new_keywords[i], "with", keyword)
                    new_keywords[i] = keyword


    # Remove any keywords that do not appear a min number of times in the dataset
    # Count the occurrence of each keyword
    keyword_dict = {}
    for keyword in new_keywords:
        if keyword not in keyword_dict:
            keyword_dict[keyword] = 1
        else:
            keyword_dict[keyword] += 1

    # Add any keywords with insufficient counts to a filter list
    filtered_keywords = []
    for keyword in keyword_dict:
        if keyword_dict[keyword] < min_keyword_freq:
            filtered_keywords.append(keyword)

    # Set the entry in the Keyword column to None so that they get filtered
    for i in range(0, len(new_keywords)):
        if new_keywords[i] in filtered_keywords:
            new_keywords[i] = "None"

    # Put keywords back into dataset
    recipes_df['Keywords'] = new_keywords

    #ok sorry I changed this cause I realized could actually use none and filter that out. (╥﹏╥)
    recipes_df = recipes_df[recipes_df['Keywords'] != "None"].reset_index(drop=True) #if keywords is not 'none' then true, reset rows back to 0


    all_ings = recipes_df['RecipeIngredientParts'].tolist() #make it into a list for the for loops.

    # Prefixes that we want to remove from ingredient names
    # Some ingredients are written like '1/3 cup of fresh mint'
    prefix_remove = [ "fresh", "of", "ground", "dried", "crushed", "lean", "frozen", 'medium', 'small', 'large' ]

    #I swapped out the base_ings to check the multi word and it will now group the ingredients if it sees a keyword.
    base_ing = {
        "tortilla" : ["flour tortilla", "corn tortilla", "tortilla"],
        "peanut butter"  : ["peanut butter"],
        "soy sauce"      : ["soy sauce"],
        "olive oil"      : ["olive oil"],
        "vegetable oil"  : ["vegetable oil", "canola oil"],
        "chicken broth"  : ["chicken broth", "chicken stock"],
        "cream cheese"   : ["cream cheese"],
        "sour cream"     : ["sour cream"],
        "baking powder"  : ["baking powder"],
        "baking soda"    : ["baking soda"],
        "vanilla"        : ["vanilla extract", "vanilla"],
        "flour"          : ["all-purpose flour", "whole wheat flour", "bread flour", "flour"],
        "sugar"          : ["brown sugar", "granulated sugar", "powdered sugar", "sugar"],
        "chicken"        : ["boneless chicken", "chicken breast", "chicken thigh", "ground chicken", "chicken"],
        "salt"           : ["sea salt", "kosher salt", "salt"],
        "milk"           : ["skim milk", "whole milk", "milk"],
        "butter"         : ["unsalted butter", "salted butter", "butter"],
        "lemon"          : ["lemon juice", "lemon zest", "lemon"],
        "lime"           : ["lime juice", "lime zest", "lime"],
        "garlic"         : ["minced garlic", "garlic cloves", "garlic"],
        "onion"          : ["green onion", "red onion", "yellow onion", "spring onion", "onion"],
        "yogurt"         : ["plain yogurt", "greek yogurt", "vanilla yogurt", "yogurt"],
        "rice vinegar"   : ["rice vinegar"],
        "rice"           : ["basmati rice", "jasmine rice", "brown rice", "long-grain rice", "rice"],
        "egg"            : ["large eggs", "whole eggs", "eggs", "egg"],
        "tomato"         : ["cherry tomatoes", "diced tomatoes", "tomato paste", "tomato sauce", "tomatoes", "tomato"],
        "oil"            : ["oil"],
        "cumin"          : ["cumin", "ground cumin"],
        "ginger"         : ["ginger", "fresh ginger", "ground ginger"],
        "cinnamon"       : ["cinnamon", "ground cinnamon"],
        "black pepper"   : ["black pepper", "ground pepper"],
        "water"          : ["water", "warm water"],
    }

    for i in range(0, len(all_ings)):#until end of list,
        # Remove duplicates from recipe ingredients
        current_recipe = list(dict.fromkeys(all_ings[i]));
        cleaned_recipe = []; #new combined ingredient list

        for j in range(0, len(current_recipe)): #loop for every ingredient
            item = str(current_recipe[j]).lower();#standardize lower

            # Remove any prefixes first
            # May need to remove multiple prefixes
            split_item = item.split(" ")
            words = len(split_item)
            while words > 0:
                if split_item[0] in prefix_remove:
                    split_item.pop(0)
                    words -= 1
                else:
                    break

            item = " ".join(split_item)
            found_match = False;#needed to break out of both loops

            for main_ing in base_ing:#loop through dictionary keys

                for alias in base_ing[main_ing]: #loop through the messy variations. checks the alias exist from base word
                    if alias in item and item != main_ing: #can find alias in ingredients, and ingredient needs to be different from existing to enter loop
                        item = main_ing; #overwrite it
                        found_match = True; #now done
                        break; #found match and gonna stop searching aliases!

                if found_match:
                    break; # found match, stop searching main dictionary!

            cleaned_recipe.append(item);#add into the combined list

        all_ings[i] = cleaned_recipe; #overwrite list with clean version

    ing_counts = {};#check how many time actual ingredient shows up and adds to a count.
    for i in range(0, len(all_ings)):
        recipe = all_ings[i];

        for j in range(0, len(recipe)):
            ing = recipe[j];

            #check if key exists in list
            if ing in ing_counts:
                ing_counts[ing] = ing_counts[ing] + 1;
            else:
                ing_counts[ing] = 1; #+0 to count basicall


    #going to check the count and go back delete the ones that don't meet our minimum
    for i in range(len(all_ings)):
        current_recipe = all_ings[i];
        popular_only = [] #ingredients that meet the minimum freq (2)

        for ing in current_recipe:
            if ing_counts[ing] >= min_ingredient_freq: #if it doesnt meet minimum freq (2)
                popular_only.append(ing);

        all_ings[i] = popular_only

    recipes_df['RecipeIngredientParts'] = all_ings #put list back into the main pandas table

    # recipes_df['AggregatedRating'].value_counts().sort_index().plot(kind='bar')
    # plt.xlabel('Star Rating')
    # plt.ylabel('Number of Recipes')
    # plt.title("Recipe Rating Distribution")
    # plt.show()

    X = recipes_df.drop(columns=drop_columns)

    # Our labels are 1 - good recipe if rating is above threshold, otherwise 0
    y = (recipes_df['AggregatedRating'].astype(float) >= rating_threshold).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=67, stratify=y)
    X_train_original, X_test_original, y_train_original, y_test_original = X_train, X_test, y_train, y_test

    if balance_training_recipe:
        # Balancing training data only!

        # good_indices_Y = y_train[y_train == 1].index
        # bad_indices_Y = y_train[y_train == 0].index
        y_train, X_train = balance_data([
            y_train[y_train == 1],
            y_train[y_train == 0],
        ], X_train)


    print("Training set label counts:")
    print(y_train.value_counts())
    print("\nTest set label counts:")
    print(y_test.value_counts())


    # The type of encoding we do will make a new feature out of every unique string we see in the category and ingredient
    # lists, if we have an entry like "blueberries" vs "frozen blueberries" should this count as one feature?
    # Or "all-purpose flour" should obviously be the same ingredient as "flour" we should do some string processing
    # to strip prefix words like that, and perhaps eliminate certain ingredients?
    # We could also limit the ingredient categories we make into features into ones that only show up several times

    # Preprocess data using a ColumnTransformer to fit everything into one ndarray of data
    preprocessor = ColumnTransformer(
     transformers=[
      ("numerical", Pipeline([("imputer",SimpleImputer(strategy="mean")),("scaler",StandardScaler())]), standard_columns),
      ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
      ("ingredients", ModifiedMultiLabelBinarizer(), ["RecipeIngredientParts"]),
      #("keywords", ModifiedMultiLabelBinarizer(), ["Keywords"]),
     ]
    )

    # Apply fit of data based on the methods we specified to each set of columns, then apply transformations
    transformed_X_train = preprocessor.fit_transform(X_train)
    transformed_X_test = preprocessor.transform(X_test)


    feature_names = preprocessor.get_feature_names_out()
    sample_x = transformed_X_train[0]
    sample_y = y_train.iloc[0]

    # Look at features of first row ########################
    print("\nchecking first row:")
    print("x features:")
    for i in range(len(feature_names)):

        if sample_x[i] != 0: #only print nonzero stuff example
            print(f"  {feature_names[i]}: {sample_x[i]:.2f}")

    print("\ny label (answer):")
    if sample_y == 1:
        print(f"  {sample_y} (good recipe)")
    else:
        print(f"  {sample_y} (bad recipe)")
    print("\n")



    # Cuisine classifier, knn for cuisine prediction #############
    target_cuisine = recipes_df['Keywords']; #keywords column of cuisine types

    #split random
    y_train_cuisine = target_cuisine.loc[y_train_original.index]; #training data. iloc index position rows. cusine label for every recipe.
    #print(y_train_cuisine)
    y_test_cuisine = target_cuisine.loc[y_test_original.index]; #testing data [maxican, german, chinese]
    #print(y_test_cuisine)

    X_train_c = X_train_original.copy()
    X_test_c = X_test_original.copy()

    if balance_training_cuisine:
        # Balance cuisine data
        y_train_cuisine, X_train_c = balance_data([
            y_categories for _, y_categories in y_train_cuisine.groupby(y_train_cuisine)
        ], X_train_c)

    # Apply fit of data based on the methods we specified to each set of columns, then apply transformations
    #turns the ingredients to binary numbers and takes standardized mean values
    cuisine_prep = ColumnTransformer(
        transformers=[
            ("numerical", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), standard_columns),
            ("ingredients", ModifiedMultiLabelBinarizer(), ["RecipeIngredientParts"]),
        ]
    );
    #create row of all ingredient names every recipe
    x_train_c_transformed = cuisine_prep.fit_transform(X_train_c);
    x_test_c_transformed = cuisine_prep.transform(X_test_c);



    print("Training set label counts:")
    print(y_train_cuisine.value_counts())
    print("\nTest set label counts:")
    print(y_test_cuisine.value_counts())

    return (
            transformed_X_train, transformed_X_test, y_train, y_test, x_train_c_transformed, x_test_c_transformed,
            y_train_cuisine, y_test_cuisine, preprocessor, cuisine_prep,
            ing_counts, prefix_remove, base_ing
            )

def balance_data(data_list_y, data_x):
    min_length_index = 0
    # Find the data-set with the smallest length
    for i in range(0, len(data_list_y)):
        if len(data_list_y[i]) < len(data_list_y[min_length_index]):
            min_length_index = i
    # Remove entries randomly from each data-set except the smallest so that they are balanced
    min_length = len(data_list_y[min_length_index])
    for i in range(0, len(data_list_y)):
        if i != min_length_index:
            drop_indices = data_list_y[i].sample(n=len(data_list_y[i])-min_length, random_state=67).index
            data_list_y[i] = data_list_y[i].drop(drop_indices)
            # Drop the associated rows in the X data as well
            data_x = data_x.drop(drop_indices)


    y_train = pd.concat(data_list_y, ignore_index=False)
    X_train = data_x.loc[y_train.index]

    # Shuffle data
    shuffled_indices = y_train.sample(frac=1, random_state=67).index

    y_train = y_train.loc[shuffled_indices]
    X_train = X_train.loc[shuffled_indices]

    return y_train, X_train

# Call the test functions to sweep for the best parameter depending on the chosen model
def train_models(transformed_X_train, transformed_X_test, y_train, y_test, x_train_c, x_test_c, y_train_cuisine, y_test_cuisine,
                 hyperparameters):
    # Variables for the trained model's results
    recipe_results = None
    cuisine_results = None
    cv_count = hyperparameters["cv_count"]
    balance_training_recipe = hyperparameters["balance_training_recipe"]
    balance_training_cuisine = hyperparameters["balance_training_cuisine"]
    param1 = hyperparameters["param1"]
    param2 = hyperparameters["param2"]
    # Classifier Model Choice
    if USER_MODEL == Models.KNN:
        recipe_results = test_knn_accuracy(
             transformed_X_train, transformed_X_test, y_train, y_test,
            max_k=hyperparameters["max_sweep"], cv=cv_count, data_balanced=balance_training_recipe, k=param1
        )
        cuisine_results = test_knn_accuracy(
             x_train_c, x_test_c, y_train_cuisine, y_test_cuisine,
            max_k=hyperparameters["max_sweep"], cv=cv_count, model_name="cuisine", data_balanced=balance_training_cuisine, k=param2
        )
    elif USER_MODEL == Models.DTREE:
        recipe_results = test_tree_accuracy(
             transformed_X_train, transformed_X_test, y_train, y_test,
            max_depth_test=hyperparameters["max_sweep"], cv=cv_count, data_balanced=balance_training_recipe, depth=param1
        )
        cuisine_results = test_tree_accuracy(
             x_train_c, x_test_c, y_train_cuisine, y_test_cuisine,
            max_depth_test=hyperparameters["max_sweep"], cv=cv_count, model_name="cuisine", data_balanced=balance_training_cuisine, depth=param2
        )
    elif USER_MODEL == Models.RFOREST:
        recipe_results = test_rforest_accuracy(
            transformed_X_train, transformed_X_test, y_train, y_test,
            max_estimator_test=hyperparameters["max_sweep"], cv=cv_count, data_balanced=balance_training_recipe, estimator=param1
        )
        cuisine_results = test_rforest_accuracy(
            x_train_c, x_test_c, y_train_cuisine, y_test_cuisine,
            max_estimator_test=hyperparameters["max_sweep"], cv=cv_count, model_name="cuisine", data_balanced=balance_training_cuisine,
            estimator=param2
        )

    return recipe_results, cuisine_results


def clean_ingredient(item, prefix_remove, base_ing):
    #lowercases and trim white spaces
    item = str(item).lower().strip()
    split_item = item.split(" ")
    while len(split_item) > 0 and split_item[0] in prefix_remove:
        split_item.pop(0)
    item = " ".join(split_item)
    #change to accepted term from base_ing
    for main_ing in base_ing:
        for alias in base_ing[main_ing]:
            if alias in item and item != main_ing:
                return main_ing
    return item

def clean_new_ingredients(raw_ingredients, ing_counts, min_ingredient_freq, prefix_remove, base_ing):
    #remove duplicates
    cleaned = [clean_ingredient(ing, prefix_remove, base_ing) for ing in list(dict.fromkeys(raw_ingredients)) if ing.strip() != ""]
    #filter again so only ingredients the model is aware of from ing_counts
    known = [ing for ing in cleaned if ing_counts.get(ing, 0) >= min_ingredient_freq]
    dropped = [ing for ing in cleaned if ing not in known]

    return known

#pack the individual raw values into single Dataframe
def build_recipe_row(cook_time, prep_time, keyword, ingredients, calories, fat, sat_fat, cholesterol, sodium, carbs, fiber, sugar, protein, servings):
    return pd.DataFrame([{
        "CookTime": cook_time,
        "PrepTime": prep_time,
        "Keywords": keyword,
        "RecipeIngredientParts": ingredients,
        "Calories": calories,
        "FatContent": fat,
        "SaturatedFatContent": sat_fat,
        "CholesterolContent": cholesterol,
        "SodiumContent": sodium,
        "CarbohydrateContent": carbs,
        "FiberContent": fiber,
        "SugarContent": sugar,
        "ProteinContent": protein,
        "RecipeServings": servings,
    }])

def predict_new_recipe(recipe_model, cuisine_model, preprocessor, cuisine_prep,
                       ing_counts, prefix_remove, base_ing, min_ingredient_freq, data=None):
    # If data supplied from URL scraper, then skip asking
    if data:
        data["ingredients"] = clean_new_ingredients(data["ingredients"], ing_counts, min_ingredient_freq, prefix_remove, base_ing)
        #print(data)
        new_row = build_recipe_row(**data)
    else:
        def ask_float(prompt, default=0.0): #default is 0.0
            raw = input(f"{prompt} [{default}]: ").strip()
            if raw == "": #return default if null input
                return default
            try:
                return float(raw)
            except ValueError: #error handling
                print("Error, using default.")
                return default
        print("\n")
        print("Enter a new recipe to classify:")
        print("-"*40)
        #asking for raw fields the model needs
        cook_time = ask_float("Cook time (minutes)")
        prep_time = ask_float("Prep time (minutes)")
        calories = ask_float("Calories")
        fat = ask_float("Fat (g)")
        sat_fat = ask_float("Saturated fat (g)")
        cholesterol = ask_float("Cholesterol (mg)")
        sodium = ask_float("Sodium (mg)")
        carbs = ask_float("Carbohydrates (g)")
        fiber = ask_float("Fiber (g)")
        sugar = ask_float("Sugar (g)")
        protein = ask_float("Protein (g)")
        servings = ask_float("Servings", default=2.0) #default=2 since zero servings doens't make sense
        #asking for cuisine identifier
        keyword_raw = input("Cuisine identifier: ").strip()
        keyword = keyword_raw if keyword_raw else "None"
        #asking for ingredients
        ingredients_raw = input("Ingredients, comma-separated: ").strip()
        ingredients_list = [i.strip() for i in ingredients_raw.split(",") if i.strip()]
        cleaned_ingredients = clean_new_ingredients(ingredients_list, ing_counts, min_ingredient_freq, prefix_remove, base_ing)
        #calling function to put all into desired position
        new_row = build_recipe_row(
            cook_time, prep_time, keyword, cleaned_ingredients, calories, fat, sat_fat, cholesterol, sodium, carbs, fiber, sugar, protein, servings,
        )
    print(f"\nIngredients used by model: {new_row['RecipeIngredientParts'].iloc[0]}")
    print(f"  Calories: {new_row['Calories'].iloc[0]}")
    print(f"  Fat: {new_row['FatContent'].iloc[0]}g")
    print(f"  Saturated Fat: {new_row['SaturatedFatContent'].iloc[0]}g")
    print(f"  Cholesterol: {new_row['CholesterolContent'].iloc[0]}mg")
    print(f"  Sodium: {new_row['SodiumContent'].iloc[0]}mg")
    print(f"  Carbohydrates: {new_row['CarbohydrateContent'].iloc[0]}g")
    print(f"  Fiber: {new_row['FiberContent'].iloc[0]}g")
    print(f"  Sugar: {new_row['SugarContent'].iloc[0]}g")
    print(f"  Protein: {new_row['ProteinContent'].iloc[0]}g")
    print(f"  Servings: {new_row['RecipeServings'].iloc[0]}")
    print(f"  Cook Time: {new_row['CookTime'].iloc[0]} min")
    print(f"  Prep Time: {new_row['PrepTime'].iloc[0]} min")
    #quality prediction reusing the fitted preprocessor + knn
    transformed_new = preprocessor.transform(new_row) #runs the row through ColumnTransformer
    quality_pred = recipe_model.predict(transformed_new)[0] #gives a single prediction, 0 or 1
    quality_proba = recipe_model.predict_proba(transformed_new)[0] #returns an array of probabilities per class per row
    good_idx = list(recipe_model.classes_).index(1) #find where value 1 sits in the array
    quality_label = "Good recipe" if quality_pred == 1 else "Bad recipe"

    print("\n" + "-"*40)
    print(f"Quality prediction: {quality_label}")
    print(f"  Confidence: {quality_proba[good_idx] * 100:.1f}% good / " f"{(1 - quality_proba[good_idx]) * 100:.1f}% bad")
    #cuisine prediction reusing the fitted cuisine_prep + knn2
    transformed_new_c = cuisine_prep.transform(new_row)
    cuisine_probs = cuisine_model.predict_proba(transformed_new_c)[0]
    cuisine_classes = cuisine_model.classes_

    print("\nCuisine probability distribution:")
    ranked = sorted(zip(cuisine_classes, cuisine_probs), key=lambda x: x[1], reverse=True)
    for cuisine, prob in ranked: #unpacking each pair and printing only ones with nonzero probability
        if prob > 0:
            print(f"  {cuisine}: {prob*100:.1f}%")
    print("-"*40 + "\n")

def change_param(param1, param2):
    if USER_MODEL == Models.KNN:
        hyperparams[0]["param1"] = param1
        hyperparams[0]["param2"] = param2
    if USER_MODEL == Models.DTREE:
        hyperparams[1]["param1"] = param1
        hyperparams[1]["param2"] = param2
    if USER_MODEL == Models.RFOREST:
        hyperparams[2]["param1"] = param1
        hyperparams[2]["param2"] = param2

def choose_model():
    param1, param2 = None, None
    choice = input(f"Choose a model (1 - KNN | 2 - DTREE | 3 - RFOREST)\n"
                   f"(Optionally, manually enter model parameters seperated by comma)\n"
                   f"(e.g. '1, 4, 5' results in KNN with K=4 for recipe, K=5 for cuisine): ").strip()
    input_list = choice.split(",")
    for i in range(0, len(input_list)):
        item = input_list[i].strip()
        item_clean = "".join([char for char in item if char.isdigit()])
        input_list[i] = item_clean
    choice_clean = input_list[0]
    try:
        model_chosen = Models(int(choice_clean))
        if len(input_list) >= 3:
            param1 = int(input_list[1])
            param2 = int(input_list[2])
    except ValueError:
        model_chosen = None
    return model_chosen, param1, param2

if __name__ == "__main__":
    # Take user choice of which model to use
    param1, param2 = None, None
    while USER_MODEL is None:
        USER_MODEL, param1, param2 = choose_model()
    print(f"{USER_MODEL.name} chosen")

    if param1 and param2:
        print(f"{param1} and {param2} parameters entered")
        change_param(param1, param2)

    hyperparam_index = USER_MODEL.value - 1
    # Perform data preprocessing depending on model chosen
    (
        transformed_X_train, transformed_X_test, y_train, y_test,
        x_train_c, x_test_c, y_train_cuisine, y_test_cuisine, preprocessor,
        cuisine_prep, ing_counts, prefix_remove, base_ing
    ) = (data_preprocess(*hyperparams[hyperparam_index].values()))

    min_ingredient_freq = hyperparams[hyperparam_index].get("min_ingredient_freq")

    # Train models as such and supply to prediction functions
    recipe_results, cuisine_results = train_models(transformed_X_train, transformed_X_test, y_train,
                                                   y_test, x_train_c, x_test_c, y_train_cuisine, y_test_cuisine,
                                                   hyperparams[hyperparam_index])
    recipe_model = recipe_results["model"]
    cuisine_model = cuisine_results["model"]

    while True:
        # Parse URL from user
        url = input("Enter the URL of a recipe from food.com or press enter for manual entry: ").strip()
        if url:
            # If URL is invalid just retry loop again
            url_data = parse_url(url)
            if url_data is None:
                continue
            else:
                predict_new_recipe(recipe_model, cuisine_model, preprocessor, cuisine_prep,
                                   ing_counts, prefix_remove, base_ing, min_ingredient_freq,
                                   url_data)
        else:
            predict_new_recipe(recipe_model, cuisine_model, preprocessor, cuisine_prep,
                               ing_counts, prefix_remove, base_ing, min_ingredient_freq)
        again = input("Classify another recipe? (y/n): ").strip().lower()
        if again != "y":
            break
