# main.py
# Main project entry point.
# Load dataset from kaggle
# Lots more to do, process data, set up models etc.

import kagglehub
import pandas as pd
import sys
import numpy as np
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from pathlib import Path

N_ROWS = 10
RATING_THRESHOLD = 4.5
MIN_INGREDIENT_FREQ = 2

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
 'RecipeCategory' : "one-hot",
 'Keywords' : "drop",
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

# Define a modified MultiLabelBinarizer for use in ColumnTransformer
class ModifiedMultiLabelBinarizer(MultiLabelBinarizer):
 def __init__(self):
  self.mlb = MultiLabelBinarizer()

 def fit(self, X, y=None):
  # Extract first column as an array of iterable objects
  X_series = pd.Series(X.iloc[:, 0]) if hasattr(X, "iloc") else pd.Series(X)
  self.mlb.fit(X_series)
  return self

 def transform(self, X):
  X_series = pd.Series(X.iloc[:, 0]) if hasattr(X, "iloc") else pd.Series(X)
  return self.mlb.transform(X_series)

 def fit_transform(self, X, y=None):
  return self.fit(X, y).transform(X)

 def get_feature_names_out(self, input_features=None):
  return [f"{c}" for c in self.mlb.classes_]



# Download the dataset, if it already exists in cache then this returns quickly
dataset_dir = Path(kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews"))
print(f"Dataset located at {dataset_dir}")


# Parquet files need a different approach to only read a certain number of entries
parquet_recipes = pq.ParquetFile(dataset_dir / "recipes.parquet")
recipes_df = next(parquet_recipes.iter_batches(batch_size=N_ROWS)).to_pandas()

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
 if method == "time":
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
 recipes_df[column].str.replace("PT", "").str.replace("H", " hours ").str.replace("M", " minutes")).dt.total_seconds() / 60

# TODO: Split up data into training/testing sets

# Specify which columns to drop

all_ings = recipes_df['RecipeIngredientParts'].tolist() #make it into a list for the for loops.

#to fix the duplicates like "allpurpose flour" vs "flour"
base_ings = [ #basic ingredients to combine.
    "flour", "sugar", "oil", "chicken", "salt", "milk", "butter", 
    "lemon", "garlic", "onion", "soy sauce", "rice", "yogurt", "egg"
];

for i in range(0, len(all_ings)):#until end of list,
    current_recipe = all_ings[i];
    cleaned_recipe = []; #new combined ingredient list
    
    for j in range(0, len(current_recipe)): #loop for every ingredient
        item = str(current_recipe[j]).lower();#standardize lower
        
        for k in range(0, len(base_ings)):#shortens string with just 1 word.
            if item.find(base_ings[k]) != -1:
                item = base_ings[k];
                break; #found match, stop searching!
                
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
        if ing_counts[ing] >= MIN_INGREDIENT_FREQ: #if it doesnt meet minimum freq (2)
            popular_only.append(ing);
            
    all_ings[i] = popular_only

recipes_df['RecipeIngredientParts'] = all_ings #put list back into the main pandas table 


X = recipes_df.drop(columns=drop_columns)

# Our labels are 1 - good recipe if rating is above threshold, otherwise 0
y = (recipes_df['AggregatedRating'].astype(float) >= RATING_THRESHOLD).astype(int)

# TODO: We may need to take a look at how one-hot encoding handles the RecipeCategory and the ingredients
# The type of encoding we do will make a new feature out of every unique string we see in the category and ingredient
# lists, if we have an entry like "blueberries" vs "frozen blueberries" should this count as one feature?
# Or "all-purpose flour" should obviously be the same ingredient as "flour" we should do some string processing
# to strip prefix words like that, and perhaps eliminate certain ingredients?
# We could also limit the ingredient categories we make into features into ones that only show up several times

# Preprocess data using a ColumnTransformer to fit everything into one ndarray of data
preprocessor = ColumnTransformer(
 transformers=[
  ("numerical", StandardScaler(), standard_columns),
  ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
  ("multi-cat", ModifiedMultiLabelBinarizer(), multicat_columns),
 ]
)

# Apply fit of data based on the methods we specified to each set of columns, then apply transformations
transformed_X = preprocessor.fit_transform(X)

# Print resulting ndarray for just first row to see results
np.set_printoptions(threshold=sys.maxsize)
print(preprocessor.get_feature_names_out())
print(transformed_X[:1])
print(y)

# TODO: Take processed data and train a classifier, evaluate metrics, generate plots
