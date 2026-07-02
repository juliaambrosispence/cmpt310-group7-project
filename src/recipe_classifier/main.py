# main.py
# Main project entry point.
# Load dataset from kaggle
# Lots more to do, process data, set up models etc.

import kagglehub
import pandas as pd
import sys
import numpy as np
import pyarrow.parquet as pq
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils import *

N_ROWS = 500
RATING_THRESHOLD = 4.5
MIN_INGREDIENT_FREQ = 5
MIN_KEYWORD_FREQ = 5

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
  #return [f"{c}" for c in self.mlb.classes_]



# Download the dataset, if it already exists in cache then this returns quickly
dataset_dir = Path(kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews"))
print(f"Dataset located at {dataset_dir}")


# Parquet files need a different approach to only read a certain number of entries
parquet_recipes = pd.read_parquet(dataset_dir / "recipes.parquet")#pq.ParquetFile(dataset_dir / "recipes.parquet")
#recipes_df = next(parquet_recipes.iter_batches(batch_size=N_ROWS)).to_pandas()
recipes_df = parquet_recipes.sample(n=N_ROWS, random_state=67)



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

# TODO: Split up data into training/testing sets

# Keywords work like ingredients
keyword_ings = recipes_df['Keywords'].tolist()
#count_string_frequency(keyword_ings, 500)

# Remove any None entries from Keywords column
for i in range(0, len(keyword_ings)):
    if keyword_ings[i][0] is None:
        keyword_ings[i] = []

# Only keep keywords that are relevant to cuisine type
keep_keywords = [
             'Canadian', 'Filipino', 'Brazilian', 'Danish', 'Turkish', 'South African', 'Indonesian',
    'Peruvian', 'Palestinian', 'Finnish',
            'Chinese', 'Indian', 'Australian', 'Southwest Asia (middle East)',
             'Thai', 'Southwestern U.S.', 'Greek', 'Moroccan', 'Japanese',
            'Spanish', 'German', 'Hawaiian', 'Swiss', 'Cajun', 'Austrian', 'Russian', 'Scottish', 'South American',
            'Polish', 'Swedish', 'New Zealand', 'Malaysian', 'Native American', 'Dutch',
            'Puerto Rican', 'Cuban', 'Vietnamese', 'Egyptian', 'Hungarian', 'Belgian', 'Portuguese'
    'Asian', 'Mexican', 'African', 'Caribbean',  'Scandinavian',
]

new_keywords = ["None"] * len(keyword_ings)

# TODO: We probably need a way to choose which cuisine to pick out of keywords list, currently it will simply
# overwrite the category as it goes through keywords list
for i in range(0, len(keyword_ings)):
    keywords = keyword_ings[i]
    #keep_words = []
    for keyword in keywords:
        if keyword in keep_keywords:
            #keep_words.append(keyword)
            new_keywords[i] = keyword

# Put keywords back into dataset
recipes_df['Keywords'] = new_keywords

#ok sorry I changed this cause I realized could actually use none and filter that out. (╥﹏╥)
recipes_df = recipes_df[recipes_df['Keywords'] != "None"].reset_index(drop=True) #if keywords is not 'none' then true, reset rows back to 0

count_category_frequency(recipes_df['Keywords'].tolist(), 500)#changed to pass on new column, so dont print 'none'

all_ings = recipes_df['RecipeIngredientParts'].tolist() #make it into a list for the for loops.

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
    "garlic"         : ["minced garlic", "garlic cloves", "garlic"],
    "onion"          : ["green onion", "red onion", "yellow onion", "spring onion", "onion"],
    "yogurt"         : ["plain yogurt", "greek yogurt", "vanilla yogurt", "yogurt"],
    "rice vinegar"   : ["rice vinegar"],
    "rice"           : ["basmati rice", "jasmine rice", "brown rice", "long-grain rice", "rice"],
    "egg"            : ["large eggs", "whole eggs", "eggs", "egg"],
    "tomato"         : ["cherry tomatoes", "diced tomatoes", "tomato paste", "tomato sauce", "tomatoes", "tomato"],
    "oil"            : ["oil"]
}

for i in range(0, len(all_ings)):#until end of list,
    # Remove duplicates from recipe ingredients
    current_recipe = list(dict.fromkeys(all_ings[i]));
    cleaned_recipe = []; #new combined ingredient list

    for j in range(0, len(current_recipe)): #loop for every ingredient
        item = str(current_recipe[j]).lower();#standardize lower
    
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
        if ing_counts[ing] >= MIN_INGREDIENT_FREQ: #if it doesnt meet minimum freq (2)
            popular_only.append(ing);
            
    all_ings[i] = popular_only

recipes_df['RecipeIngredientParts'] = all_ings #put list back into the main pandas table 

count_string_frequency(all_ings, 500)

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
  ("ingredients", ModifiedMultiLabelBinarizer(), ["RecipeIngredientParts"]),
  #("keywords", ModifiedMultiLabelBinarizer(), ["Keywords"]),
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

feature_names = preprocessor.get_feature_names_out()
sample_x = transformed_X[0] 
sample_y = y.iloc[0]          


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
