# main.py
# Main project entry point.
# Load dataset from kaggle
# Lots more to do, process data, set up models etc.

import kagglehub
import pandas as pd
import pyarrow.parquet as pq
import sklearn as sk
from pathlib import Path

N_ROWS = 10
RATING_THRESHOLD = 4.5

# Download the dataset, if it already exists in cache then this returns quickly
dataset_dir = Path(kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews"))
print(f"Dataset located at {dataset_dir}")


# Parquet files need a different approach to only read a certain number of entries
parquet_recipes = pq.ParquetFile(dataset_dir / "recipes.parquet")
recipes_df = next(parquet_recipes.iter_batches(batch_size=N_ROWS)).to_pandas()
#parquet_reviews = pq.ParquetFile(dataset_dir / "reviews.parquet")
#reviews_df = next(parquet_reviews.iter_batches(batch_size=N_ROWS)).to_pandas()


# This is just temporary to see we have the data downloaded properly
#recipes_df = pd.read_parquet(dataset_dir / "recipes.parquet", nrows=N_ROWS)
print(recipes_df)

#print(recipes_df.columns.tolist())

#reviews_df = pd.read_parquet(dataset_dir / "reviews.parquet", nrows=N_ROWS)
#print(reviews_df.head())

#print(reviews_df.columns.tolist())
# Next we should do processing on the data

# Specify which columns to drop
X = recipes_df.drop(columns=[
'RecipeId', 'Name', 'AuthorId', 'AuthorName',
 'TotalTime', 'DatePublished', 'Description',
 'Images', 'Keywords', 'RecipeIngredientQuantities',
 'AggregatedRating',
 'RecipeServings', 'RecipeInstructions'
])

# Our labels are 1 - good recipe if rating is above threshold, otherwise 0
y = (recipes_df['AggregatedRating'].astype(float) >= RATING_THRESHOLD).astype(int)

# Specify which columns are numeric standardized features and which are categoric (one-hot)
numeric_features = []
categorical_features = []

print(X)
print(y)