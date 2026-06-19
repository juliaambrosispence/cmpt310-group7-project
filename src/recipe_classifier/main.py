# main.py
# Main project entry point.
# Load dataset from kaggle
# Lots more to do, process data, set up models etc.

import kagglehub
import pandas as pd
from pathlib import Path

# Download the dataset, if it already exists in cache then this returns quickly
dataset_dir = Path(kagglehub.dataset_download("irkaal/foodcom-recipes-and-reviews"))
print(f"Dataset located at {dataset_dir}")

# This is just temporary to see we have the data downloaded properly
recipes_df = pd.read_parquet(dataset_dir / "recipes.parquet")
print(recipes_df.head())

reviews_df = pd.read_parquet(dataset_dir / "reviews.parquet")
print(reviews_df.head())

# Next we should do processing on the data, we don't need all 500K entries