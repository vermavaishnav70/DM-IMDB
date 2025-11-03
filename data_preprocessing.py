"""
IMDb Data Preprocessing Script

Cleans and preprocesses IMDb datasets for analysis.
Handles missing values, data type conversions, and creates merged datasets.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_dataset(filename, data_dir='data'):
    """
    Load a TSV dataset with proper handling of IMDb null values.
    
    Args:
        filename (str): Name of the TSV file
        data_dir (str): Directory containing the data
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    filepath = os.path.join(data_dir, filename)
    print(f"Loading {filename}...")
    
    # IMDb uses '\\N' to represent NULL values
    df = pd.read_csv(filepath, sep='\t', na_values='\\N', low_memory=False)
    print(f"Loaded {len(df):,} records from {filename}")
    
    return df

def preprocess_title_basics(df):
    """
    Preprocess title basics dataset.
    
    Args:
        df (pd.DataFrame): Raw title basics dataframe
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print("\nPreprocessing title_basics...")
    
    # Filter for movies and TV series
    df = df[df['titleType'].isin(['movie', 'tvSeries', 'tvMovie'])].copy()
    
    # Convert year columns to numeric
    df['startYear'] = pd.to_numeric(df['startYear'], errors='coerce')
    df['endYear'] = pd.to_numeric(df['endYear'], errors='coerce')
    
    # Convert runtime to numeric
    df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'], errors='coerce')
    
    # Filter out very old movies and invalid data
    df = df[df['startYear'] >= 1900]
    df = df[df['runtimeMinutes'] > 0]
    
    # Convert isAdult to boolean
    df['isAdult'] = df['isAdult'].astype(str) == '1'
    
    print(f"After preprocessing: {len(df):,} records")
    
    return df

def preprocess_title_ratings(df):
    """
    Preprocess title ratings dataset.
    
    Args:
        df (pd.DataFrame): Raw title ratings dataframe
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print("\nPreprocessing title_ratings...")
    
    # Ensure numeric types
    df['averageRating'] = pd.to_numeric(df['averageRating'], errors='coerce')
    df['numVotes'] = pd.to_numeric(df['numVotes'], errors='coerce')
    
    # Filter out entries with insufficient votes (less than 100)
    df = df[df['numVotes'] >= 100].copy()
    
    print(f"After preprocessing: {len(df):,} records")
    
    return df

def preprocess_title_crew(df):
    """
    Preprocess title crew dataset.
    
    Args:
        df (pd.DataFrame): Raw title crew dataframe
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print("\nPreprocessing title_crew...")
    
    # Split comma-separated director and writer IDs
    df['directors'] = df['directors'].fillna('')
    df['writers'] = df['writers'].fillna('')
    
    print(f"After preprocessing: {len(df):,} records")
    
    return df

def preprocess_name_basics(df):
    """
    Preprocess name basics dataset.
    
    Args:
        df (pd.DataFrame): Raw name basics dataframe
        
    Returns:
        pd.DataFrame: Preprocessed dataframe
    """
    print("\nPreprocessing name_basics...")
    
    # Convert year columns to numeric
    df['birthYear'] = pd.to_numeric(df['birthYear'], errors='coerce')
    df['deathYear'] = pd.to_numeric(df['deathYear'], errors='coerce')
    
    print(f"After preprocessing: {len(df):,} records")
    
    return df

def create_merged_dataset(title_basics, title_ratings, title_crew, output_dir='data'):
    """
    Create a merged dataset combining title information, ratings, and crew.
    
    Args:
        title_basics (pd.DataFrame): Preprocessed title basics
        title_ratings (pd.DataFrame): Preprocessed title ratings
        title_crew (pd.DataFrame): Preprocessed title crew
        output_dir (str): Directory to save the merged dataset
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    print("\nCreating merged dataset...")
    
    # Merge title basics with ratings
    merged = title_basics.merge(title_ratings, on='tconst', how='inner')
    print(f"After merging with ratings: {len(merged):,} records")
    
    # Merge with crew information
    merged = merged.merge(title_crew, on='tconst', how='left')
    print(f"After merging with crew: {len(merged):,} records")
    
    # Calculate weighted rating (IMDB's formula)
    C = merged['averageRating'].mean()
    m = merged['numVotes'].quantile(0.9)
    
    def weighted_rating(row, C=C, m=m):
        v = row['numVotes']
        R = row['averageRating']
        return (v / (v + m) * R) + (m / (v + m) * C)
    
    merged['weightedRating'] = merged.apply(weighted_rating, axis=1)
    
    # Save merged dataset
    output_path = os.path.join(output_dir, 'merged_imdb_data.csv')
    merged.to_csv(output_path, index=False)
    print(f"\nMerged dataset saved to: {output_path}")
    
    return merged

def generate_summary_statistics(df):
    """
    Generate and display summary statistics for the merged dataset.
    
    Args:
        df (pd.DataFrame): Merged dataset
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nTotal records: {len(df):,}")
    print(f"\nTitle types distribution:")
    print(df['titleType'].value_counts())
    
    print(f"\nYear range: {df['startYear'].min():.0f} - {df['startYear'].max():.0f}")
    
    print(f"\nRating statistics:")
    print(df['averageRating'].describe())
    
    print(f"\nRuntime statistics:")
    print(df['runtimeMinutes'].describe())
    
    print(f"\nTop 10 genres:")
    all_genres = df['genres'].str.split(',').explode()
    print(all_genres.value_counts().head(10))
    
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("No missing values in key columns")
    
    print("\n" + "="*60)

def main():
    """Main preprocessing pipeline."""
    
    print("="*60)
    print("IMDb Data Preprocessing Pipeline")
    print("="*60)
    
    data_dir = 'data'
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"\nError: Data directory '{data_dir}' not found!")
        print("Please run data_acquisition.py first to download the datasets.")
        return
    
    try:
        # Load datasets
        title_basics = load_dataset('title_basics.tsv', data_dir)
        title_ratings = load_dataset('title_ratings.tsv', data_dir)
        title_crew = load_dataset('title_crew.tsv', data_dir)
        name_basics = load_dataset('name_basics.tsv', data_dir)
        
        # Preprocess individual datasets
        title_basics = preprocess_title_basics(title_basics)
        title_ratings = preprocess_title_ratings(title_ratings)
        title_crew = preprocess_title_crew(title_crew)
        name_basics = preprocess_name_basics(name_basics)
        
        # Create merged dataset
        merged_data = create_merged_dataset(title_basics, title_ratings, title_crew, data_dir)
        
        # Generate summary statistics
        generate_summary_statistics(merged_data)
        
        print("\nPreprocessing completed successfully!")
        print("\nNext steps:")
        print("1. Run exploratory_analysis.py to perform EDA")
        print("2. Run visualization.py to create interactive visualizations")
        print("3. Run predictive_models.py to build prediction models")
        
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("Please ensure all required datasets are downloaded.")
        print("Run data_acquisition.py to download the datasets.")
    except Exception as e:
        print(f"\nError during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
