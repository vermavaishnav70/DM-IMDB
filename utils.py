"""
Utility functions shared across IMDb analysis scripts.

This module contains common helper functions used by multiple analysis scripts.
"""

import pandas as pd


def categorize_runtime(runtime_minutes):
    """
    Categorize movie runtime into standard categories.
    
    Args:
        runtime_minutes (float or pd.Series): Runtime in minutes
        
    Returns:
        str or pd.Series: Runtime category
    """
    return pd.cut(runtime_minutes, 
                  bins=[0, 60, 90, 120, 150, 500],
                  labels=['Short (<60)', 'Standard (60-90)', 
                         'Long (90-120)', 'Very Long (120-150)', 
                         'Epic (>150)'])


def truncate_title(title, max_length=30):
    """
    Truncate a title to a maximum length.
    
    Args:
        title (str): Title to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated title with ellipsis if needed
    """
    return f"{title[:max_length]}..." if len(title) > max_length else title


def calculate_weighted_rating(rating, votes, min_votes_quantile=0.9, df=None):
    """
    Calculate weighted rating using IMDb's formula.
    
    Args:
        rating (float or pd.Series): Average rating
        votes (int or pd.Series): Number of votes
        min_votes_quantile (float): Quantile for minimum votes threshold
        df (pd.DataFrame): Optional dataframe to calculate C and m from
        
    Returns:
        float or pd.Series: Weighted rating
    """
    if df is not None:
        C = df['averageRating'].mean()
        m = df['numVotes'].quantile(min_votes_quantile)
    else:
        # Default values if dataframe not provided
        C = 7.0
        m = 1000
    
    return (votes / (votes + m) * rating) + (m / (votes + m) * C)


def explode_genres(df, genre_column='genres'):
    """
    Explode a dataframe so that each genre becomes a separate row.
    
    Args:
        df (pd.DataFrame): Input dataframe
        genre_column (str): Name of the column containing comma-separated genres
        
    Returns:
        pd.DataFrame: Dataframe with exploded genres
    """
    df_copy = df.copy()
    df_copy[genre_column] = df_copy[genre_column].fillna('Unknown')
    df_copy = df_copy[df_copy[genre_column] != '']
    
    # Split and explode genres
    df_copy[genre_column] = df_copy[genre_column].str.split(',')
    df_exploded = df_copy.explode(genre_column)
    df_exploded[genre_column] = df_exploded[genre_column].str.strip()
    
    return df_exploded


def explode_directors(df, director_column='directors'):
    """
    Explode a dataframe so that each director becomes a separate row.
    
    Args:
        df (pd.DataFrame): Input dataframe
        director_column (str): Name of the column containing comma-separated director IDs
        
    Returns:
        pd.DataFrame: Dataframe with exploded directors
    """
    df_copy = df.copy()
    df_copy = df_copy[df_copy[director_column].notna() & (df_copy[director_column] != '')]
    
    # Split and explode directors
    df_copy[director_column] = df_copy[director_column].str.split(',')
    df_exploded = df_copy.explode(director_column)
    df_exploded[director_column] = df_exploded[director_column].str.strip()
    
    return df_exploded


def load_name_mapping(name_basics_path, name_column='primaryName', id_column='nconst'):
    """
    Load name to ID mapping from name_basics dataset.
    
    Args:
        name_basics_path (str): Path to name_basics.tsv file
        name_column (str): Column containing names
        id_column (str): Column containing IDs
        
    Returns:
        dict: Mapping from ID to name
    """
    try:
        names_df = pd.read_csv(name_basics_path, sep='\t', na_values='\\N',
                              usecols=[id_column, name_column])
        return dict(zip(names_df[id_column], names_df[name_column]))
    except Exception as e:
        print(f"Warning: Could not load name mapping: {e}")
        return {}


def get_decade(year):
    """
    Get decade from year.
    
    Args:
        year (int or pd.Series): Year value(s)
        
    Returns:
        int or pd.Series: Decade value(s)
    """
    return (year // 10) * 10
