"""
IMDb Exploratory Data Analysis Script

Performs comprehensive exploratory analysis on IMDb data including:
- Genre trends over time
- Director and actor performance metrics
- Budget and earnings relationships
- Regional rating differences
- Runtime vs rating correlations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_merged_data(data_dir='data'):
    """
    Load the preprocessed merged dataset.
    
    Args:
        data_dir (str): Directory containing the data
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    filepath = os.path.join(data_dir, 'merged_imdb_data.csv')
    print(f"Loading merged dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records")
    return df

def analyze_genre_trends(df, output_dir='plots'):
    """
    Analyze movie trends across different genres.
    
    Args:
        df (pd.DataFrame): Merged dataset
        output_dir (str): Directory to save plots
    """
    print("\n" + "="*60)
    print("GENRE ANALYSIS")
    print("="*60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Explode genres (each genre becomes a separate row)
    df_genres = df.copy()
    df_genres['genres'] = df_genres['genres'].fillna('Unknown')
    df_genres = df_genres[df_genres['genres'] != ''].copy()
    
    # Split genres and create separate rows
    genre_data = []
    for _, row in df_genres.iterrows():
        genres = str(row['genres']).split(',')
        for genre in genres:
            genre_data.append({
                'genre': genre.strip(),
                'averageRating': row['averageRating'],
                'numVotes': row['numVotes'],
                'startYear': row['startYear'],
                'runtimeMinutes': row['runtimeMinutes']
            })
    
    df_genre_exploded = pd.DataFrame(genre_data)
    
    # Top genres by count
    print("\nTop 15 genres by number of titles:")
    genre_counts = df_genre_exploded['genre'].value_counts().head(15)
    print(genre_counts)
    
    # Average rating by genre
    print("\nAverage rating by genre (top 15):")
    genre_ratings = df_genre_exploded.groupby('genre')['averageRating'].mean().sort_values(ascending=False).head(15)
    print(genre_ratings)
    
    # Genre trends over time
    print("\nAnalyzing genre popularity over time...")
    top_genres = genre_counts.head(10).index.tolist()
    df_top_genres = df_genre_exploded[df_genre_exploded['genre'].isin(top_genres)]
    
    # Count titles per year for each genre
    genre_year_counts = df_top_genres.groupby(['startYear', 'genre']).size().reset_index(name='count')
    
    # Filter for recent years (2000 onwards) for clearer visualization
    genre_year_counts = genre_year_counts[genre_year_counts['startYear'] >= 2000]
    
    print(f"Genre trends calculated for {len(genre_year_counts)} data points")
    
    return df_genre_exploded, genre_counts

def analyze_ratings_distribution(df, output_dir='plots'):
    """
    Analyze the distribution of ratings across different dimensions.
    
    Args:
        df (pd.DataFrame): Merged dataset
        output_dir (str): Directory to save plots
    """
    print("\n" + "="*60)
    print("RATINGS DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Overall rating distribution
    print("\nRating distribution statistics:")
    print(df['averageRating'].describe())
    
    # Ratings by title type
    print("\nAverage rating by title type:")
    type_ratings = df.groupby('titleType')['averageRating'].agg(['mean', 'median', 'std', 'count'])
    print(type_ratings)
    
    # Ratings by decade
    df['decade'] = (df['startYear'] // 10) * 10
    print("\nAverage rating by decade:")
    decade_ratings = df.groupby('decade')['averageRating'].agg(['mean', 'median', 'count'])
    print(decade_ratings[decade_ratings['count'] >= 100])  # Only decades with sufficient data
    
    return type_ratings, decade_ratings

def analyze_runtime_correlations(df):
    """
    Analyze correlations between runtime and other variables.
    
    Args:
        df (pd.DataFrame): Merged dataset
    """
    print("\n" + "="*60)
    print("RUNTIME CORRELATION ANALYSIS")
    print("="*60)
    
    # Runtime statistics
    print("\nRuntime statistics:")
    print(df['runtimeMinutes'].describe())
    
    # Correlation between runtime and rating
    runtime_rating_corr = df[['runtimeMinutes', 'averageRating']].corr()
    print("\nCorrelation between runtime and rating:")
    print(runtime_rating_corr)
    
    # Runtime by title type
    print("\nAverage runtime by title type:")
    runtime_by_type = df.groupby('titleType')['runtimeMinutes'].agg(['mean', 'median', 'std'])
    print(runtime_by_type)
    
    # Optimal runtime for high ratings
    print("\nAverage rating by runtime categories:")
    df['runtime_category'] = pd.cut(df['runtimeMinutes'], 
                                     bins=[0, 60, 90, 120, 150, 500],
                                     labels=['Short (<60)', 'Standard (60-90)', 
                                            'Long (90-120)', 'Very Long (120-150)', 
                                            'Epic (>150)'])
    runtime_cat_ratings = df.groupby('runtime_category')['averageRating'].agg(['mean', 'median', 'count'])
    print(runtime_cat_ratings)
    
    return runtime_rating_corr

def analyze_director_performance(df, name_basics_path='data/name_basics.tsv'):
    """
    Analyze director performance metrics.
    
    Args:
        df (pd.DataFrame): Merged dataset
        name_basics_path (str): Path to name basics dataset
    """
    print("\n" + "="*60)
    print("DIRECTOR PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Filter out entries without directors
    df_with_directors = df[df['directors'].notna() & (df['directors'] != '')].copy()
    
    # Explode directors (handle multiple directors)
    director_data = []
    for _, row in df_with_directors.iterrows():
        directors = str(row['directors']).split(',')
        for director_id in directors:
            director_id = director_id.strip()
            if director_id:
                director_data.append({
                    'director_id': director_id,
                    'tconst': row['tconst'],
                    'averageRating': row['averageRating'],
                    'numVotes': row['numVotes'],
                    'startYear': row['startYear'],
                    'primaryTitle': row['primaryTitle']
                })
    
    df_directors = pd.DataFrame(director_data)
    
    # Calculate director statistics
    director_stats = df_directors.groupby('director_id').agg({
        'averageRating': ['mean', 'median', 'std'],
        'numVotes': ['mean', 'sum'],
        'tconst': 'count'
    }).reset_index()
    
    director_stats.columns = ['director_id', 'avg_rating', 'median_rating', 'std_rating',
                              'avg_votes', 'total_votes', 'num_titles']
    
    # Filter directors with at least 3 titles
    director_stats = director_stats[director_stats['num_titles'] >= 3]
    
    print(f"\nDirectors with at least 3 titles: {len(director_stats):,}")
    
    # Top directors by average rating
    print("\nTop 20 directors by average rating (min 5 titles):")
    top_directors = director_stats[director_stats['num_titles'] >= 5].nlargest(20, 'avg_rating')
    
    # Try to load names if available
    try:
        if os.path.exists(name_basics_path):
            name_basics = pd.read_csv(name_basics_path, sep='\t', na_values='\\N', 
                                     usecols=['nconst', 'primaryName'])
            top_directors = top_directors.merge(name_basics, 
                                               left_on='director_id', 
                                               right_on='nconst', 
                                               how='left')
            print(top_directors[['primaryName', 'avg_rating', 'num_titles', 'total_votes']].to_string(index=False))
        else:
            print(top_directors[['director_id', 'avg_rating', 'num_titles', 'total_votes']].to_string(index=False))
    except Exception as e:
        print(f"Could not load director names: {e}")
        print(top_directors[['director_id', 'avg_rating', 'num_titles', 'total_votes']].to_string(index=False))
    
    return df_directors, director_stats

def analyze_popularity_factors(df):
    """
    Analyze factors affecting movie popularity (number of votes).
    
    Args:
        df (pd.DataFrame): Merged dataset
    """
    print("\n" + "="*60)
    print("POPULARITY FACTORS ANALYSIS")
    print("="*60)
    
    # Popularity distribution
    print("\nVotes distribution:")
    print(df['numVotes'].describe())
    
    # Create popularity categories
    df['popularity_category'] = pd.qcut(df['numVotes'], 
                                        q=4, 
                                        labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Average rating by popularity
    print("\nAverage rating by popularity category:")
    pop_ratings = df.groupby('popularity_category')['averageRating'].agg(['mean', 'median', 'count'])
    print(pop_ratings)
    
    # Correlation matrix
    print("\nCorrelation matrix for key metrics:")
    correlation_cols = ['averageRating', 'numVotes', 'runtimeMinutes', 'startYear']
    corr_matrix = df[correlation_cols].corr()
    print(corr_matrix)
    
    return corr_matrix

def generate_insights_report(df, genre_data, director_stats):
    """
    Generate a comprehensive insights report.
    
    Args:
        df (pd.DataFrame): Merged dataset
        genre_data (pd.DataFrame): Genre analysis data
        director_stats (pd.DataFrame): Director statistics
    """
    print("\n" + "="*70)
    print(" "*20 + "KEY INSIGHTS REPORT")
    print("="*70)
    
    print("\n1. DATASET OVERVIEW")
    print(f"   - Total titles analyzed: {len(df):,}")
    print(f"   - Year range: {df['startYear'].min():.0f} - {df['startYear'].max():.0f}")
    print(f"   - Average rating: {df['averageRating'].mean():.2f}")
    print(f"   - Median votes: {df['numVotes'].median():,.0f}")
    
    print("\n2. GENRE INSIGHTS")
    top_3_genres = genre_data['genre'].value_counts().head(3)
    print(f"   - Most common genres: {', '.join(top_3_genres.index.tolist())}")
    best_rated = genre_data.groupby('genre')['averageRating'].mean().nlargest(3)
    print(f"   - Highest rated genres: {', '.join(best_rated.index.tolist())}")
    
    print("\n3. RUNTIME INSIGHTS")
    avg_runtime = df['runtimeMinutes'].mean()
    print(f"   - Average runtime: {avg_runtime:.1f} minutes")
    high_rated = df[df['averageRating'] >= 8.0]['runtimeMinutes'].mean()
    print(f"   - Average runtime of highly-rated titles (8.0+): {high_rated:.1f} minutes")
    
    print("\n4. DIRECTOR INSIGHTS")
    print(f"   - Directors with 3+ titles: {len(director_stats):,}")
    prolific = director_stats.nlargest(1, 'num_titles')
    print(f"   - Most prolific director: {prolific['num_titles'].values[0]:.0f} titles")
    
    print("\n5. TEMPORAL TRENDS")
    recent = df[df['startYear'] >= 2010]
    older = df[(df['startYear'] >= 1990) & (df['startYear'] < 2010)]
    print(f"   - Average rating (2010+): {recent['averageRating'].mean():.2f}")
    print(f"   - Average rating (1990-2009): {older['averageRating'].mean():.2f}")
    
    print("\n6. POPULARITY PATTERNS")
    popular = df.nlargest(1000, 'numVotes')['averageRating'].mean()
    less_popular = df.nsmallest(1000, 'numVotes')['averageRating'].mean()
    print(f"   - Average rating of top 1000 popular titles: {popular:.2f}")
    print(f"   - Average rating of bottom 1000 popular titles: {less_popular:.2f}")
    
    print("\n" + "="*70)

def main():
    """Main exploratory analysis pipeline."""
    
    print("="*60)
    print("IMDb Exploratory Data Analysis")
    print("="*60)
    
    # Check if merged data exists
    if not os.path.exists('data/merged_imdb_data.csv'):
        print("\nError: Merged dataset not found!")
        print("Please run data_preprocessing.py first.")
        return
    
    try:
        # Load data
        df = load_merged_data()
        
        # Perform analyses
        genre_data, genre_counts = analyze_genre_trends(df)
        type_ratings, decade_ratings = analyze_ratings_distribution(df)
        runtime_corr = analyze_runtime_correlations(df)
        director_data, director_stats = analyze_director_performance(df)
        corr_matrix = analyze_popularity_factors(df)
        
        # Generate insights report
        generate_insights_report(df, genre_data, director_stats)
        
        print("\n" + "="*60)
        print("Exploratory analysis completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run visualization.py to create interactive visualizations")
        print("2. Run predictive_models.py to build prediction models")
        print("3. Open analysis_notebook.ipynb for interactive exploration")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
