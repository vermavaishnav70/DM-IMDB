"""
IMDb Data Visualization Script

Creates interactive visualizations including:
- Top movies by director
- Genre distribution and trends
- Rating and earnings visualizations
- Regional analysis
- Correlation heatmaps
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

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

def load_data(data_dir='data'):
    """Load the preprocessed merged dataset."""
    filepath = os.path.join(data_dir, 'merged_imdb_data.csv')
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records")
    return df

def create_output_directory(output_dir='plots'):
    """Create output directory for plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return output_dir

def visualize_top_movies_by_director(df, name_basics_path='data/name_basics.tsv', output_dir='plots'):
    """
    Create interactive visualization of top 3 movies by director.
    
    Args:
        df (pd.DataFrame): Merged dataset
        name_basics_path (str): Path to name basics dataset
        output_dir (str): Directory to save plots
    """
    print("\nCreating Top Movies by Director visualization...")
    
    # Filter movies with directors
    df_with_directors = df[df['directors'].notna() & (df['directors'] != '')].copy()
    
    # Explode directors
    director_movies = []
    for _, row in df_with_directors.iterrows():
        directors = str(row['directors']).split(',')
        for director_id in directors:
            director_id = director_id.strip()
            if director_id:
                director_movies.append({
                    'director_id': director_id,
                    'title': row['primaryTitle'],
                    'rating': row['averageRating'],
                    'votes': row['numVotes'],
                    'year': row['startYear']
                })
    
    df_director_movies = pd.DataFrame(director_movies)
    
    # Count movies per director
    director_counts = df_director_movies['director_id'].value_counts()
    
    # Get directors with at least 5 movies
    prolific_directors = director_counts[director_counts >= 5].head(20).index.tolist()
    
    # Load director names
    director_names = {}
    try:
        if os.path.exists(name_basics_path):
            name_basics = pd.read_csv(name_basics_path, sep='\t', na_values='\\N', 
                                     usecols=['nconst', 'primaryName'])
            for _, row in name_basics.iterrows():
                director_names[row['nconst']] = row['primaryName']
    except Exception as e:
        print(f"Could not load director names: {e}")
    
    # Get top 3 movies for each prolific director
    top_movies_data = []
    for director_id in prolific_directors:
        director_movies_df = df_director_movies[df_director_movies['director_id'] == director_id]
        top_3 = director_movies_df.nlargest(3, 'rating')
        
        director_name = director_names.get(director_id, director_id)
        
        for idx, (_, movie) in enumerate(top_3.iterrows(), 1):
            top_movies_data.append({
                'Director': director_name,
                'Rank': idx,
                'Title': movie['title'],
                'Rating': movie['rating'],
                'Votes': movie['votes'],
                'Year': int(movie['year'])
            })
    
    df_top_movies = pd.DataFrame(top_movies_data)
    
    # Create interactive plot
    fig = go.Figure()
    
    # Group by director and create bars
    for director in df_top_movies['Director'].unique()[:10]:  # Top 10 directors
        director_data = df_top_movies[df_top_movies['Director'] == director]
        
        fig.add_trace(go.Bar(
            name=director,
            x=[truncate_title(row['Title'], 30) for _, row in director_data.iterrows()],
            y=director_data['Rating'],
            text=director_data['Rating'].round(2),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         'Rating: %{y:.2f}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title='Top 3 Movies by Director (Top 10 Directors)',
        xaxis_title='Movie Title',
        yaxis_title='Average Rating',
        barmode='group',
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    output_file = os.path.join(output_dir, 'top_movies_by_director.html')
    fig.write_html(output_file)
    print(f"Saved: {output_file}")
    
    return df_top_movies

def visualize_genre_distribution(df, output_dir='plots'):
    """
    Create genre distribution visualizations.
    
    Args:
        df (pd.DataFrame): Merged dataset
        output_dir (str): Directory to save plots
    """
    print("\nCreating Genre Distribution visualizations...")
    
    # Explode genres
    df['genres'] = df['genres'].fillna('Unknown')
    genre_list = df['genres'].str.split(',').explode().str.strip()
    genre_counts = genre_list.value_counts().head(15)
    
    # Create interactive bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=genre_counts.index,
            y=genre_counts.values,
            marker_color='steelblue',
            text=genre_counts.values,
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='Top 15 Genres by Number of Titles',
        xaxis_title='Genre',
        yaxis_title='Number of Titles',
        height=500
    )
    
    output_file = os.path.join(output_dir, 'genre_distribution.html')
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

def visualize_rating_trends(df, output_dir='plots'):
    """
    Create rating trend visualizations over time.
    
    Args:
        df (pd.DataFrame): Merged dataset
        output_dir (str): Directory to save plots
    """
    print("\nCreating Rating Trends visualization...")
    
    # Filter for years with sufficient data
    df_filtered = df[df['startYear'] >= 1950].copy()
    
    # Calculate average rating by year
    yearly_ratings = df_filtered.groupby('startYear').agg({
        'averageRating': ['mean', 'median'],
        'tconst': 'count'
    }).reset_index()
    
    yearly_ratings.columns = ['Year', 'Mean Rating', 'Median Rating', 'Count']
    
    # Filter years with at least 50 titles
    yearly_ratings = yearly_ratings[yearly_ratings['Count'] >= 50]
    
    # Create interactive line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly_ratings['Year'],
        y=yearly_ratings['Mean Rating'],
        mode='lines+markers',
        name='Mean Rating',
        line=dict(color='blue', width=2),
        hovertemplate='Year: %{x}<br>Mean Rating: %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=yearly_ratings['Year'],
        y=yearly_ratings['Median Rating'],
        mode='lines+markers',
        name='Median Rating',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='Year: %{x}<br>Median Rating: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Average Movie Ratings Over Time',
        xaxis_title='Year',
        yaxis_title='Rating',
        height=500,
        hovermode='x unified'
    )
    
    output_file = os.path.join(output_dir, 'rating_trends.html')
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

def visualize_runtime_vs_rating(df, output_dir='plots'):
    """
    Create runtime vs rating scatter plot.
    
    Args:
        df (pd.DataFrame): Merged dataset
        output_dir (str): Directory to save plots
    """
    print("\nCreating Runtime vs Rating visualization...")
    
    # Sample data for better performance
    df_sample = df.sample(min(5000, len(df)), random_state=42)
    
    # Create scatter plot
    fig = px.scatter(
        df_sample,
        x='runtimeMinutes',
        y='averageRating',
        color='numVotes',
        size='numVotes',
        hover_data=['primaryTitle', 'startYear'],
        title='Runtime vs Rating (Sample of 5000 titles)',
        labels={
            'runtimeMinutes': 'Runtime (minutes)',
            'averageRating': 'Average Rating',
            'numVotes': 'Number of Votes'
        },
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=600)
    
    output_file = os.path.join(output_dir, 'runtime_vs_rating.html')
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

def visualize_correlation_heatmap(df, output_dir='plots'):
    """
    Create correlation heatmap for key metrics.
    
    Args:
        df (pd.DataFrame): Merged dataset
        output_dir (str): Directory to save plots
    """
    print("\nCreating Correlation Heatmap...")
    
    # Select numeric columns
    numeric_cols = ['averageRating', 'numVotes', 'runtimeMinutes', 'startYear']
    corr_matrix = df[numeric_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=corr_matrix.values.round(3),
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Heatmap of Key Metrics',
        height=500,
        width=600
    )
    
    output_file = os.path.join(output_dir, 'correlation_heatmap.html')
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

def visualize_top_rated_titles(df, output_dir='plots', top_n=20):
    """
    Create visualization of top-rated titles.
    
    Args:
        df (pd.DataFrame): Merged dataset
        output_dir (str): Directory to save plots
        top_n (int): Number of top titles to show
    """
    print(f"\nCreating Top {top_n} Rated Titles visualization...")
    
    # Filter titles with significant votes (top quartile)
    min_votes = df['numVotes'].quantile(0.75)
    df_qualified = df[df['numVotes'] >= min_votes]
    
    # Get top rated
    top_titles = df_qualified.nlargest(top_n, 'averageRating')
    
    # Create horizontal bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=top_titles['averageRating'],
            y=[truncate_title(title, 40) for title in top_titles['primaryTitle']],
            orientation='h',
            marker_color='gold',
            text=top_titles['averageRating'].round(2),
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Rating: %{x:.2f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title=f'Top {top_n} Rated Titles (with significant votes)',
        xaxis_title='Average Rating',
        yaxis_title='Title',
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    output_file = os.path.join(output_dir, 'top_rated_titles.html')
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

def visualize_decade_comparison(df, output_dir='plots'):
    """
    Create decade-by-decade comparison visualization.
    
    Args:
        df (pd.DataFrame): Merged dataset
        output_dir (str): Directory to save plots
    """
    print("\nCreating Decade Comparison visualization...")
    
    # Create decade column
    df['decade'] = (df['startYear'] // 10) * 10
    
    # Filter for decades with sufficient data
    decade_stats = df.groupby('decade').agg({
        'averageRating': ['mean', 'median'],
        'numVotes': 'mean',
        'tconst': 'count'
    }).reset_index()
    
    decade_stats.columns = ['Decade', 'Mean Rating', 'Median Rating', 'Avg Votes', 'Count']
    decade_stats = decade_stats[decade_stats['Count'] >= 100]
    
    # Create subplot with multiple metrics
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Average Rating by Decade', 'Number of Titles by Decade'),
        vertical_spacing=0.15
    )
    
    # Rating trend
    fig.add_trace(
        go.Bar(x=decade_stats['Decade'], y=decade_stats['Mean Rating'], 
               name='Mean Rating', marker_color='steelblue'),
        row=1, col=1
    )
    
    # Title count
    fig.add_trace(
        go.Bar(x=decade_stats['Decade'], y=decade_stats['Count'],
               name='Number of Titles', marker_color='coral'),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Decade", row=2, col=1)
    fig.update_yaxes(title_text="Average Rating", row=1, col=1)
    fig.update_yaxes(title_text="Number of Titles", row=2, col=1)
    
    fig.update_layout(height=800, showlegend=False, 
                     title_text="Movie Trends by Decade")
    
    output_file = os.path.join(output_dir, 'decade_comparison.html')
    fig.write_html(output_file)
    print(f"Saved: {output_file}")

def main():
    """Main visualization pipeline."""
    
    print("="*60)
    print("IMDb Data Visualization")
    print("="*60)
    
    # Check if data exists
    if not os.path.exists('data/merged_imdb_data.csv'):
        print("\nError: Merged dataset not found!")
        print("Please run data_preprocessing.py first.")
        return
    
    try:
        # Load data
        df = load_data()
        
        # Create output directory
        output_dir = create_output_directory()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        visualize_top_movies_by_director(df, output_dir=output_dir)
        visualize_genre_distribution(df, output_dir)
        visualize_rating_trends(df, output_dir)
        visualize_runtime_vs_rating(df, output_dir)
        visualize_correlation_heatmap(df, output_dir)
        visualize_top_rated_titles(df, output_dir)
        visualize_decade_comparison(df, output_dir)
        
        print("\n" + "="*60)
        print("Visualization completed successfully!")
        print("="*60)
        print(f"\nAll visualizations saved in '{output_dir}/' directory")
        print("\nGenerated files:")
        print("  - top_movies_by_director.html")
        print("  - genre_distribution.html")
        print("  - rating_trends.html")
        print("  - runtime_vs_rating.html")
        print("  - correlation_heatmap.html")
        print("  - top_rated_titles.html")
        print("  - decade_comparison.html")
        print("\nOpen these HTML files in a web browser to view interactive visualizations.")
        
    except Exception as e:
        print(f"\nError during visualization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
