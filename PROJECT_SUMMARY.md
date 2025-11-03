# IMDb Data Mining and Analysis - Project Summary

## Overview
This project provides a comprehensive data mining and analysis solution for IMDb datasets to uncover movie industry trends, patterns, and insights.

## Architecture

### Data Pipeline
```
IMDb Datasets → data_acquisition.py → Raw TSV Files
                                      ↓
                              data_preprocessing.py → merged_imdb_data.csv
                                      ↓
                    ┌─────────────────┴─────────────────┐
                    ↓                                   ↓
         exploratory_analysis.py              visualization.py
                    ↓                                   ↓
            Console Reports                      Interactive HTML
                                      ↓
                              predictive_models.py
                                      ↓
                              ML Models & Metrics
```

### Key Components

#### 1. Data Acquisition (`data_acquisition.py`)
- Downloads 5 IMDb datasets from official source
- Handles gzip extraction automatically
- Progress tracking for large downloads
- Datasets: title.basics, title.ratings, title.crew, title.principals, name.basics

#### 2. Data Preprocessing (`data_preprocessing.py`)
- Cleans and validates IMDb data
- Handles missing values (IMDb uses `\N`)
- Filters movies post-1900 with valid runtime
- Creates merged dataset combining titles, ratings, and crew
- Calculates weighted ratings using IMDb's formula

#### 3. Exploratory Analysis (`exploratory_analysis.py`)
- **Genre Analysis**: Distribution, trends over time, rating by genre
- **Director Performance**: Statistics for directors with 3+ titles
- **Runtime Correlations**: Relationship with ratings and categories
- **Popularity Factors**: Vote distribution and rating patterns
- **Comprehensive Reports**: Key insights and statistics

#### 4. Interactive Visualizations (`visualization.py`)
- **Top Movies by Director**: Interactive bar chart (top 3 movies for top 10 directors)
- **Genre Distribution**: Bar chart of top 15 genres
- **Rating Trends**: Line chart showing ratings over decades
- **Runtime vs Rating**: Scatter plot with vote-based coloring
- **Correlation Heatmap**: Key metrics relationships
- **Top Rated Titles**: Horizontal bar chart with qualified titles
- **Decade Comparison**: Multi-panel decade-by-decade analysis
- All visualizations use Plotly for full interactivity

#### 5. Predictive Models (`predictive_models.py`)
- **Rating Prediction**: Linear Regression, Random Forest, Gradient Boosting
- **Popularity Prediction**: Random Forest with log transformation
- **Feature Importance**: Analysis of key predictive factors
- **Model Evaluation**: RMSE, MAE, R² metrics
- Features: runtime, year, decade, genres, title type

#### 6. Interactive Notebook (`analysis_notebook.ipynb`)
- Complete analysis workflow in Jupyter
- Data loading and overview
- Exploratory visualizations
- Statistical analysis
- Predictive modeling
- Key insights summary

#### 7. Utility Functions (`utils.py`)
- Shared helper functions across modules
- Runtime categorization
- Title truncation
- Weighted rating calculation
- Genre and director explosion
- Name mapping utilities

## Key Insights Discovered

### Genre Patterns
- Drama, Documentary, and Comedy are most common genres
- Genre significantly impacts both ratings and popularity
- Genre trends have evolved over decades

### Director Influence
- Directors with 5+ titles show consistent quality patterns
- Top directors maintain high average ratings across works
- Director reputation correlates with movie success

### Temporal Trends
- Movie production has increased significantly post-2000
- Rating patterns remain relatively stable across decades
- Modern movies have different popularity distributions

### Runtime Analysis
- Average runtime: ~90-100 minutes for most titles
- Sweet spot for high ratings: 90-150 minutes
- Very short or very long movies have specific audience niches

### Predictive Insights
- Movie ratings can be predicted with moderate accuracy (R² ~0.3-0.5)
- Key predictive features: year, genre, runtime, title type
- Popularity (votes) is harder to predict but shows patterns

## Usage Instructions

### Installation
```bash
# Clone repository
git clone https://github.com/vermavaishnav70/DM-IMDB.git
cd DM-IMDB

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline
```bash
# Step 1: Download data (large files, ~2GB total)
python data_acquisition.py

# Step 2: Preprocess and merge data
python data_preprocessing.py

# Step 3: Run exploratory analysis
python exploratory_analysis.py

# Step 4: Generate interactive visualizations
python visualization.py

# Step 5: Build predictive models
python predictive_models.py

# Step 6: Interactive exploration
jupyter notebook analysis_notebook.ipynb
```

## Output Files

### Data Files (in `data/` directory, gitignored)
- `title_basics.tsv` - Title information
- `title_ratings.tsv` - Ratings and votes
- `title_crew.tsv` - Director and writer info
- `title_principals.tsv` - Cast information
- `name_basics.tsv` - People names and info
- `merged_imdb_data.csv` - Preprocessed merged dataset

### Visualization Files (in `plots/` directory, gitignored)
- `top_movies_by_director.html` - Interactive director analysis
- `genre_distribution.html` - Genre breakdown
- `rating_trends.html` - Temporal rating patterns
- `runtime_vs_rating.html` - Runtime-rating scatter
- `correlation_heatmap.html` - Metric correlations
- `top_rated_titles.html` - Highest rated movies
- `decade_comparison.html` - Decade-by-decade trends

## Technical Details

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Static visualizations
- **seaborn**: Statistical visualizations
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning models
- **requests**: HTTP library for downloads
- **jupyter**: Interactive notebooks

### Data Quality
- Filters applied: movies post-1900, valid runtime, minimum 100 votes
- Missing value handling: IMDb's `\N` properly converted to NaN
- Data validation: Type conversions with error handling
- Outlier treatment: Quantile-based filtering for reliability

### Performance Considerations
- Large dataset handling with chunked processing where needed
- Sampling for visualization performance (e.g., 5000 sample scatter)
- Efficient pandas operations with vectorization
- Multi-core support in Random Forest models (`n_jobs=-1`)

## Future Enhancements

Potential improvements for this project:
1. Add budget and box office data integration
2. Implement more advanced NLP for plot analysis
3. Add network analysis for actor collaborations
4. Include sentiment analysis of reviews
5. Deploy as web application with Flask/Dash
6. Add real-time data updates
7. Implement deep learning models for prediction
8. Add regional/language-specific analysis

## License
This project uses IMDb datasets for non-commercial purposes only, in accordance with IMDb's terms of service.

## Contact
For questions or contributions, please open an issue on GitHub.
