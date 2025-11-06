
Project Title
Mining IMDb for Movie Trends


Overview

The global film industry produces thousands of movies each year across genres, languages, and production scales. This project aims to analyze and mine IMDb data to uncover meaningful patterns, correlations, and insights about movie performance. The study combines descriptive analytics, predictive modeling, and interactive visualization to understand the factors that influence a movie’s success.



Objectives

1. Explore and quantify the impact of genre, runtime, budget, director, and region on IMDb ratings.
2. Analyze temporal trends in movie genres and popularity.
3. Identify directors and actors with consistently high-performing films.
4. Investigate correlations between budget, ratings, and gross earnings.
5. Compare critic ratings with audience ratings.
6. Examine whether movie runtime correlates with popularity or rating.
7. Build an interactive dashboard that allows users to select a director and view their top three movies by rating and runtime.


Dataset Information

Source: IMDb Non-Commercial Datasets ([https://developer.imdb.com/non-commercial-datasets/](https://developer.imdb.com/non-commercial-datasets/))

* title.basics.tsv.gz: Contains title type, movie name, genres, release year, and runtime
* title.ratings.tsv.gz: Contains IMDb ratings and vote counts
* title.crew.tsv.gz: Includes directors and writers
* title.principals.tsv.gz: Lists main cast and crew
* title.akas.tsv.gz: Provides regional title variations





Hypotheses

H1: High-budget films tend to receive higher ratings only up to a threshold, after which returns diminish.

H2: Directors with consistent average ratings maintain stronger audience trust.

H3: Longer movies (over 150 minutes) tend to receive higher critic ratings but lower audience popularity.

H4: Regional cinema (for example, India or Korea) has achieved higher global IMDb ratings since 2015.

H5: Multi-genre movies perform better commercially but receive lower critical ratings.



Methodology

1. Data Collection

   * Download IMDb datasets and merge them using unique movie identifiers.

2. Data Preprocessing

   * Handle missing values and duplicates.
   * Convert data types such as release year and runtime.
   * Create new derived attributes such as decade, genre count, and rating per vote.
   * Detect and handle outliers based on runtime or votes.
   * Normalize numeric fields for better comparison.

3. Exploratory Data Analysis

   * Analyze genre and sub-genre evolution over time.
   * Examine top-rated directors and actors.
   * Study correlations between runtime, rating, and budget.
   * Visualize trends through graphs and charts using Matplotlib, Seaborn, and Plotly.

4. Predictive Modeling

   * Apply regression models to predict IMDb rating based on features such as genre, runtime, and budget.
   * Use clustering to group movies with similar performance patterns.
   * Evaluate models using metrics such as RMSE and R-squared.

5. Interactive Visualization

   * Create an interactive dashboard that allows the user to select a director and view the top three movies by rating and runtime.
   * Implemented using Streamlit and Plotly.



Expected Outcomes**

* Insights into changing audience preferences over time.
* Identification of directors and genres with consistent high performance.
* Understanding of how budget influences both ratings and financial success.
* Quantified correlation between runtime, ratings, and votes.
* An interactive analytical dashboard for data exploration.



Technologies and Tools

Programming Language: Python
Data Libraries: Pandas, NumPy
Visualization: Matplotlib, Seaborn, Plotly
Machine Learning: Scikit-learn
Dashboard Development: Streamlit
Version Control: Git, GitHub
Documentation: Jupyter Notebook, Markdown, Google Docs



References

IMDb Non-Commercial Datasets: [https://developer.imdb.com/non-commercial-datasets/](https://developer.imdb.com/non-commercial-datasets/)
Kaggle: IMDb 5000 Movie Dataset
ACM Journal: Predicting Movie Success Using Data Mining Techniques
Plotly and Streamlit Documentation

