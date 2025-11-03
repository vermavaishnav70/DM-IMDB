# DM-IMDB: IMDb Data Mining and Analysis Project

A comprehensive data mining and analysis project to uncover movie industry trends using IMDb datasets.

## Overview

This project analyzes IMDb data to explore:
- Movie industry trends across genres, directors, actors, budgets, and regions
- How various factors influence ratings, popularity, and earnings
- Correlations between runtime, ratings, and regional differences
- Interactive visualizations including top movies by director
- Predictive and descriptive analytics to reveal hidden insights and evolving cinematic patterns

## Dataset

Data source: [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/)

The project uses the following IMDb datasets:
- title.basics.tsv.gz - Contains basic title information
- title.ratings.tsv.gz - Contains IMDb ratings and votes
- title.crew.tsv.gz - Contains director and writer information
- title.principals.tsv.gz - Contains principal cast/crew information
- name.basics.tsv.gz - Contains names and basic information for people

## Project Structure

```
DM-IMDB/
├── data_acquisition.py      # Script to download and prepare IMDb datasets
├── data_preprocessing.py    # Data cleaning and preprocessing
├── exploratory_analysis.py  # Exploratory Data Analysis (EDA)
├── visualization.py         # Interactive visualizations
├── predictive_models.py     # Machine learning models for predictions
├── analysis_notebook.ipynb  # Jupyter notebook with complete analysis
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vermavaishnav70/DM-IMDB.git
cd DM-IMDB
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Acquisition
Download and prepare IMDb datasets:
```bash
python data_acquisition.py
```

### 2. Data Preprocessing
Clean and preprocess the data:
```bash
python data_preprocessing.py
```

### 3. Exploratory Analysis
Run exploratory data analysis:
```bash
python exploratory_analysis.py
```

### 4. Visualizations
Generate interactive visualizations:
```bash
python visualization.py
```

### 5. Predictive Models
Build and evaluate predictive models:
```bash
python predictive_models.py
```

### 6. Interactive Analysis
Explore the Jupyter notebook for interactive analysis:
```bash
jupyter notebook analysis_notebook.ipynb
```

## Key Features

### Data Analysis
- Genre trend analysis over time
- Director and actor performance metrics
- Budget vs earnings correlation studies
- Regional rating differences
- Runtime vs rating relationships

### Visualizations
- Interactive top 3 movies by director
- Genre distribution charts
- Rating and earnings trends
- Regional performance maps
- Correlation heatmaps

### Predictive Analytics
- Movie rating prediction models
- Box office revenue forecasting
- Feature importance analysis
- Genre classification

## Results and Insights

The analysis reveals:
- Key factors influencing movie ratings and earnings
- Genre popularity trends over time
- Regional preferences and differences
- Director and actor impact on movie success
- Optimal movie characteristics for high ratings

## License

This project uses IMDb datasets for non-commercial purposes only, in accordance with IMDb's terms of service.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.