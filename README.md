# IMDb Movie Trends Analysis

## Overview

This repository contains the Phase 1 implementation for **Project 1: Mining IMDb for Movie Trends**. The goal is to establish a reproducible data science workflow that transforms the IMDb non-commercial datasets into analysis-ready assets and documents the research vision for subsequent phases.

Phase 1 deliverables include:

- Curated project structure with version control hygiene
- Automated data acquisition and preprocessing pipelines
- Exploratory data analysis (EDA) assets and visual documentation
- Research questions, methodology blueprint, and Phase 1 report template

## Dataset

All data originates from the [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/). The download script retrieves the following core tables:

- `title.basics.tsv`
- `title.ratings.tsv`
- `title.akas.tsv`
- `title.principals.tsv`
- `title.crew.tsv`
- `name.basics.tsv`

> ?? **License:** Users must agree to the IMDb data usage terms prior to downloading. Access requires registering for an S3 access key via IMDb.

## Repository Layout

```
data/
??? raw/              # IMDb source extracts (.tsv.gz)
??? processed/        # Cleaned, analysis-ready datasets
docs/                 # Research questions, methodology, meeting notes
notebooks/            # Jupyter notebooks for EDA and prototyping
reports/
??? figures/          # Generated plots saved as images
??? phase1/           # Phase 1 written deliverables
src/                  # Python source code (download, preprocessing, utils)
README.md             # Project documentation (this file)
requirements.txt      # Python dependencies
.gitignore            # Git hygiene rules
```

## Getting Started

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Download IMDb datasets (requires configured credentials)
python src/download_data.py

# 4. Run preprocessing pipeline
python src/preprocessing.py

# 5. Launch the exploratory notebook
jupyter notebook notebooks/01_phase1_eda.ipynb
```

## Phase 1 Scope

1. **Work Planning & Division** ? Leadership rotation plan, contribution tracking guidance, and meeting templates.
2. **Problem Understanding** ? Eight innovative research questions aligned with the course rubric and supported by literature references.
3. **Data Preprocessing** ? Configurable ETL pipeline (profiling, cleaning, feature synthesis) with reproducible outputs.
4. **Innovation in Hypotheses** ? Proposal of novel analytical angles extending beyond baseline questions.
5. **Methodology Blueprint** ? Planned analytical techniques, modeling approaches, and visualization roadmap.
6. **Consistency Evidence** ? Git workflow recommendations, commit conventions, and progress tracking artifacts.

## Team Customization Checklist

- [ ] Update `docs/research_questions_methodology.md` with team-specific framing and citations.
- [ ] Replace placeholders in `reports/phase1/phase1_report_template.md` with actual findings.
- [ ] Populate `docs/work_plan.md` with member names, roles, and meeting notes.
- [ ] Regenerate figures by executing `notebooks/01_phase1_eda.ipynb` after running preprocessing.
- [ ] Review and adjust configuration parameters in `src/config.py` to match data availability.

## Contributing

1. Create a feature branch: `git checkout -b feature/<short-description>`
2. Commit changes following [Conventional Commits](https://www.conventionalcommits.org/):

   ```bash
   git commit -m "feat(preprocessing): add outlier detection"
   ```

3. Push and open a pull request for peer review before merging.

## License

This project is released under the MIT License. Refer to `LICENSE` for details. IMDb data is subject to the IMDb Non-Commercial License and **must not** be re-distributed.