# Research Questions and Methodology Blueprint

This document articulates innovative research questions, supporting rationale, and the methodology plan for Phase 1 of the IMDb Movie Trends project.

## Analytical Philosophy

1. **Holistic storytelling:** Blend descriptive, diagnostic, and predictive analyses to uncover not only what happened but why it matters.
2. **Evidence-first iteration:** Start with reproducible EDA, iterate hypotheses using statistical testing, and progress to modelling only where signals emerge.
3. **Reproducibility:** Every figure or metric should be traceable to a notebook cell, script, or logged output.
4. **Audience empathy:** Prioritise insights that resonate with stakeholders (students, instructors, industry observers) and feed Phase 3 presentation narratives.

## Priority Research Questions

| ID | Research Question | Motivation | Key Variables | Proposed Techniques |
|----|-------------------|------------|---------------|---------------------|
| RQ1 | Are modern audiences harder to please compared with earlier decades? | Examine rating variance over time to test the "evolution paradox" hypothesis. | `averageRating`, `decade`, `era` | Variance analysis, Levene's test, line charts with confidence bands |
| RQ2 | Which director-actor partnerships consistently deliver high-rated films? | Identify collaborative synergies valuable for studios and streaming platforms. | `director_ids`, `top_cast_ids`, `averageRating`, `success_score` | Network centrality, pivoted heat maps, partnership leaderboards |
| RQ3 | Is there a "Goldilocks" runtime that maximises audience satisfaction for top genres? | Investigate runtime optimisation for programming and scheduling decisions. | `runtimeMinutes`, `primary_genre`, `success_score` | Spline smoothing, runtime bins, bootstrap CI |
| RQ4 | Does genre diversity (genre blending) correlate with film success? | Test whether multi-genre storytelling broadens or dilutes appeal. | `num_genres`, `averageRating`, `numVotes` | ANOVA / Kruskal-Wallis, box plots, effect size estimation |
| RQ5 | Which hidden gems (high rating, low visibility) deserve re-discovery campaigns? | Spotlight catalogue opportunities for streamers and educators. | `averageRating`, `numVotes`, `startYear`, `primary_genre` | Quantile filtering, z-score anomaly detection, scatter plots |
| RQ6 | Can success be predicted without budget or marketing data? | Build a "budget-free" model using publicly available IMDb signals. | `runtimeMinutes`, `num_genres`, `startYear`, `primary_genre`, `popularity_score` | Random Forest/Gradient Boosting regressors, SHAP feature importance |
| RQ7 | Do regional releases achieve cross-cultural resonance? | Understand localisation vs. universal appeal trade-offs. | `title.akas` join (regions), `averageRating`, `numVotes` | Geographic aggregation, small multiples, chi-square tests |
| RQ8 | What trajectory do directors follow from debut to breakout success? | Explore career development timelines for mentorship and scouting insights. | `director_ids`, `startYear`, `averageRating`, `numVotes` | Cohort analysis, survival-style curves, mixed-effects regression |

> Update the question phrasing or add domain-specific angles that resonate with your mentor. Cite supporting literature for each hypothesis in the Phase 1 report.

## Data Sources and Feature Plan

- Core tables: `title.basics`, `title.ratings`, `title.crew`, `title.principals`, `name.basics`, `title.akas` (optional for Phase 1, required for RQ7).
- Derived features: `decade`, `era`, `runtime_category`, `popularity_score`, `success_score`, `genre_list`, collaborative pair features.
- External context (optional): Box Office Mojo aggregates, TMDb metadata, streaming availability indices. Document any external integrations clearly.

## EDA Pipeline Overview

1. **Data validation:** Row counts, null analysis, schema checks (captured in preprocessing stats JSON).
2. **Univariate analysis:** Distribution plots for ratings, runtime, votes, genres.
3. **Temporal lens:** Decadal trends, era comparison charts, movie production volume.
4. **Cross-sectional insights:** Genre vs. rating, runtime vs. rating, popularity vs. rating.
5. **Collaboration analysis:** Director and cast aggregation, partnership matrices, centrality metrics (Phase 2 deep dive).

## Statistical Techniques

| Question | Statistical Test | Purpose | Notes |
|----------|-----------------|---------|-------|
| RQ1 | Levene's test / Brown-Forsythe | Compare rating variance across decades | Validate assumptions before ANOVA |
| RQ3 | LOESS smoothing / polynomial regression | Model runtime vs. success relationships | Evaluate by genre clusters |
| RQ4 | One-way ANOVA (fallback Kruskal-Wallis) | Assess genre-count impact on ratings | Report eta-squared effect size |
| RQ5 | Isolation Forest / z-score filter | Flag hidden gems | Prioritise interpretable thresholds |
| RQ6 | Random Forest Regressor | Predict ratings using IMDb metadata | Use train/validation split, report R^2, RMSE, MAE |
| RQ7 | Chi-square / ANOVA | Compare rating distributions across regions | Requires `title.akas` integration |
| RQ8 | Mixed-effects or GAM | Model director rating trajectory | Explore in Phase 2+ |

## Modelling Roadmap

1. **Baseline (Phase 1)**
   - Random Forest model predicting `averageRating` from metadata (demonstrates feasibility).
   - Feature importance and partial dependence plots to surface drivers.

2. **Phase 2 Enhancements**
   - Gradient boosting (XGBoost/LightGBM) with hyperparameter tuning.
   - Classification framing (e.g., top quartile vs. rest) for business-friendly insights.
   - Calibration checks and cross-validation.

3. **Phase 3 Aspirations**
   - Ensemble models, fairness analysis, scenario simulations.
   - Integrate external signals (awards, streaming availability) if time permits.

## Visualisation Strategy

- **Static visuals:** Matplotlib/Seaborn for baseline distributions, box plots, trend lines.
- **Interactive visuals:** Plotly Express and Dash (Phase 2/3) for director explorer, regional maps.
- **Design principles:** Consistent colour palette, accessible annotations, descriptive captions stored in `reports/figures/metadata.json` (optional).

## Literature and Benchmarking

Record at least five references supporting hypotheses or methods. Suggested starting points:

1. IMDb rating system methodology notes.
2. Research on genre hybridisation and audience reception.
3. Studies examining runtime and narrative pacing.
4. Collaboration networks in cinema (e.g., Wasserman & Faust, social network analysis texts).
5. Predictive modelling case studies for movie success.

Add citations in the final report using APA or IEEE style, and include BibTeX entries if you maintain a `references.bib` file.

## Deliverables Checklist

- [ ] Eight research questions validated by literature references.
- [ ] Documented statistical and modelling plan with justification.
- [ ] Figure storyboard with chart types, purpose, and responsible owner.
- [ ] Risks and assumptions logged (link back to `docs/work_plan.md`).
- [ ] Gantt-style timeline aligning research tasks with rubric milestones.

Update this file as the team iterates on hypotheses and share it with mentors for feedback before Phase 2 execution.
