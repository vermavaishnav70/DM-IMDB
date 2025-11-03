# Phase 1 Execution Checklist

Use this checklist to track progress during the first evaluation phase. Update statuses daily and link to evidence (commit hashes, pull requests, screenshots).

## Day 1 - Repository and Data Acquisition

- [ ] Create GitHub repository and invite all collaborators.
- [ ] Merge `setup_project` branch establishing project structure.
- [ ] Configure virtual environment and install `requirements.txt`.
- [ ] Run `python src/download_data.py all` to begin downloading IMDb datasets.
- [ ] Capture meeting notes for the kick-off meeting in `docs/meeting_notes/`.

## Day 2 - Preprocessing Baseline

- [ ] Review downloaded files and validate checksums.
- [ ] Execute `python src/preprocessing.py run --sample-frac 0.1` for a smoke test.
- [ ] Inspect `data/processed/preprocessing_stats.json` and log anomalies.
- [ ] Update `docs/work_plan.md` with actual team assignments and risks.
- [ ] Draft literature review notes for at least three sources.

## Day 3 - Exploratory Analysis and Hypotheses

- [ ] Run `notebooks/01_phase1_eda.ipynb` end-to-end on sampled data.
- [ ] Regenerate plots into `reports/figures/` and document captions.
- [ ] Finalise the eight research questions in `docs/research_questions_methodology.md`.
- [ ] Align statistical tests with hypotheses and record in the methodology section.
- [ ] Prepare questions for mentor/TA feedback session.

## Day 4 - Reporting and Documentation

- [ ] Populate `reports/phase1/phase1_report_template.md` with preliminary findings.
- [ ] Add contribution evidence (commit table, meeting attendance) to the report appendix.
- [ ] Update README with current project status and major learnings.
- [ ] Ensure all notebooks are cleared of extraneous outputs before committing.
- [ ] Schedule dry run of the Phase 1 presentation narrative.

## Day 5 - Quality Assurance and Submission

- [ ] Re-run preprocessing and EDA on full dataset (if feasible) and archive outputs.
- [ ] Validate that `.gitignore` prevents large raw data files from being committed.
- [ ] Generate git contribution stats (e.g., `git shortlog -sn`) and capture screenshot.
- [ ] Complete final proofreading of the Phase 1 report and export to PDF (if required).
- [ ] Tag Phase 1 release (`git tag phase-1-v1 && git push origin phase-1-v1`).

## Optional Stretch Items

- [ ] Build an interactive prototype (Plotly/Dash/Streamlit) for the director explorer.
- [ ] Automate dataset downloads with GitHub Actions workflow (limited to metadata, no raw files).
- [ ] Schedule pair-programming rotations for Phase 2 backlog items.

Remember to update this checklist after each work session and mirror progress in your shared project management tool.
