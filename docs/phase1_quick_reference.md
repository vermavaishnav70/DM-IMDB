# Phase 1 Quick Reference Card

Keep this one-pager handy for daily execution. Print or pin it in your workspace.

## Core Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Data acquisition
python src/download_data.py list
python src/download_data.py all

# Preprocessing
python src/preprocessing.py run --sample-frac 0.1  # smoke test
python src/preprocessing.py run                     # full run

# Exploratory notebook
jupyter notebook notebooks/01_phase1_eda.ipynb

# Git workflow
git checkout -b feature/<task>
git add <files>
git commit -m "feat: concise description"
git push origin feature/<task>
```

## Daily Non-Negotiables

- Update progress in shared tracker or stand-up channel.
- Commit code with descriptive messages (minimum one per contributor per day).
- Capture learnings or blockers in meeting notes.
- Review teammate pull requests before end of day.

## Key Artefacts

| Artefact | Location | Owner | Status |
|----------|----------|-------|--------|
| Processed dataset | `data/processed/movies_processed.csv` | Data engineer | Pending |
| Preprocessing stats | `data/processed/preprocessing_stats.json` | Data engineer | Pending |
| EDA notebook | `notebooks/01_phase1_eda.ipynb` | Analyst | Pending |
| Research plan | `docs/research_questions_methodology.md` | Research lead | Pending |
| Phase 1 report | `reports/phase1/phase1_report_template.md` | Communications lead | Pending |

Update owners and statuses at the start of each day.

## Rubric Focus (Phase 1)

- **Work planning (15%)** - leadership rotation, documented stand-ups, contribution evidence.
- **Problem understanding (20%)** - innovative questions, literature-backed rationale.
- **Data preprocessing (10%)** - reproducible pipeline, documented thresholds.
- **Innovation (20%)** - hypotheses that extend beyond provided prompts.
- **Methodology (20%)** - analysis plan linked to research questions.
- **Consistency (15%)** - regular commits, meeting notes, on-time submission.

## Quality Gates Before Submission

- [ ] All scripts and notebooks run without manual intervention.
- [ ] No large raw files tracked by git (`git status` clean).
- [ ] Figures regenerated and saved in `reports/figures/`.
- [ ] README and report reflect latest progress.
- [ ] Git history shows balanced contributions across team members.

## Rapid Troubleshooting

- **Import errors:** Confirm virtual environment is active, re-run `pip install -r requirements.txt`.
- **Missing files:** Check working directory (`pwd`), verify `.gitignore` entries, re-run download/preprocess scripts.
- **Notebook kernel crash:** Use sample fraction (`--sample-frac 0.05`) or subset rows for testing.
- **Git conflicts:** `git pull --rebase origin main`, resolve conflicts, `git add`, `git rebase --continue`.

## Communication Escalation Ladder

1. Post question in team chat with context and error logs.
2. Pair with teammate for 15-minute debugging session.
3. Document issue in meeting notes and tag responsible owner.
4. Escalate to mentor or teaching assistant with summary, reproduction steps, and attempted fixes.

Stay disciplined, keep artefacts up to date, and log every major decision to maximise Phase 1 scores.
