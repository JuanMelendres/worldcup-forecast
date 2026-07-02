# Worldcup Forecast AI

> AI-powered football match prediction using Bayesian statistics, machine learning, explainable AI, and Monte Carlo simulation.

Worldcup Forecast AI is an end-to-end sports analytics project designed to predict international football match outcomes and exact score probabilities. The project combines interpretable statistical models, supervised machine learning, probabilistic modeling, and simulation to estimate match results with uncertainty.

This repository is intended as a portfolio-grade AI/ML engineering project that demonstrates skills in data science, feature engineering, machine learning for count prediction, Bayesian-style probabilistic modeling, sports analytics, model evaluation, calibration, explainable AI, clean software architecture, and MLOps foundations.

> This project is for educational, research, and portfolio purposes only. It is not betting advice.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Why Football Prediction Is Hard](#why-football-prediction-is-hard)
- [Core Features](#core-features)
- [Modeling Approach](#modeling-approach)
- [Project Architecture](#project-architecture)
- [Repository Structure](#repository-structure)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Evaluation Strategy](#evaluation-strategy)
- [Visualizations](#visualizations)
- [Explainability](#explainability)
- [How to Run](#how-to-run)
- [Roadmap](#roadmap)
- [Suggested LinkedIn Post](#suggested-linkedin-post)
- [References](#references)
- [Disclaimer](#disclaimer)

---

## Project Overview

Worldcup Forecast AI predicts football match outcomes by estimating expected goals for each team and converting those estimates into probabilistic scorelines.

Instead of only predicting a winner, the project estimates:

- Expected goals for each team
- Win / draw / loss probabilities
- Exact score probabilities
- Top most likely scorelines
- Score heatmaps
- First-half goal probability
- Monte Carlo simulated outcomes
- Model comparison across statistical and machine learning approaches

The project is designed to evolve from a research notebook into a production-style AI analytics platform.

---

## Problem Statement

Given two national football teams and a match context, estimate:

```text
P(Home Win)
P(Draw)
P(Away Win)
P(Exact Score = h-a)
P(At least one first-half goal)
Expected Goals Home
Expected Goals Away
```

Example:

```text
Argentina vs Austria

Expected goals:
Argentina: 2.10
Austria: 0.85

Outcome probabilities:
Argentina win: 65.2%
Draw: 21.4%
Austria win: 13.4%

Most likely scores:
2-0, 1-0, 2-1, 1-1, 3-0
```

The objective is not to create a deterministic prediction, but to model uncertainty in a transparent and explainable way.

---

## Why Football Prediction Is Hard

Football is a low-scoring and high-variance sport. A stronger team can dominate a match and still lose due to:

- Low goal frequency
- Red cards
- Penalties
- Tactical matchups
- Player injuries
- Lineup rotation
- Weather and altitude
- Travel fatigue
- Tournament pressure
- Random finishing variance

This makes football ideal for probabilistic modeling rather than simple classification.

---

## Core Features

### Current Scope

- Historical international results ingestion
- Data cleaning and team normalization
- Team-level match transformation
- Dynamic Elo rating calculation
- Recent form metrics
- Volatility metrics
- Baseline Poisson model
- XGBoost Poisson regression
- XGBoost Tweedie regression
- Bayesian-style empirical shrinkage
- Poisson-lognormal uncertainty simulation
- Dixon-Coles style low-score adjustment
- Monte Carlo match simulation
- Score probability matrix
- First-half goal probability
- Model comparison charts
- Matchday summary dashboard

### Future Scope

- Full Bayesian hierarchical model with PyMC
- CatBoost and LightGBM comparison
- SHAP explainability
- Calibration curves
- Brier Score and Log Loss tracking
- Backtesting by tournament windows
- API with FastAPI or Spring Boot
- Dashboard with Streamlit or Next.js
- Dockerized deployment
- CI/CD with GitHub Actions

---

## Modeling Approach

The project uses a hybrid modeling strategy:

```text
Historical International Results
        ↓
Data Cleaning
        ↓
Team-Level Dataset
        ↓
Feature Engineering
        ↓
Dynamic Elo Ratings
        ↓
Recent Form + Volatility
        ↓
Statistical Models + ML Models
        ↓
Expected Goals Ensemble
        ↓
Score Probability Matrix
        ↓
Dixon-Coles Style Adjustment
        ↓
Monte Carlo Simulation
        ↓
Final Probabilities + Visualizations
```

The final model can be implemented as an ensemble:

```text
Final Expected Goals =
    w1 × Baseline Poisson
  + w2 × XGBoost Poisson
  + w3 × XGBoost Tweedie
  + w4 × Bayesian-Style Shrinkage
  + w5 × Poisson-Lognormal Simulation
```

The weights can be tuned through validation and backtesting.

---

## Project Architecture

```text
worldcup-forecast-ai/

├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_poisson.ipynb
│   ├── 04_xgboost_poisson.ipynb
│   ├── 05_xgboost_tweedie.ipynb
│   ├── 06_bayesian_model.ipynb
│   ├── 07_poisson_lognormal.ipynb
│   ├── 08_model_comparison.ipynb
│   └── 09_world_cup_predictions.ipynb
│
├── src/
│   └── worldcup-forecast-ai/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── load_data.py
│       │   └── preprocessing.py
│       ├── features/
│       │   ├── elo.py
│       │   ├── recent_form.py
│       │   ├── volatility.py
│       │   └── team_features.py
│       ├── models/
│       │   ├── baseline_poisson.py
│       │   ├── xgboost_poisson.py
│       │   ├── xgboost_tweedie.py
│       │   ├── bayesian_shrinkage.py
│       │   ├── poisson_lognormal.py
│       │   └── ensemble.py
│       ├── simulation/
│       │   ├── monte_carlo.py
│       │   └── score_matrix.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   ├── calibration.py
│       │   └── backtesting.py
│       ├── explainability/
│       │   ├── shap_analysis.py
│       │   └── feature_importance.py
│       └── visualization/
│           ├── plots.py
│           └── dashboards.py
│
├── api/
│   ├── main.py
│   ├── routes/
│   └── schemas/
│
├── app/
│   ├── streamlit_app.py
│   └── components/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│   ├── trained/
│   └── artifacts/
│
├── reports/
│   ├── figures/
│   ├── model_cards/
│   └── experiments/
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_simulation.py
│
├── docs/
│   ├── methodology.md
│   ├── model_assumptions.md
│   ├── limitations.md
│   └── future_work.md
│
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
│
└── README.md
```

---

## Repository Structure

The recommended repository structure separates research, production code, API, dashboard, documentation, and tests.

### `notebooks/`

Research and experimentation notebooks. These should explain the modeling process step by step.

### `src/worldcup-forecast-ai/`

Reusable Python package with clean, testable modules.

### `api/`

Optional FastAPI or Spring Boot service that exposes the prediction engine.

### `app/`

Optional Streamlit or Next.js dashboard for interactive match predictions.

### `reports/`

Generated figures, model cards, backtesting reports, and experiment summaries.

### `docs/`

Technical documentation, methodology, assumptions, limitations, and future work.

### `tests/`

Unit and integration tests for feature engineering, model outputs, and simulation logic.

---

## Dataset

The main dataset used for the initial version is the International Football Results dataset by `martj42/international_results`.

The dataset includes historical international football matches with:

- Date
- Home team
- Away team
- Home score
- Away score
- Tournament
- Country
- Neutral-site flag

Example columns:

```text
date
home_team
away_team
home_score
away_score
tournament
city
country
neutral
```

Future versions can integrate additional external datasets:

- FIFA rankings
- Public Elo ratings
- Team confederation
- Squad market value
- Player injuries
- Expected lineups
- Weather
- Altitude
- Travel distance
- Rest days

---

## Feature Engineering

### Team Strength Features

- Dynamic Elo rating
- Elo difference
- Recent goals scored
- Recent goals conceded
- Recent points per match
- Goals per match
- Goals conceded per match

### Team Volatility Features

- Rolling standard deviation of goals scored
- Rolling standard deviation of goals conceded
- Average absolute goal difference
- Match-to-match scoring volatility

### Match Context Features

- Neutral-site flag
- Home / away indicator
- Tournament
- Year
- Potential tournament stage
- Potential rest days
- Potential travel distance
- Potential weather and altitude

### Bayesian-Style Features

- Shrunk attack strength
- Shrunk defensive weakness
- Prior-weighted team strength
- Global scoring baseline

---

## Models

### 1. Baseline Poisson Model

A simple statistical benchmark.

Assumption:

```text
Goals ~ Poisson(λ)
```

Where λ is estimated using:

```text
team attack strength × opponent defensive weakness × global scoring rate
```

Strengths:

- Simple
- Interpretable
- Good baseline
- Useful for exact score probability matrices

Limitations:

- Assumes mean ≈ variance
- Can be too conservative
- Does not fully capture overdispersion

---

### 2. XGBoost Poisson Regression

A machine learning model designed for count targets.

```python
objective = "count:poisson"
```

Strengths:

- Learns non-linear relationships
- Handles mixed numerical and categorical features
- Works well with engineered football features
- More flexible than a pure Poisson baseline

Limitations:

- Requires careful validation
- Can overfit if features leak future information
- Needs explainability tools for interpretation

---

### 3. XGBoost Tweedie Regression

Tweedie regression is useful when goals are overdispersed.

Poisson assumes:

```text
variance ≈ mean
```

Tweedie allows:

```text
variance = φ × mean^p
```

Example configuration:

```python
objective = "reg:tweedie"
tweedie_variance_power = 1.35
```

Strengths:

- More flexible variance structure
- Useful for overdispersed outcomes
- Good experimental model for goal intensity

Limitations:

- Does not directly produce discrete exact scores
- Needs conversion into a score probability model
- Requires tuning the variance power

---

### 4. Bayesian-Style Shrinkage Model

This model approximates the behavior of a hierarchical Bayesian model.

Teams with fewer recent matches are pulled closer to the global average.

```text
Adjusted Attack =
    weight × observed attack
  + (1 - weight) × global attack prior
```

Strengths:

- Reduces overfitting
- Handles teams with limited data better
- Adds stability to the ensemble

Limitations:

- Not a full MCMC model
- Does not produce posterior distributions directly

---

### 5. Poisson-Lognormal Simulation

A Poisson-lognormal mixture models uncertainty around expected goals.

Instead of assuming λ is fixed:

```text
λ = fixed value
```

We model:

```text
λ ~ LogNormal(log(λ_base), σ)
Goals ~ Poisson(λ)
```

Strengths:

- Adds uncertainty around expected goals
- Allows more explosive scorelines
- Handles overdispersion better than plain Poisson
- Useful for Monte Carlo simulation

Limitations:

- Requires careful choice of σ
- May overstate high-scoring probabilities if not calibrated

---

### 6. Dixon-Coles Style Low-Score Adjustment

Football has many low-scoring outcomes. A Dixon-Coles style adjustment modifies probabilities for:

- 0-0
- 1-0
- 0-1
- 1-1

Strengths:

- Better low-score behavior
- More realistic football score distributions

Limitations:

- Full Dixon-Coles should be estimated by maximum likelihood
- Lightweight correction is an approximation

---

### 7. Final Ensemble

The final model combines multiple approaches:

```text
Baseline Poisson
+ XGBoost Poisson
+ XGBoost Tweedie
+ Bayesian-Style Shrinkage
+ Poisson-Lognormal Simulation
```

The ensemble helps balance:

- Interpretability
- Flexibility
- Stability
- Uncertainty modeling

---

## Evaluation Strategy

A strong sports prediction project should not only produce predictions. It should evaluate whether the probabilities are meaningful.

### Regression Metrics for Goals

- MAE
- RMSE
- Poisson Deviance
- Tweedie Deviance

### Classification Metrics for Outcomes

- Accuracy
- Log Loss
- Brier Score
- Ranked Probability Score

### Calibration Metrics

- Reliability curve
- Calibration curve
- Expected calibration error
- Probability bin analysis

### Backtesting

Use time-aware splits:

```text
Train: historical matches up to year N
Validate: matches from year N+1
Test: future tournament window
```

Avoid random splits when evaluating sports forecasting models because random splits may leak temporal information.

---

## Visualizations

The project should include:

### Match-Level Visualizations

- Expected goals comparison
- Win / draw / loss probability chart
- Top exact score chart
- Exact score heatmap
- First-half goal probability chart
- Monte Carlo score distribution

### Model-Level Visualizations

- Feature importance
- SHAP values
- Validation metric comparison
- Calibration curves
- Backtesting performance over time

### Dashboard-Level Visualizations

- Match selector
- Probability cards
- Score heatmap
- Model comparison
- Explainability panel
- Prediction history

---

## Explainability

A professional AI project should explain why the model predicts a result.

Recommended explainability tools:

- SHAP values
- Feature importance
- Sensitivity analysis
- Scenario simulation

Example questions:

```text
Why is Team A favored?
What happens if Team B has lower recent form?
How much does Elo difference affect the prediction?
How sensitive is the model to first-half scoring assumptions?
```

Example explainability outputs:

- Elo difference contribution
- Recent goals contribution
- Defensive weakness contribution
- Tournament effect
- Volatility effect

---

## How to Run

### Option 1: Google Colab

Open the main notebook:

```text
notebooks/09_world_cup_predictions.ipynb
```

Run all cells.

### Option 2: Local Environment

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks:

```bash
jupyter lab
```

### Option 3: Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Option 4: API

```bash
uvicorn api.main:app --reload
```

---

## Example Usage

```python
from worldcup-forecast-ai.models.ensemble import predict_match

prediction = predict_match(
    home_team="Argentina",
    away_team="Austria",
    neutral=True,
    tournament="FIFA World Cup",
    match_date="2026-06-23"
)

print(prediction["home_win_probability"])
print(prediction["draw_probability"])
print(prediction["away_win_probability"])
print(prediction["top_scores"])
```

Expected output:

```text
Argentina win probability: 62.4%
Draw probability: 22.1%
Austria win probability: 15.5%

Most likely scores:
2-0
1-0
2-1
1-1
3-0
```

---

## Example Dashboard

The final dashboard should allow users to select two teams and generate:

```text
Team A vs Team B

Expected Goals:
Team A: 1.85
Team B: 0.95

Outcome Probabilities:
Team A win: 58.2%
Draw: 24.0%
Team B win: 17.8%

Top Scores:
1-0
2-0
1-1
2-1
0-0

First-Half Goal:
At least one first-half goal: 54.3%
```

---

## Roadmap

### Phase 1 — Research Notebook

- [x] Load historical results
- [x] Clean dataset
- [x] Build team-level dataset
- [x] Add Poisson baseline
- [x] Add score heatmap
- [x] Add first-half goal probability

### Phase 2 — Machine Learning

- [x] Add Dynamic Elo
- [x] Add recent form features
- [x] Add XGBoost Poisson regression
- [x] Add XGBoost Tweedie regression
- [x] Add volatility features

### Phase 3 — Probabilistic Modeling

- [x] Add Bayesian-style shrinkage
- [x] Add Poisson-lognormal simulation
- [x] Add Dixon-Coles style correction
- [ ] Add full Bayesian hierarchical model with PyMC
- [ ] Add posterior uncertainty intervals

### Phase 4 — Model Evaluation

- [ ] Add time-aware backtesting
- [ ] Add Brier Score
- [ ] Add Log Loss
- [ ] Add calibration curves
- [ ] Add model comparison report

### Phase 5 — Explainability

- [ ] Add SHAP values
- [ ] Add feature importance reports
- [ ] Add scenario simulation
- [ ] Add sensitivity analysis

### Phase 6 — Productization

- [ ] Convert notebooks into reusable Python package
- [ ] Add unit tests
- [ ] Add CI/CD with GitHub Actions
- [ ] Add FastAPI service
- [ ] Add Streamlit or Next.js dashboard
- [ ] Add Docker support
- [ ] Add model cards

---

## Suggested LinkedIn Post

```text
I built Worldcup Forecast AI, an end-to-end football match prediction project that combines statistical modeling, machine learning, Bayesian-style inference, and Monte Carlo simulation.

The project estimates:
- Expected goals
- Win/draw/loss probabilities
- Exact score probabilities
- First-half goal probability
- Score heatmaps
- Model comparison across Poisson, XGBoost, Tweedie and Bayesian-style models

The goal was not only to predict match outcomes, but to build a transparent and explainable AI system for sports analytics.

Tech stack:
Python, Pandas, NumPy, Scikit-learn, XGBoost, SciPy, Matplotlib, Jupyter, Streamlit/FastAPI.

This project helped me explore how AI can be applied to uncertain real-world systems where probability, interpretability and model validation matter as much as raw prediction accuracy.
```

---

## Technical Skills Demonstrated

This project demonstrates:

- Python programming
- Data cleaning
- Feature engineering
- Statistical modeling
- Machine learning
- Count regression
- Probabilistic modeling
- Monte Carlo simulation
- Model evaluation
- Data visualization
- Explainable AI
- Software architecture
- MLOps foundations
- API design
- Dashboard design

---

## References

Useful references and concepts:

- Poisson regression for football score modeling
- Dixon-Coles football score model
- Elo rating systems
- XGBoost count regression
- Tweedie regression
- Bayesian hierarchical modeling
- Monte Carlo simulation
- Model calibration
- SHAP explainability

Dataset:

```text
https://github.com/martj42/international_results
```

---

## Disclaimer

This project is for educational and portfolio purposes only.

Football matches are inherently uncertain. The predictions produced by this project should not be interpreted as guaranteed outcomes or betting recommendations.

The model does not currently include all real-world variables that influence football matches, such as confirmed lineups, injuries, tactical decisions, weather, referee behavior, or in-game events.

---

## License

MIT License. See `LICENSE` for more information.

---

## Author

Built as a portfolio project to demonstrate applied AI, machine learning engineering, probabilistic modeling, and sports analytics.
