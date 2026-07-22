# Tennis Match Outcome Classification

## Machine Learning Web Application

This academic machine learning project compares multiple classification
algorithms to classify professional women's tennis match outcomes using
player and match statistics.

The project includes exploratory data analysis, model training,
performance evaluation, serialized models, and a web application for
visualizing player statistics and generating classification results.

## Project Overview

The objective of this project was to study the relationship between
tennis performance statistics and match outcomes.

The dataset contains 127 professional women's tennis matches. Each
record describes a match between two players and includes the final
result and several performance indicators.

The target variable is:

```text
Result
```

Where:

- `1`: Player 1 won the match.
- `0`: Player 1 lost the match.

This is therefore a supervised binary classification problem.

## Project Objectives

- Explore professional tennis match data.
- Analyze player performance statistics.
- Prepare data for binary classification.
- Train different machine learning models.
- Compare model performance using multiple metrics.
- Select the model with the best classification performance.
- Develop a web interface for displaying results and predictions.

## Dataset

The project uses a public academic dataset obtained from Kaggle.

The dataset contains professional women's tennis match information,
including player names, match results, tournament rounds, and performance
statistics.

### Main Variables

- `FSP`: First-service percentage.
- `FSW`: Points won on the first service.
- `SSP`: Second-service points played.
- `SSW`: Points won on the second service.
- `ACE`: Number of aces.
- `DBF`: Number of double faults.
- `WNR`: Number of winners.
- `UFE`: Number of unforced errors.
- `BPC`: Break-point opportunities.
- `BPW`: Break points won.
- `NPA`: Net-point attempts.
- `NPW`: Net points won.
- `TPW`: Total points won.
- `Result`: Binary match outcome.

### Data Source

- Platform: Kaggle
- Dataset: [Add exact dataset name]
- Source: [Add Kaggle dataset link]
- License: [Add dataset license]

The dataset is included for academic and educational purposes according
to its original license.

## Exploratory Data Analysis

The exploratory analysis examined player performance and the
distribution of the main tennis statistics.

The analysis included:

- Players with the most match victories.
- Players with the most break points won.
- Players with the fewest double faults.
- Players with the fewest unforced errors.
- Players with the highest average number of aces.
- Players with the highest average number of winners.

Aggregated player statistics were also created to support the web
application.

## Machine Learning Models

Three supervised classification models were evaluated.

### Logistic Regression

Logistic Regression was used as an interpretable baseline model for
estimating the probability of a binary match result.

### Support Vector Machine

A Support Vector Machine with a Radial Basis Function kernel was used
to model potential nonlinear relationships between match statistics
and outcomes.

### Random Forest

Random Forest was used as an ensemble model capable of capturing
nonlinear relationships and interactions between variables.

## Model Evaluation

The models were compared using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC

| Model | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 88.46% | 90.00% | 81.82% | 85.71% | 93.94% |
| SVM (RBF) | 80.77% | 87.50% | 63.64% | 73.68% | 87.27% |
| Random Forest | 84.62% | 81.82% | 81.82% | 81.82% | 88.79% |

## Main Result

Logistic Regression achieved the strongest overall performance:

- Accuracy: **88.46%**
- Precision: **90.00%**
- Recall: **81.82%**
- F1-score: **85.71%**
- ROC AUC: **93.94%**

For this dataset and evaluation split, Logistic Regression provided the
best balance between classification accuracy, recall, and class
discrimination.

These results should be interpreted cautiously because the dataset
contains only 127 matches.

## Web Application

The project includes a Python web application with the following
sections:

- Home page.
- Individual player statistics.
- Player rankings.
- Match outcome classification.
- Comparison of classification models.

The application loads the trained models and the preprocessing scaler
from serialized `.pkl` files.

## Tools and Technologies

- Python
- Pandas
- Scikit-learn
- Flask
- Microsoft Excel
- HTML
- Logistic Regression
- Support Vector Machine
- Random Forest

## Repository Structure

```text
tennis-match-prediction/
├── app/
│   ├── templates/
│   │   ├── base.html
│   │   ├── individual.html
│   │   ├── inicio.html
│   │   ├── prediccion.html
│   │   └── ranking.html
│   ├── README.md
│   └── modelo_aplicado.py
├── data/
│   ├── player_statistics_sample.xlsx
│   ├── tennis_matches_sample.csv
│   └── README.md
├── images/
│   └── README.md
├── models/
│   ├── modelo_log.pkl
│   ├── modelo_rf.pkl
│   ├── modelo_svm.pkl
│   ├── scaler.pkl
│   └── README.md
└── README.md
```

## Model Files

The `models` directory contains the serialized objects used by the
application:

- `modelo_log.pkl`: trained Logistic Regression model.
- `modelo_rf.pkl`: trained Random Forest model.
- `modelo_svm.pkl`: trained SVM model.
- `scaler.pkl`: fitted feature scaler.

> Pickle files should only be loaded from trusted sources.

## Limitations

- The dataset contains only 127 matches, so the reported metrics may
  vary significantly with a different train-test split.
- Some variables describe events that occur during or after a match.
  Consequently, the project should be interpreted primarily as match
  outcome classification rather than a strictly pre-match prediction
  system.
- Variables such as total points and final sets may contain direct or
  indirect information about the final result.
- The model has not been validated using a larger independent dataset.
- Historical classification performance does not guarantee equivalent
  performance on future matches.

## Possible Improvements

- Use a larger dataset covering multiple tournaments and seasons.
- Use only information available before each match.
- Remove variables that may generate target leakage.
- Split training and testing data chronologically.
- Apply cross-validation and hyperparameter optimization.
- Compare model calibration in addition to classification metrics.
- Add player ranking, playing surface, recent form, and head-to-head
  history.
- Improve the reproducibility of the model-training process.

## Academic Context

This repository presents an academic machine learning project developed
for educational purposes.

## Disclaimer

This project is intended for educational and portfolio purposes. The
application demonstrates machine learning classification techniques and
should not be interpreted as a guaranteed prediction system.
