# House-Pricing

# House Prices — Advanced Regression Techniques
A end-to-end machine learning project for predicting residential home sale prices in Ames, Iowa. Built on the classic Kaggle House Prices competition dataset, the project walks through every stage of the data science workflow: exploratory analysis, preprocessing pipelines, feature engineering, model tuning, and ensemble stacking.

## Dataset Overview
The dataset describes the sale of individual residential properties in Ames, Iowa between 2006 and 2010. It contains 79 explanatory variables covering nearly every aspect of a home.
Split      Rows      Columns  
Train      1,460     81 (incl. SalePrice)
Test       1,459     80

Target Variable — SalePrice
Statistic        Value
Mean             $180,921
Median           $163,000
Std Dev          $79,443
Min              $34,900
Max              $755,000
The target is right-skewed and is log-transformed prior to modelling to better satisfy regression assumptions.

## Exploratory Data Analysis
The notebook answers seven key analytical questions using interactive Plotly visualisations:

Distribution of dwelling types and their relationship to sale prices
Zoning classification impact on average sale price
Street and alley access types effect on pricing
Property shape and land contour vs. average sale price
Property age (year sold − year built) correlation with sale price
Above-grade living area (GrLivArea) correlation with sale price
Year-over-year price trends via box plots and average overlays

Missing value analysis is also performed, with scrollable HTML tables used to inspect null counts and their percentages across all features.

## Preprocessing Pipeline
A reproducible sklearn pipeline is built with ColumnTransformer to ensure consistent transformations are applied to both train and test sets:

Numerical features → SimpleImputer (mean strategy) → StandardScaler
Categorical features → SimpleImputer (most frequent strategy) → OneHotEncoder


## Feature Engineering
A custom FunctionTransformer creates the following derived features before the preprocessing pipeline:
New Feature                  Description
PropertyAge                  YrSold − YearBuilt
TotalSF                      TotalBsmtSF + 1stFlrSF + 2ndFlrSF
TotalBath                    Full + half baths above and below grade (weighted)
HasRemodeled                 Binary flag — remodel date differs from build date
Has2ndFloor                  Binary flag — second floor square footage > 0
HasGarage                    Binary flag — garage area > 0
YrSold_cat                   Year sold cast as categorical
MoSold_cat                   Month sold cast as categorical
YearBuilt_cat                Year built cast as categorical
MSSubClass_cat               Dwelling type code cast as categorical

## Models Trained
All models are tuned using GridSearchCV with KFold cross-validation and evaluated with RMSE on a held-out 20% validation split.
Model                                              Notes
Linear Regression                                  Baseline
Random Forest (RandomForestRegressor)              Ensemble of decision trees
XGBoost (XGBRegressor)                             Best standalone model
MLP Neural Network (MLPRegressor)                  Tuned over architecture, activation, solver, regularisation

Models are trained in two experimental tracks:
Baseline — raw preprocessed features
PCA — preprocessed features reduced to components explaining 95% of variance
Feature Engineering (FE) — enriched features (submitted)


## Ensemble Methods
Two ensemble strategies are applied on top of the individually tuned models:
1. Simple Average Ensemble
Predictions from Random Forest, XGBoost, and MLP are averaged:
pythony_avg = (y_rf + y_xgboost + y_mlp) / 3
2. Stacking (StackingRegressor)
The three base models feed into a meta-learner. The meta-model is itself tuned via GridSearchCV, with candidates including MLP, Linear Regression, and XGBoost. The best meta-model is selected automatically and used for the final submission.

## Getting Started
Prerequisites
bashpip install pandas numpy scikit-learn xgboost plotly scipy jupyter
Run the Notebook
bashjupyter notebook Untitled.ipynb
Make sure train.csv, test.csv, and data_description.txt are in the same directory as the notebook.

## Tech Stack
Library                    Purpose
pandas / numpy             Data manipulation
plotly                     Interactive visualisations
scipy                      Statistical analysis & distribution fitting
scikit-learn               Pipelines, preprocessing, models, tuning
xgboost                    Gradient boosting regressor

## Data Source  
Kaggle — House Prices: Advanced Regression Techniques
