# News Article Share Prediction

## Project Description

This project implements a machine learning pipeline to predict the number of social media shares a news article receives, using the article metadata, content statistics and keyword share history in `data/mini_project_1_data.csv`. The pipeline covers data ingestion, data cleaning, exploratory analysis, feature scaling and encoding, training of three regression models, hyperparameter tuning, and a final evaluation on a held-out test set.

The project is split into two deliverables:

* `eda.ipynb` — the exploratory data analysis notebook, including univariate and bivariate analysis, correlation analysis, model training and comparison, coefficient interpretation and the final model selection.
* `main.py` with `src/` — the configurable pipeline that reproduces the same cleaning, split, training and evaluation from the command line.

---

## Prerequisites and Installation Instructions

### Requirements

* Anaconda or Miniconda
* Python 3.11
* Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/weirenong/aiap_mini_project_1
cd aiap_mini_project_1
```

2. Create and activate a conda environment:
```bash
conda create -n aiapenv python=3.11
conda activate aiapenv
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

---

## Instructions for Executing the Pipeline

To run the complete pipeline, execute:

```bash
bash run.sh
```

Or directly:

```bash
python main.py
```

To open and run the analysis notebook:

```bash
jupyter notebook eda.ipynb
```

---

## Description of Logical Steps/Flow of the Pipeline

### Logical Flow of the Pipeline

The main script orchestrates the entire pipeline by:

* Loading configuration and the raw CSV into a DataFrame
* Cleaning the data and dropping columns that carry no predictive value
* Splitting the cleaned data into training (80%), validation (10%) and test (10%) sets
* Building the preprocessing pipeline (scaling plus one-hot encoding)
* Training three baseline regressors: Linear Regression, Ridge and Lasso
* Tuning Ridge and Lasso with `GridSearchCV`
* Selecting the best model by validation R² and evaluating it once on the held-out test set

### 1. Configuration (`config.yaml`)

The pipeline loads configuration from `src/config.yaml`, which specifies:

* CSV file path (`./data/mini_project_1_data.csv`)
* Target column name (`shares`)
* Feature categorisation: 22 numerical features, 2 nominal features, no passthrough features
* Train/validation/test split ratios (`val_test_size: 0.2`, `val_size: 0.5`, giving 80/10/10)
* Hyperparameter grid (`regressor__alpha` in `[0.01, 0.1, 1, 10, 100, 1000]`, `regressor__fit_intercept` in `[True, False]`)
* Cross-validation folds (`5`) and scoring metric (`r2`)

No column names are hardcoded in the pipeline; the feature lists drive both cleaning and preprocessing, so adding a feature is a one-line config change.

### 2. Data Ingestion (`main.py`)

The raw data is read from a single CSV of 35,680 rows and 30 columns. `ID` and `URL` are unique for every row (0 duplicates in either column), so the dataset needs no de-duplication step. The only structural defect is the null and sentinel values handled below.

### 3. Data Preparation (`DataPreparation` class)

`DataPreparation.clean_data()` applies every cleaning decision taken in the notebook, in this order:

* **Drop `ID` and `URL`.** Both are unique row identifiers with no generalisable predictive value.
* **Fill null content counts with 0.** `num_hrefs`, `num_self_hrefs`, `num_imgs`, `num_videos`, `self_reference_min_shares`, `self_reference_max_shares` and `self_reference_avg_shares` are null for 721 to 16,755 rows. A null means the article has no such content (no links, no images, no videos, no referenced article), not that the value is unknown, so 0 is the correct value rather than an imputation.
* **Fill null `data_channel` with `"none"`.** There are 5,462 null channels, and their median share count (1,900) is higher than world (1,100), entertainment (1,200) and business (1,400). The null group carries its own signal, so it is kept as an explicit category instead of being dropped or merged.
* **Clip negative keyword share values at 0.** `kw_min_min` (20,660 rows), `kw_avg_min` (767 rows) and `kw_min_avg` (6 rows) contain `-1` sentinels. A share count cannot be negative, so the sentinel is treated as 0.
* **Drop the 1,061 articles with `n_tokens_content == 0`.** For these rows `n_unique_tokens`, `n_non_stop_words`, `n_non_stop_unique_tokens` and `average_token_length` are all 0 too, so every text-derived ratio is degenerate.
* **Drop near-constant columns.** `kw_min_min` is `-1` for 58% of articles, `kw_max_max` is capped at 843,300 for 76% of articles, and `n_non_stop_words` is essentially 1.0 for almost every article. None of them can separate articles with different share counts.
* **Select the configured columns.** The frame is filtered to the 22 numerical features, 2 nominal features and the target, so nothing unexpected reaches the model.

The result is 34,619 rows and 25 columns (24 features plus `shares`).

The target's heavy right tail is deliberately kept. The maximum of 843,300 shares coincides exactly with the platform keyword cap that `kw_max_max` records for 76% of articles, so it is a genuine viral article at the platform ceiling, not a data entry error.

### 4. Preprocessing Pipeline (`DataPreparation._create_preprocessor`)

A `ColumnTransformer` is constructed with:

* **Numerical** (`StandardScaler`): the 22 numerical features, which differ in scale by several orders of magnitude (`n_comments` is typically below 10, `kw_avg_avg` is in the thousands, `kw_max_min` reaches six figures). Scaling is required because Ridge and Lasso penalise coefficients on the scale they are measured in.
* **Nominal** (`OneHotEncoder(drop='first', handle_unknown='ignore')`): `weekday` and `data_channel`. Dropping the first level avoids the dummy variable trap; `handle_unknown='ignore'` keeps inference safe when an unseen category appears.
* **Passthrough**: the config declares an empty list, and the code only appends the passthrough step when the list is non-empty.
* `remainder='drop'` — any column not declared in the config is excluded.

The transformed design matrix has 34 columns: 22 scaled numerical features, 6 weekday dummies and 6 channel dummies.

### 5. Model Training (`ModelTraining` class)

`ModelTraining` manages:

* Splitting the cleaned data into training (80%), validation (10%) and test (10%) sets (`random_state=42`)
* Training three baseline regressors with default hyperparameters: Linear Regression, Ridge and Lasso
* Tuning Ridge and Lasso with `GridSearchCV` over 6 alpha values and 2 intercept settings, 5-fold cross-validation, scored on R²
* Evaluating every model on the validation set with MAE, MSE, RMSE and R²
* Selecting the best model by validation R² and evaluating it once on the test set

---

## Overview of Key Findings from EDA

### Target Variable Characteristics

`shares` is a continuous, heavily right-skewed count with no zeros:

| Statistic | Value |
|-----------|-------|
| Minimum | 5 |
| Median | 1,400 |
| Mean | 3,370 |
| 90th percentile | 6,100 |
| 99th percentile | 31,864 |
| Maximum | 843,300 |

The mean is more than twice the median. The top 1% of articles start at 31,864 shares, and the single largest article reaches the platform cap of 843,300. The histogram is effectively a single bar at the left edge plus a long thin tail, while the zoomed view shows one peak between 500 and 2,000 shares. This shape is why RMSE ends up several times larger than MAE for every model.

### Key Predictors of Shares

**The one genuinely useful feature:**

* **`n_comments`** — the strongest signal by a wide margin. Spearman ρ = 0.36 and Pearson r = 0.52, and the scatter plot shows a widening cone: articles with fewer than 10 comments exceed 10,000 shares 1.1% of the time, while articles with 50 or more comments do so 10.2% of the time. The Pearson value is higher than Spearman because a handful of enormous comment threads also have enormous share counts.

**A weak secondary cluster:**

* **Keyword share history** — `kw_avg_avg` (Spearman 0.26, Pearson 0.10), `kw_max_avg` (0.22, 0.05), `kw_min_avg` (0.10), `kw_avg_min` (0.10), `kw_max_min` (0.09). The rank correlation is much stronger than the linear one, so the relationship is monotonic but not proportional.
* **Self-reference shares** — `self_reference_avg_shares` (0.20), `self_reference_min_shares` (0.19), `self_reference_max_shares` (0.18). Articles that reference popular pieces on the same site are shared somewhat more.
* **Link and image counts** — `num_hrefs` (0.10) and `num_imgs` (0.08) are barely above noise, and the scatter plots show vertical stripes rather than a trend.

**Features that carry essentially nothing:**

`n_tokens_content` (0.02), `n_unique_tokens` (-0.04), `n_non_stop_unique_tokens` (-0.07), `average_token_length` (-0.05), `n_tokens_title` (-0.04), `timedelta` (0.04), `num_videos` (0.03) and `num_keywords` (0.07) are all too small to build an interpretation on. Article length in particular has no relationship with sharing at all, so its role in the model is only to control the text-derived ratios.

### Categorical Findings

Median shares by `data_channel`:

| Channel | Median shares |
|---------|---------------|
| social_media | 2,100 |
| none (null channel) | 1,900 |
| technology | 1,700 |
| lifestyle | 1,600 |
| business | 1,400 |
| entertainment | 1,200 |
| world | 1,100 |

Median shares by `weekday`:

| Weekday | Median shares |
|---------|---------------|
| Saturday | 2,000 |
| Sunday | 1,900 |
| Friday | 1,500 |
| Monday | 1,400 |
| Thursday | 1,400 |
| Tuesday | 1,300 |
| Wednesday | 1,300 |

Weekend articles are shared more often, but the gap between the best and worst category is under 1,000 shares while the standard deviation of shares is nearly 12,000. Both categorical features are therefore weak on their own, and the boxplots show their interquartile ranges overlapping almost completely.

### Multicollinearity

The Pearson heatmap shows the keyword share features are strongly correlated with each other (`kw_avg_avg` and `kw_max_avg` at roughly 0.9), and the three `self_reference_*_shares` columns are even tighter. This is why the fitted Linear Regression assigns +1,707 to `kw_avg_avg` but -1,048 to `kw_max_avg`: once one is in the model, the other only carries the residual difference between them. It is also the main motivation for testing Ridge and Lasso.

---

## Feature Handling Description

### Data Cleaning

* Identifier columns (`ID`, `URL`) are dropped; both are unique per row and cannot generalise.
* Nulls in content counts (`num_hrefs`, `num_self_hrefs`, `num_imgs`, `num_videos`, `self_reference_min_shares`, `self_reference_max_shares`, `self_reference_avg_shares`) are filled with 0 because a null encodes "the article has none of this content".
* Nulls in `data_channel` become the explicit category `none` because the group has a distinct median share count.
* Negative values in `kw_min_min`, `kw_avg_min` and `kw_min_avg` are clipped at 0; `-1` is a sentinel for "no data", not a measurement.
* Articles with `n_tokens_content == 0` are dropped because all of their text-derived ratios are 0 and therefore meaningless.
* `kw_min_min`, `kw_max_max` and `n_non_stop_words` are dropped as near-constant columns.
* The viral tail is retained in full, including the 843,300-share article that sits on the platform keyword cap.

### Feature Processing

* **Numerical features** — all 22 are standardised with `StandardScaler`. This puts every coefficient on the same "per standard deviation" scale, which is what makes the coefficient bar plot comparable and what makes the Lasso penalty fair across features.
* **Nominal features** — `weekday` and `data_channel` are one-hot encoded with `drop='first'`, avoiding perfect collinearity between the dummies.
* **Passthrough features** — none. The config keeps the key for future use, and `ColumnTransformer` only includes the step when the list is non-empty.
* **`remainder='drop'`** — the model sees exactly the 24 configured features and nothing else.

| Feature | Reason for Exclusion |
|---------|----------------------|
| `ID` | Unique row identifier; no predictive value |
| `URL` | Unique per article; free text with no consistent structure |
| `kw_min_min` | Constant `-1` for 58% of articles, 22 distinct values in total |
| `kw_max_max` | Capped at 843,300 for 76% of articles; a platform ceiling rather than a signal |
| `n_non_stop_words` | Effectively 1.0 for almost every article |
| Rows with `n_tokens_content == 0` | 1,061 articles with all text ratios equal to 0 |

---

## Explanation of Model Choices

### Models Implemented and Justifications

1. **Linear Regression** — the interpretable baseline. It makes no assumption about which features matter, produces coefficients that can be read as shares per standard deviation, and is the right reference point for a target that is dominated by one moderate linear signal.
2. **Ridge Regression** — adds an L2 penalty, which shrinks coefficients toward zero and stabilises them when features are collinear. The keyword share columns correlate at roughly 0.9, so Ridge is the natural test of whether that collinearity is hurting the plain linear fit.
3. **Lasso Regression** — adds an L1 penalty, which can shrink coefficients to exactly zero and therefore performs feature selection. It tests whether the large block of near-zero features can be discarded without losing accuracy.

### Hyperparameter Tuning

* `GridSearchCV` explores `alpha` in `[0.01, 0.1, 1, 10, 100, 1000]` and `fit_intercept` in `[True, False]` for both Ridge and Lasso.
* 5-fold cross-validation on the training set is used so that the chosen alpha is not a product of a single split.
* Scoring is R², the same metric used for model selection.
* Tuning is performed on the training set only; the validation set is used to compare the tuned models against the baselines, and the test set is untouched until the final evaluation.

---

## Evaluation of Models

### Evaluation Metrics

* **MAE** — mean absolute error, in shares. The typical size of a prediction error, and the most intuitive metric here because it is not dominated by outliers.
* **MSE** — mean squared error, in shares squared. Penalises large errors heavily, which in this dataset means the viral articles dominate it.
* **RMSE** — square root of MSE, back in shares. Comparable to MAE, and the gap between the two measures how much of the error comes from the tail.
* **R²** — proportion of variance in shares explained by the model. The primary selection metric, and the only one that is independent of the target's scale.

### Model Performance Comparison

#### Baseline Models (Before Hyperparameter Tuning)

| Model | MAE | MSE | RMSE | R² |
| :--- | :---: | :---: | :---: | :---: |
| Linear Regression | 2552.0818 | 140,025,193.1107 | 11833.2241 | 0.32000 |
| Ridge Regression | 2552.0669 | 140,026,237.5150 | 11833.2683 | 0.31999 |
| Lasso Regression | 2551.7331 | 140,039,140.6186 | 11833.8134 | 0.31993 |

All three land within 0.0001 R² of each other. With a penalty of `alpha=1` and a target measured in thousands, neither regulariser changes the fit at all: the Lasso shrinks 0 of its 34 coefficients to zero. The models explain about 32% of the variance in shares, with a typical error of roughly 2,552 shares and an RMSE more than four times larger, which is the viral tail showing up in the squared errors.

#### Hyperparameter Tuned Models (After Grid Search)

| Model | MAE | MSE | RMSE | R² |
| :--- | :---: | :---: | :---: | :---: |
| Ridge Regression (Tuned) | 2548.7315 | 141,092,214.3030 | 11878.2244 | 0.31481 |
| Lasso Regression (Tuned) | 2645.6746 | 147,973,449.1153 | 12164.4338 | 0.28140 |

Both grid searches select `alpha=1000` with `fit_intercept=True`. Cross-validation on the training set prefers heavy shrinkage, but the validation set disagrees:

* Tuned Ridge loses 0.005 R² and 0.004 RMSE against the unregularised model. Shrinkage buys a little stability and gives up a little signal.
* Tuned Lasso loses 0.039 R² and 0.038 RMSE. At `alpha=1000` it shrinks 33 of the 34 coefficients to exactly zero and keeps only `n_comments`, so it degenerates into a one-feature model and discards the small but real keyword and self-reference contributions.

The disagreement between cross-validated training R² and validation R² is itself informative: with a target this skewed, each fold of the training data contains a different mix of viral articles, so the alpha that wins on the folds does not generalise. The unregularised models are the more robust choice on this feature set.

### Final Model Selection and Interpretation

The **Linear Regression** model was selected as the best model, with a validation R² of 0.32000, the lowest validation MAE (2,552.0818), no hyperparameters to tune, and coefficients that can be read directly. Its final test metrics are:

| Model | MAE | MSE | RMSE | R² |
| :--- | :---: | :---: | :---: | :---: |
| Linear Regression (Final Test) | 2299.9733 | 69,888,442.6436 | 8359.9308 | 0.26725 |

Key observations from the evaluation:

1. The choice of model matters far less than the choice of features. All five models fall within 0.04 R² of each other, and the two regularised models only get worse once tuned. The dataset does not contain enough article-level signal to explain share counts precisely, and a more complex model has nothing extra to exploit.
2. Validation R² (0.320) and test R² (0.267) differ by roughly five points. That gap is sampling noise in a single 3,462-row test set whose squared error is dominated by a handful of viral articles. Both numbers are reported rather than the more flattering one.
3. RMSE is several times MAE on both sets (11,833 vs 2,552 on validation, 8,360 vs 2,300 on test). The model is reasonable at ranking ordinary articles but cannot forecast the share count of an individual article that might go viral.
4. The model explains roughly a third of the variance. That is useful for a relative ranking use case - prioritising which articles to promote - but not for a financial forecast of referral traffic. Such a forecast would need features this dataset does not have: author, topic, publication time of day, or social follower counts.
5. Ridge remains the sensible fallback. Its validation performance is statistically identical to Linear Regression, and it would be preferred if the feature set were later expanded with more collinear keyword or share-history columns.

---

## Considerations for Deployment

### Model Persistence

To make this project production-ready, implement model persistence:

1. Add model saving functionality to the `ModelTraining` class:
```python
import joblib

def save_model(self, model, filename):
    """Save the trained model to a file."""
    joblib.dump(model, filename)

def load_model(self, filename):
    """Load a trained model from a file."""
    return joblib.load(filename)
```

2. Create a separate inference script (`inference.py`) that loads the saved pipeline and applies it to new articles. Because the preprocessor is part of the pipeline, saving the pipeline is enough - the scaler and encoder are persisted with the model and no separate preprocessing step is needed at inference time.

### Monitoring and Drift

* The model should be retrained on a rolling window. `timedelta` is measured relative to dataset acquisition, and social platform mechanics change, so a model trained on 2013-2014 sharing behaviour will decay.
* Prediction quality should be monitored with MAE on articles whose true share counts arrive later. R² is not a useful live signal because it depends on the variance of the batch being scored.
* Because the error distribution is dominated by viral articles, monitoring should track the share of predictions whose error exceeds a business threshold, not only the mean error.

### Operational Caveats

* Predictions should be surfaced as a relative ranking (for example, "promote / hold") rather than as an expected share count. The model explains about a third of the variance, and an RMSE above 8,000 shares on the test set is far too wide for a point forecast.
* The `data_channel` encoder uses `handle_unknown='ignore'`, so a new channel will be encoded as all zeros (the dropped baseline level). This is safe but silently treats the new channel as the reference category, so adding a channel should trigger a retrain.
* The empty-passthrough path in `_create_preprocessor` exists so that binary flags can be passed through unscaled in a future iteration without restructuring the pipeline.
