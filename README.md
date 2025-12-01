## Comparing Aggregate Models for Wage Prediction

This project explores the idea of "wisdom of the crowd" in a regression setting by comparing individual regression models to a VotingRegressor ensemble. The goal is to predict hourly wages from census-like survey data and to investigate whether an ensemble of models can outperform the best single model, as well as to understand which features matter most for wage prediction.

## Dataset

The data comes from OpenML (data_id=534) and contains 534 rows with 11 columns, including:

- **Target**
  - `WAGE`: hourly wage (float)

- **Numeric features**
  - `EDUCATION`: years of education  
  - `EXPERIENCE`: years of work experience  
  - `AGE`: age in years  

- **Categorical features**
  - `SOUTH`, `SEX`, `UNION`, `RACE`, `OCCUPATION`, `SECTOR`, `MARR`

Categorical features were converted to dummy/indicator variables using `pandas.get_dummies`, making the full feature matrix numeric and ready for modeling.

## Methods

The workflow in the notebook includes:

1. **Data preparation**
   - Load the dataset via `fetch_openml`.
   - Separate the target (`WAGE`) from the predictors.
   - Apply one-hot encoding to categorical variables using `pd.get_dummies(drop_first=True)`.
   - Split the encoded data into training and test sets with `train_test_split`.

2. **Preprocessing**
   - Use `StandardScaler` inside a `Pipeline` so that models sensitive to feature scale (e.g., KNN, SVR) are trained on standardized features.

3. **Individual models evaluated**
   - `LinearRegression`
   - `Ridge`
   - `KNeighborsRegressor`
   - `DecisionTreeRegressor`
   - `SVR`

   Each model is wrapped in a pipeline with scaling where appropriate, and evaluated using Root Mean Squared Error (RMSE) on the test set.

4. **Ensemble (VotingRegressor)**
   - First ensemble: `VotingRegressor` combining Ridge, SVR, KNN, and DecisionTree.
   - Tuned ensemble: a second `VotingRegressor` that removes the weakest individual models, keeping only:
     - `Ridge`
     - `LinearRegression`
     - `SVR`

5. **Interpretability with permutation importance**
   - Use `sklearn.inspection.permutation_importance` on the best model (Ridge) to measure how shuffling each feature affects prediction error, providing a model-agnostic view of feature importance.

## Results

### Individual models

On the test set, the individual models produced the following RMSE (lower is better):

- **Ridge**: ~4.42  
- LinearRegression: ~4.42 (very close to Ridge, but slightly worse)  
- SVR: ~4.66  
- KNeighborsRegressor: ~4.78  
- DecisionTreeRegressor: ~7.65  

Ridge Regression emerged as the strongest single model and was used as the performance baseline.

### VotingRegressor (all models)

The initial `VotingRegressor` that combined Ridge, SVR, KNN, and DecisionTree achieved an RMSE of about **4.75**, which is **worse** than Ridge alone. This indicates that the weaker models (especially KNN and DecisionTree) pulled down the ensemble’s overall performance.

### VotingRegressor (strong models only)

A second ensemble was built using only the stronger models:

- Ridge  
- LinearRegression  
- SVR  

This tuned ensemble achieved an RMSE of about **4.46**, which improved over the first ensemble (4.75) but still **did not beat Ridge Regression alone** (4.42). In this dataset, the best single model remained more accurate than the averaged predictions of the group.

### Interpretation: Does the “wisdom of the crowd” win?

For this exercise, the “wisdom of the crowd” did **not** outperform the strongest individual learner. Removing weaker models helped narrow the gap, but Ridge Regression remained the best performer. This illustrates that ensemble methods are not guaranteed to outperform a well-specified single model, especially when some base learners are significantly weaker and are given equal weight.

## Feature Importance and Interpretability

To understand which features mattered most in predicting wages, permutation importance was computed for the Ridge Regression model.

Key findings:

- **EDUCATION** had the highest importance by a wide margin, suggesting that years of education are the most influential factor in predicting wages.
- **OCCUPATION_Management** and **OCCUPATION_Professional** were the next most important features, indicating that working in management or professional roles is strongly associated with higher or more predictable wages.
- **SEX_male** also showed meaningful importance, capturing wage differences across gender.
- Features such as **OCCUPATION_Service**, **RACE_White**, **SOUTH_yes**, and **AGE** had modest but non-zero contributions.
- Variables like **EXPERIENCE**, **RACE_Other**, and the **SECTOR** dummy variables had very low or slightly negative permutation importance, suggesting they did not materially improve the model and may introduce noise.

Permutation importance was chosen because it is model-agnostic and directly measures how much each feature impacts prediction error when its values are randomly shuffled.

Here is your section properly formatted in clean, valid Markdown:

Here is the corrected Markdown syntax — everything is now properly opened and closed:

````markdown
## How to Run This Project

1. **Clone the repository:**

   ```bash
   git clone https://github.com/<your-username>/wisdom-of-the-crowd.git
   cd wisdom-of-the-crowd-regression
````

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter and open the notebook:**

   ```bash
   jupyter notebook notebooks/wisdom_of_the_crowd.ipynb
   ```

5. **Run all cells** to reproduce the analysis.

```


```


