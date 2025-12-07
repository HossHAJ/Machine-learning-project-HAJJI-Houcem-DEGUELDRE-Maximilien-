
# NBA Team Points Prediction – Machine Learning Project

## Project Overview
This project aims to **predict the number of points (PTS) scored by an NBA team in a game** using Machine Learning techniques.  
The goal is to explore the data, build predictive models, and evaluate their performance.

The project follows a **complete ML pipeline**:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Model training
- Evaluation and interpretation of results

---

##  Dataset
The dataset contains **NBA game statistics**, including team-level features used to predict:
- **Target variable:** `PTS` (Points scored by the team)

---

##  Exploratory Data Analysis (EDA)
The EDA includes:
- Dataset overview (`shape`, `info`, `describe`)
- Data types (`dtypes`)
- Missing values analysis
- Target distribution analysis
- Correlation heatmap
- Residual analysis

---

##  Models Used
The following Machine Learning models are implemented:
- **XGBoost Regressor**
- **Random Forest Regressor**
- **CatBoost Regressor**

Hyperparameter optimization is performed using **Optuna**.

---

##  Evaluation Metrics
Model performance is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE / RMSE)
- R² Score
- Residual plots

---

##  Libraries & Tools
Main Python libraries used:
```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
catboost
optuna
streamlit
```
##  Contributors & Responsibilities

###  Houcem Hajji
- Exploratory Data Analysis (EDA)
- Feature analysis & data understanding
- XGBoost model implementation
- Model optimization using Optuna
- Streamlit application development

###  Maximilien Degueldre
- Random Forest model implementation
- CatBoost model implementation
- Model optimization using Optuna
- Streamlit application development

