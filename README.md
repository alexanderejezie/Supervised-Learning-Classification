# 🎯What I Aimed For.

The project aims to build and train a machine learning model to predict the likelihood of a Systemic crisis emergence given a set of indicators like the annual inflation rates.

This project is a checkpoint for the 'Systemic Crisis, Banking Crisis, inflation Crisis In Africa' dataset that was provided by Kaggle.

# ℹ️ Instructions.

## Data Handling:
- Import Data and Perform Basic Exploration: Import the dataset and explore its basic structure and statistics.
- Display General Information: Display basic information and summary statistics about the dataset.
- Create a Pandas Profiling Report: Generate a pandas profiling report to gain insights into the dataset.
- Handle Missing and Corrupted Values: Address any missing or corrupted values in the dataset.
- Remove Duplicates: Remove any duplicate records if they exist.
- Handle Outliers: Detect and handle any outliers in the dataset.
- Encode Categorical Features: Encode categorical features to prepare them for machine learning algorithms.
- Select Target Variable and Features: Select the target variable (Energy) and the feature set.
- Split Dataset: Split the dataset into training and test sets.

## Model Building and Evaluation:
- Select ML Regression Algorithm: Based on the data exploration phase, select a suitable ML regression algorithm.
- Train the Model: Train the model on the training set.
- Assess Model Performance: Evaluate the model performance on the test set using relevant evaluation metrics.

# 🛠️ Tools Used:
- VSCode: Code editor.
- GitHub: Repository hosting.
- Git: Version control.
- Python: Programming language.
- Google: For research and troubleshooting.
- YouTube: For educational videos.

# Libraries Used:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, recall_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
