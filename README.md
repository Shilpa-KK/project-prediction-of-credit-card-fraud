# project-prediction-of-credit-card-fraud
Efficient classification model to predict whether a transaction is fraudulent or not
## Project’s Purpose:
As we move towards the digital world, cybersecurity is becoming a critical part of our lives. For example, when we purchase any product online many customers prefer credit cards as an option. But on another side credit, debit and other prepaid card related fraud activities have been rising these days. The common purpose of these activities is to extract finance related credentials from individuals and perform financial transactions on their behalf. To handle this problem these days machine learning algorithms can help us track abnormal transactions, classify them and stop the transaction process if required.
This project aims to predict such fraudulent transactions performed on credit cards by developing an efficient machine learning model. Several classification algorithms can perform best and are easily deployable. We will be using Logistic Regression, Random Forest and XGBoost models to build our fraud detector and found out that XGBoost is the efficient one.
## Problem Statement:
A credit card is one of the most used financial products to make online purchases and payments. Though the Credit cards can be a convenient way to manage your finances, they can also be risky. Credit card fraud is the unauthorized use of someone else's credit card or credit card information to make purchases or withdraw cash.
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. 
The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
We have to build a classification model to predict whether a transaction is fraudulent or not.
## Data sources and methodology:
The dataset for this project can be accessed by clicking the link provided below.
[creditcard.csv](url)
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
## Importing the Packages
First, we need to install [Anaconda distribution](url) which includes most of the packages you will come across and then launch Jupyter notebook. Then import all the dependencies or primary packages into our python environment.

```#Import Required Packages
import pandas as pd 
import numpy as np 
import os
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
!pip install scikit-learn
!pip install imbalanced-learn
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
!pip install xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
```
- pandas: used to perform data manipulation and analysis.
- numpy: used to perform a wide variety of mathematical operations on arrays.
- matplotlib: used for data visualization.
-	seaborn: built on top of matplotlib with similar functionalities.
-	warnings: to manipulate warnings details filterwarnings('ignore') is to ignore the warnings thrown by the modules (gives clean results).
-	%matplotlib: to enable the inline plotting.
-	Scikit-learn: used for data split, for building and evaluating the classification models such as Logistic Regression and Random Forest, used to import StandardScaler class for standardizing features by removing the mean and scaling to unit variance, used to import GridSearchCV for hyper-parameter tuning in machine learning models.
-	imbalanced-learn: used for handling imbalanced datasets i.e. SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class
-	xgboost: used for importing xgboost classifier model algorithm
## Data Processing and EDA
Load the csv data from local directory using os  python library and convert that into dataframe using panda library and view initial data. We found 31 columns and out of that only non-transformed variables to work with are ‘Time’, ‘Amount’ and ‘Class’ (1: fraud, 0: non_fraud) also no missing values found. 
Let’s have a look at how many fraud cases and non-fraud cases are there in our dataset. Along with that, let’s also compute the percentage of fraud cases in the overall recorded transactions.
We can see that out of 284,807 samples, there are only 492 fraud cases which is only 0.17 percent of the total samples. So, we can say that the data we are dealing with is highly imbalanced data and needs to be balanced before modeling and evaluating.
While seeing the statistics, it is seen that the values in the ‘Amount’ variable are varying enormously when compared to the rest of the variables. To reduce its wide range of values, we can normalize it using the ‘StandardScaler’ method in python.
### Correlation Matrix
The correlation matrix graphically gives us an idea of how features correlate with each other and can help us predict what are the features that are most relevant for the prediction.
- In the HeatMap we can clearly see that most of the features do not correlate to other features but there are some features that either has a positive or a negative correlation with each other. For example, V2 and V5 are highly negatively correlated with the feature called Amount. We also see some correlation with V20 and Amount. 
-	Notice that if lower the negative correlations values are, the more likely the end result will be a fraudulent transaction. also, if higher the positive correlation values are, the more likely the end result will be a fraudulent transaction.
## Feature Engineering & Feature Selection
Since the dataset most features are dimensionally reduced already, we will assume no additional feature engineering is needed at this moment. While seeing the statistics, it is seen that the values in the ‘Amount’ variable are varying enormously when compared to the rest of the variables. To reduce its wide range of values, we can normalize it using the ‘StandardScaler’ method in python. Also dropping Time attribute as it is not an important feature.
### Standard Scaling
Standard scaling, also known as standardization, is a preprocessing technique in machine learning used to transform the features of your data so that they have a mean of zero and a standard deviation of one. This process ensures that all features contribute equally to the learning process, preventing any single feature from dominating due to its larger magnitude. 
While seeing the statistics, it is seen that the values in the ‘Amount’ variable are varying enormously when compared to the rest of the variables. To reduce its wide range of values, we can normalize it using the ‘StandardScaler’ method in python. 
After running the code, we can see an array with a scaled value ranging from 0-1. 

### Splitting the data into features and target variable: 
Store the input attributes in variable X and output attribute in variable y. Here, Class column is the target and all other columns are features.
## Balancing the Data
### Oversampling with SMOTE (Synthetic Minority Oversampling Technique)
One issue with unbalanced classification is that there are too few samples of the minority class for a model to learn the decision boundary successfully. Oversampling instances from the minority class is one solution to the issue. Before fitting a model, we duplicate samples from the minority class in the training set.
Synthesizing new instances from the minority class is an improvement over replicating examples from the minority class. It is a particularly efficient type of data augmentation for tabular data. In our project, create an instance of the SMOTE class and then apply SMOTE to the dataset (X, y). We can see equal distribution over resampled dataset shape.
## Train-Test split
In this approach, we split the data randomly into a train set and a test set (80% — 20% split) using train_test_split function from the sklearn.model_selection module. In our case the function is like:
```x_train, x_test, y_train, y_test = train_test_split(x_smote, y_smote, test_size=0.2, random_state=42)
```
1. x_smote: This is the feature set that you want to split into training and testing sets.
2.	y_smote: This is the target variable corresponding to x_smote that you want to split.
3.	test_size=0.2: This specifies the proportion of the dataset to include in the test split. Here, 20% of the data will be used for testing, and the remaining 80% will be used for training.
4.	random_state=42: This is a seed value to ensure reproducibility of the results. Using the same seed value will produce the same split every time you run the code.
So, the function splits x_smote and y_smote into four sets: x_train, x_test, y_train, and y_test, with 80% of the data used for training and 20% for testing.
## Training & Performance Evaluation of Models
Steps involved are:
1.	Fitting each model (Logistic Regression, Random Forest, XGBoost) to the Training set
2.	Predicting the test result for each model
3.	Measure the classification metrics for each model in the test set 
4.	Visualizing the test set result
5.	Select the model that achieved the best metrics in the singular test set and do hyperparameter tuning with that model for model improvement 
6.	After fitting with the best parameters, predictions are made with best model and evaluate the classification metrics for that model
To perform Model Training, we will use several different models like Logistic Regression, Random Forest, XGBoost, etc., Before that, we need to split the data. All these models can be built feasibly using the algorithms provided by the scikit-learn package. Only for the XGBoost model, we are going to use the xgboost package.
**Logistic Regression:-** Logistic Regression is a statistical and machine learning technique used for binary classification problems –that is, situations where your data observations belong to one of two possible categories. Therefore, the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1. 
Steps involved are:
1.	Binary Outcomes Modeling: The crux of Logistic Regression lies in its ability to estimate the probability that a given input point belongs to a particular category.
2.	Odds Ratio: Unlike Linear Regression which predicts a continuous output, Logistic Regression predicts the log-odds of the dependent variable.
3.	Sigmoid Function: It employs a sigmoid (or logistic) function to squeeze the output of a linear equation between 0 and 1 — the core of obtaining something interpretable as a probability.
4.	Maximize Likelihood: The fitting process involves maximizing the likelihood of the observed data, making the observed outcomes as probable as possible given the model’s parameters.
5.	Threshold Determination: Finally, by setting a threshold, often 0.5, the model decides to which category to assign the new observation.
Cons of Logistic Regression:
-	Assumes a linear relationship between the independent variables and the log-odds.
-	Not as powerful as more complex classifiers like Random Forest or Gradient Boosting.
-	Can be sensitive to outliers and influential points.
-	Performance may suffer with non-linear decision boundaries.
**Random Forest:-** Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset. The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.
Procedure:
Random Forest works in two-phase first is to create the random forest by combining N decision tree, and second is to make predictions for each tree created in the first phase.
Steps involved are:
Step-1: Select random K data points from the training set.
Step-2: Build the decision trees associated with the selected data points (Subsets).
Step-3: Choose the number N for decision trees that you want to build.
Step-4: Repeat Step 1 & 2.
Step-5: For new data points, find the predictions of each decision tree, and assign the new data points to the category that wins the majority votes.
**XGBoost (Extreme Gradient Boosting):-** XGBoost employs a technique called gradient boosting. This is where the algorithm starts by creating a simple decision tree and then iteratively adds more trees to the model, each one focusing on correcting the errors made by the previous trees. This process continues until a predetermined number of trees is reached or the model’s performance no longer improves significantly. It has become one of the most popular and widely used machine learning algorithms due to its ability to handle large datasets and its ability to achieve state-of-the-art performance in many machine learning tasks such as classification and regression. 
XGBoost generally outperforms Random Forest in many aspects, including:
-	Overfitting: XGBoost's tree pruning prevents overfitting, while Random Forest can overfit with similar samples.
-	Imbalanced data: XGBoost handles imbalanced data effectively, while Random Forest may struggle.
-	Hyperparameter tuning: XGBoost requires fewer hyperparameters and is less sensitive to changes, making it easier to tune.
-	Categorical variables: XGBoost handles categorical variables more robustly than Random Forest.
### Hyperparameter tuning with XGBoost model
Hyperparameter tuning is an important step in building a robust machine learning model, including the XGBoost model for a credit card fraud detection problem. We are using GridSearchCV from the scikit-learn library for hyperparameter tuning of an XGBoost model. 
1. Hyperparameter Grid: A grid of hyperparameters is defined. You can adjust these values based on your requirements. These parameters help in tuning the model to achieve better performance and generalization. Here’s a breakdown of each hyperparameter in our grid:
 -	n_estimators: This parameter specifies the number of trees in the ensemble. More trees can improve performance but also increase computation time.
    Values: [100, 200, 300]
 -	max_depth: This controls the maximum depth of each tree. Deeper trees can capture more complex patterns but may also lead to overfitting.
    Values: [3, 5, 7]
 -	learning_rate: This parameter scales the contribution of each tree. Lower values make the model more robust to overfitting but require more trees.
    Values: [0.01, 0.1, 0.2]
 -	subsample: This fraction determines the proportion of the training data used to fit each tree. Using a fraction less than 1.0 can help prevent overfitting.
    Values: [0.6, 0.8, 1.0]
 -	colsample_bytree: This fraction specifies the proportion of features (columns) used to build each tree. Using a subset of features can improve generalization.
    Values: [0.6, 0.8, 1.0]
2.	Grid Search: GridSearchCV is used to perform cross-validation and find the best hyperparameters based on the F1 score. In our case we are setting up a GridSearchCV for an XGBoost model to optimize hyperparameters. The parameter description are as follows:
 -	estimator=xgb_model: Specifies the model you’re tuning, which is an XGBoost model.
 -	param_grid=param_grid: The dictionary of hyperparameters you want to test.
 -	scoring='f1': Uses the F1 score as the evaluation metric, which is great for imbalanced datasets.
 -	cv=3: Performs 3-fold cross-validation.
 -	verbose=1: Provides detailed output during the fitting process.
 -	n_jobs=-1: Utilizes all available CPU cores to speed up the computation.
3.	Model Evaluation: After fitting with the best parameters, predictions are made, and a confusion matrix and classification report are printed.
    Best parameters found: ```{'colsample_bytree': 0.6, 'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 300, 'subsample': 1.0}```
### Evaluation Metrics
To evaluate our built models, we are using the evaluation metrics provided by the scikit-learn package. Our main objective in this process is to find the best model for our given case. The evaluation metrics we are going to use are the accuracy score, precision, recall, f1 score, roc-auc score and finally the confusion matrix.
1. Accuracy score
Accuracy score is one of the most basic evaluation metrics which is widely used to evaluate classification models. The accuracy score is calculated simply by dividing the number of correct predictions made by the model by the total number of predictions made by the model (can be multiplied by 100 to transform the result into a percentage). Accuracy isn't a suitable metric for our problem. 
2. Precision
It is the total number of true positives divided by the true positives and false positives. Precision makes sure we don't spot good transactions as fraudulent in our problem.
3. Recall
It is the total number of true positives divided by the true positives and false negatives. Recall assures we don't predict fraudulent transactions as all good and therefore get good accuracy with a terrible model.
4. F1 Score
It is the harmonic mean of precision and recall. It makes a good average between both metrics. It is calculated by dividing the product of the model’s precision and recall by the value obtained on adding the model’s precision and recall and finally multiplying the result with 2. The more the F1 score better will be performance.
5. ROC-AUC (Receiver Operating Characteristic – Area Under Curve) score
It tells us how well a machine learning model can separate things into different groups. The score is between 0.5(random guessing) and 1(perfect performance).
6. Confusion Matrix
Typically, a confusion matrix is a visualization of a classification model that shows how well the model has predicted the outcomes when compared to the original ones. Usually, the predicted outcomes are stored in a variable that is then converted into a correlation table. Using the correlation table, the confusion matrix is plotted in the form of a heatmap. 
There are 4 terms you should keep in mind: 
1.	True Positives: It is the case where we predicted Yes and the real output was also yes.
2.	True Negatives: It is the case where we predicted No and the real output was also No.
3.	False Positives: It is the case where we predicted Yes but it was actually No.
4.	False Negatives: It is the case where we predicted No but it was actually Yes.
#### Models Test Results
![image](https://github.com/user-attachments/assets/4baadf54-18c1-4457-ae8d-d3943de26a95)
-	For logistic regression model, out of 56976 True Fraud transactions, 4706 is predicting incorrectly as normal. i.e. Recall score is low. It has only 91% recall score.

![image](https://github.com/user-attachments/assets/582e076f-5b04-449a-9cde-2887e3faf599)
- For Random Forest model Accuracy is 99% and out of 56976 True Fraud transactions, all transactions are predicting correctly as Fraud and nothing is predicted incorrectly. 
-	But it has high computational cost and it is time consuming.

![image](https://github.com/user-attachments/assets/554bc56b-f095-458b-b0c1-e090568747a4)
-	For XGBoost model also accuracy is 99% and all Fraud transactions are predicted correctly as Fraud. 
-	XGBoost is efficient in terms of computational cost and speed and it prevents overfitting

![image](https://github.com/user-attachments/assets/590874ec-721c-4d4a-8432-9f2f6d6f1a8c)

-	Result shows as the XGBoost model performance is increased with hyperparameter tuning
-	Accuracy is 99%, F1 score is 0.9997 and recall score is 1 i.e., all Fraud transactions are predicted as correctly.

























