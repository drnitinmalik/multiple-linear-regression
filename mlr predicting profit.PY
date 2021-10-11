# Code by Nitin Malik, India, https://www.linkedin.com/in/drnitinmalik/

# Multiple Linear Regression: predicting profits from various types of expenditures in USA startup companies

# STEP1: IMPORTING THE LIBRARIES
import numpy as np
import pandas as pd
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import scipy.stats as stats
import statsmodels.api as sm

# STEP2: IMPORTING THE DATASET
dataset = pd.read_csv('50-startups.csv')
dataset.head()

# STEP3: EXPLORATORY DATA ANALYSIS
dataset.info()
dataset.describe()
dataset.shape #size

# STEP4: DATA PREPROCESSING

# duplicate data
dataset.duplicated() # not all boolean values visible
dataset.duplicated().sum() # no duplicates # in case there are, drop them

# missing values
dataset.isnull().sum() # no missing values # in case there are,  impute them

# STEP5: CHECK FOR MULTICOLLINEARITY
dataset.corr() # correlation matrix # correlation between each pair of independent variables

# STEP6: ENCODING CATEGORICAL DATA

dummies=pd.get_dummies(dataset)
dummies
dummies.columns # returns the names of all the columns as a list

# STEP7: SEGGREGATING THE FEATURE MATRIX AND TARGET
X = dummies.iloc[:, [0,1,2,4,5,6]] 
y = dummies.iloc[:, 3] # Take profit (4th column) as dependent variable
X.head()
y.head()

# STEP8: TRAIN (X_train & Y_train) - TEST (X test & y_test) SPLIT 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

# dimensions of train and test data
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# description of train and test data
X_train.describe()
X_test.describe()
y_train.describe()
y_test.describe()

# STEP9: BUILDIDING MLR MODEL
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # object regressor is created
regressor.fit(X_train, y_train)  # Fitting Linear Regression to the Training set

# K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10)
accuracies

# Regression coefficients
regressor.coef_ # slope
regressor.intercept_

# STEP10: PREDICTING ON TEST DATA
y_pred = regressor.predict(X_test)
y_pred

# STEP11: CHECK FOR ASSUMPTIONS

# Check for Endogenity of regressors
linearity_test_df = pd.DataFrame(X_test)
endogenity_check = linearity_test_df.corr()
endogenity_check

# Levene test for Homoscedasticity
residual_test = np.column_stack([y_test,y_pred])
residual_test = pd.DataFrame(residual_test)
residual_test.columns='Y_test predictions'.split()
sns.jointplot(x='Y_test', y='predictions', data=residual_test, kind='reg')

stats.levene(residual_test['Y_test'], residual_test['predictions'])

# STEP12: MODEL EVALUATION: MSE, R2
from sklearn.metrics import mean_squared_error, r2_score
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)

# Fit the model, according to the OLS (ordinary least squares)
regressor_OLS = sm.OLS(endog = y, exog = X).fit()
regressor_OLS.summary()
