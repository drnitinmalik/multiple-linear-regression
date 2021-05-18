# Pre-regression steps, Regression model, Post-regression analysis

In multiple regression (regression is also known as function approximation), we are interested in predicting one (dependent) variable from two or more (independent) variables e.g. predicting height from weight and age. Regression implies causation. Change in the dependent variable is due to the change in the independent variables. Linear regression implies that the relationship between the dependent variable and independent variables is linear and thus can be described by a linear plane known as the regression plane. We are in the process of finding a regression plane that fits (touches) the maximum number of data points (no. of data points = no. of records in the dataset).

**MLR MODEL**

Objective: Predict the dependent variable Yi

Model structure: Yi=β0 + β1XI + β2X2 + ...  βKXK + εi where β0 (y-intercept/constant) and βK (slope) are population regression coefficients and εi is the prediction error (prediction does not match with the actual) (error also termed as residue)

Model parameters: sample regression coefficients b0, b1... bk [determined by solving normal equations]

Model hyperparameters: Summed squared error, Mean square error [error functions]

Methods to estimate model parameters: Ordinary Least square (Minimise error function), Maximum likelihood (likelihood is the probability that a dependent variable can be predicted from the independent variables)

Model assumptions: Relationship between dependent and independent variables is linear, Dependent variable is continuous (interval or ratio), No correlation (a measure of the relationship between two variables derived from covariance) between errors & between error and independent variables, all errors have equal variances, errors follow a normal probability distribution, independent variables are not collinear (no multicollinearity)

**USE CASE**

Let's say we want to predict the profits (dependent variable) of 50 US Startup companies from the data which contains three types of expenditures (R&D spend, Administrative spend and Marketing spend, all of which is numerical data) and their location (categorical data). The [python code](https://github.com/drnitinmalik/multiple-linear-regression/blob/main/mlr%20predicting%20profit.PY) and the [data file](https://github.com/drnitinmalik/multiple-linear-regression/blob/main/50-startups.csv) are available on GitHub

**RESULTS**

Duplicate values: None
Missing value: None
Multicollinearity: Yes. 
Regression coefficients:
array([ 5.69631699e-16, 4.99600361e-16, -3.05311332e-16, 1.17961196e-15,
4.05057932e-16, -8.32667268e-17, 1.00000000e+00, 2.20489350e-13, 2.81742200e-12])
y-intercept: -2.6193447411060333e-10
prediction on test data
array([ 96778.92, 35673.41, 191792.06, 192261.83, 132602.65, 108552.04,
69758.98, 110352.25, 146121.95, 125370.37, 182901.99, 77798.83, 96712.8 ])
Levene test for homoscedasticity
test statistic=5.351611538586933e-29, p-value=1.0.

**DISCUSSION**

The correlation between marketing spend and R&D is on the higher side which confirms multicollinearity.
After encoding the categorical variable using one-hot encoding, we have 10 columns
