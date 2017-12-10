![](front.jpeg)
# BST260 Final Project: House Price Prediction

# Our Team

![](team.png)

# Watch Our Video on YouTube!
[![Watch the video on YouTube](http://img.youtube.com/vi/cK9vh489Di8/1.jpg)](https://www.youtube.com/watch?v=cK9vh489Di8&feature=youtu.be)


# Overview and Motivation

![](1.jpg)
(For this demo visualization, data was provided by [Redfin](https://www.redfin.com/), a national real estate brokerage)




Growing unaffordability of housing has become one of the major challenges for metropolitan cities around the world. In order to gain a better understanding of the commercialized housing market we are currently facing, we want to figure out what are the top influential factors of the housing price. Apart from the more obvious driving forces such as the inflation and the scarcity of land, there are also a number of variables that are worth looking into. Therefore, we choose to study the house prices predicting problem on Kaggle, which enables us to dig into the variables in depth and to provide a model that could more accurately estimate home prices. In this way, people could make better decisions when it comes to home investment.

Our object is to discuss the major factors that affect housing price and make precise predictions for it. We use 79 explanatory variables including almost every aspect of residential homes in Ames, Iowa. Methods of both statistical regression models and machine learning regression models are applied and further compared according to their performance to better estimate the final price of each house. The model provides price prediction based on similar comparables of people’s dream houses, which allows both buyers and sellers to better negotiate home prices according to market trend. 


# Related Work

- Stepwise
- PCA
- Random Forest
- Gradient Boosting
- (Ensemble Learning)
- Some kernal in Kaggle


# Initial Questions
Through this project, we sought to answer some major questions: 

1. What are the important features that affect the house price?

2. How to build a model to predict the house price? 

3. How to evaluate our prediction performance?



It is our job to predict the sales price for each house. For each Id in the test set, we must predict the value of the SalePrice variable. 

The metric to evaluate the models is Root-Mean-Square-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. Our predictions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price.


# Data

- Source

Our data was obtained from [Ames Housing dataset](https://ww2.amstat.org/publications/jse/v19n3/decock.pdf), which was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. The data includes 79 explanatory variables describing (almost) every aspect of residential homes. 

We also participated in the Kaggle Competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) 
Our best entry for the competition is 0.1169, which leads us to 358/2636 (top 15%) in the leaderboard.

![](kaggle.png)




# Exploratory Data Analysis

Firstly, we do some EDAs to gain a general understanding of our data.

- Some important matrices and trends
 
SalesPrice vs. Living Area 
![](eda1.png)

From the plot above we can see that there are two outliers which has high areas but low sale price. When fitting the models, we delete these two outliers in the training data.

![](eda2.png)

Conincided with our intuition, if the overall quality of the house is better, then the house price is higher.


![](eda3.png)

![](eda4.png)

In general, the newer the house is, the higher the price is. But the correlation is not very strong.


- Correlation Matrix

Here we examined the correlations between variables and correlations with our outcome of interest: SalePrice.

correlations between variables:

![](cor.png)
Figure: Correlation visualization with R packages(corrplot, ggplot2)

Correlations with SalePrice:
Here we use the R package tabplots to find strong-related variables to "Saleprice" among 79 variables, which would further help us do feature selection and engineering. 

Here are some of the plots we generated with R package tabplots to show the number and range of values for each variable as well as the covariance among the variables:


![](vk.png)
![](v0.png)

![](v2.png)
![](v3.png)

Of all numeric variables, Variables strongly correlated with hourse price (SalePrice) are:
 OverallQual, YearBuilt, YearRemodAdd, MasvnrArea, BsmtFinSF1, TotalBsmtSF, 1stFlrSF, GrLiveArea, FullBath, TotRmsAbvGrd, FirePlaces, GarageYrBlt, GarageCars, GarageArea, WoodDeskSF and OpenPorchSF
Which is consistent with our findings in the head map below:
![](corr.png)
 
# Data Cleaning
 
Before we rush into regression and machine learning prediction, it is very important to get our data "cleaned" enough. This process usually take 80% of time in a real-world data problem. In fact, in our project, we spend about 60% of our time cleaning the data ourselves! 

1. Missing Data and Different Data Types

When using the data, be careful about the following variables:

Ordinal feature: ExterCond, ExterQual, Fence, FireplaceQu, Functional, GarageFinish, GarageQual, HeatingQC, KitchenQual, OverallCond, OverallQual, BsmtCond, BsmtQual, BsmtExposure, BsmtFinType1, BsmtFinType2, GarageCond, PavedDrive

Read as numerial but actually is categorical: MoSold, MSSubClass

2. filling NAs and scale the data


 
 
- Stepwise selections

Stepwise Selection combines elements of both forward selection and backward elimination, allow us either to remove covariates from our previous model or add back in covariates that we had previously eliminated from our model, and in this sense, giving us chances to consider all possible subsets of the pool of explanatory variables and find the model that best fits the data according to some prespecified criterion, such as AIC(Akaike Information Criterion), BIC(Bayesian Information Criterion), and adjusted R square.[]

- Lowess Anlaysis

LOWESS (Locally Weighted Scatterplot Smoothing), or LOESS (Locally Weighted Smoothing), is often applied in regression analysis that creates a smooth line through a scatter plot. It is especially helpful when detecting nonlinear relationship between variables and predicting trends. In our study, LOWESS was first used to detect potential nonlinear associations between variables and sale prices. Since it performed the best results compared to other smoothing methods, we then used it to predict prices after PCA preprocessing.[]

- Principal component analysis

Principal component analysis (PCA) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components [1].

![](pca.png)

As mentioned in class, we can’t estimate the estimators of a high-dimensional nonlinear model via lm function. So we applied PCA to estimate predictors by minimizing  the squared error of the approximation.

- Lasso Regression

Lasso (Least Absolute Shrinkage and Selection Operator) regression is a regularized linear regression. It uses L1 norm to constrain the coefficients of the fitting model. Usually, some coefficients will be set to 0 under the constrain. Therefore, the lasso regression is more robust compared to ordinary linear regression.

![](lasso.png)

# Machine Learning

- Random Forest

Random forest is an ensembling machine learning method basing on classification tree or regression tree. In general, random forest will generate many decision trees and average their predictions to make the final prediction. When generating each decision tree, the random forest will use a subset of all features, which avoids the overfitting problem.

![](randomf.png)

![](rf.png)
- Regression Tree 

The regression tree is a good friend to help us decide which features matter when buying houses. Here is am example of regression tree:
![](tree.png)


To our surprise, the overall quality of the house is more important than the total square feet. 
The year when the house was built or remodeled also plays an important role in pricing. This coincide with our intuition since the year is related to the quality.


- Gradient Boosting

Similar to random forest, gradient boosting is another ensembling machine learning method basing on classification tree or regression tree. While in random forest every tree is weighted the same, every tree in gradient boosting tries to minimize the error between target and trees built previously. Gradient boosting is now a popular machine learning framework for both academia and industry.
![](xgboots.png)


- Ensemble Methods

Ensemble learning combines multiple statistical and machine learning algorithms together to achieve better predictive performance than any algorithm alone, because the errors in each model may cancel out in the ensembled model. In our project, we will try to ensemble the regression techniques we use (e.g. lasso regression, gradient boosting), to predict the sale prices and compare the ensembled model with other models.

In our project, we just simply stack several models, i.e. average their predictions to make our final prediction.

# Final Analysis

Our goal is to minimize the RMSE after log transformation, so when training the model, the target value is the logarithm of the observed sales price. Besides, we add one more feature - total square feet “TotalSF”, which is defined as TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF.

Some models (e.g. linear models) perform better when the predictors are “normal”. Therefore we use Box-Cox transformation to transform the features of which skewness is high. 



- Basic Models


We use 5-fold cross validation to evaluate how each model performs. Each model’s RMSEs in cross validation (CV) and in leaderboard (LB) are as follows:

![](table.png)

Alough lasso performs best in cross validation, but gradient boosting model provided by sk-learn is better in leader board. We think that it comes from the overfitting problem of lasso regression. In both cross validation and leaderboard, the random forest does not perform well. In this test, random forest avoid the problem of overfitting, but it underfits the data at the same time. The “PCA + LOESS” model performs worst, since LOESS model is not a good model for complex regression problem.


- Ensemble Method (Stacking)

Based on the above result, we choose two models - lasso and gradient boosting in sklearn, and average their predictions to make our final prediction. The RMSE of the stacking model is 0.1169, which leads us to 358/2636 (top 15%) in the leaderboard.

![](kaggle.png)

Code can be found on: https://github.com/BST260-final-group-project/project-files/tree/master/final-analysis


# Reference and Source
- The Source of Data
- The Github page of our full project
https://github.com/BST260-final-group-project



