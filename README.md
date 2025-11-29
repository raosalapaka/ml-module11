# Overview

This READMe has the analysis for the features of a vehicle that drive its price. The approach was to follow CRISP-DM framework to find the best model to predict the target price and then use this model to find the features that have the most impact on the target price which can be communicated to the clients who are used cars salespersons. vehicles.csv dataset from Kaggle was used for this analysis. The Jupyter notebook assignment11_1_RS.ipynb has the code for this analysis

Link to the jupyter notebook: [Assignment 11_1 Jupyter notebook](https://github.com/raosalapaka/ml-module11/blob/main/assignment11_1_RS.ipynb)
Link to github: [Github] (https://github.com/raosalapaka/ml-module11)
Link to the repository: [Git repository](https://github.com/raosalapaka/ml-module11.git)

# Analysis

---
# **Features driving price of a vehicle**

## **CRISP-DM**

## Data Exploration

### Initial exploration 

Some of the columns were eliminated right away from analysis as they are likely not useful. Some explanation on the reasoning:

**'id'**: this is just an identifier and does not capture a feature of the vehicle

**'region'**: this could be impactful but given these are all in the US, and the number of unique values are large, analyzing this can become complex with little benefit

**'VIN'**: is another identifier that does not capture any useful information about the car

**'paint_color'**: could be useful but likely will have less impact

**'state'**: same reasoning as region

**'model'**: will also drop the model column to make the analysis simpler, even though model likely has a direct impact on price because of branding. This column has too many string values which will complicate the training a lot especially when using onehotencoder for this.


### Further exploration

Analyzing other columns. Looking at the spread of the columns among its different values. The initial idea is that if most of the rows have the same value, it may not be useful. For example, if we take an extreme case, if all vehicles are of fuel type gas, it does not provide useful information

For 9 of the following columns we plot the counts against values and look for skews and outliers

![alt text](image.png)

### Remark
At this point, if we blindly drop all rows with NaN values, the data set reduces by a lot (>80%). 

### Further exploration (conclusion)

From above it is clear that the following columns do not provide a lot of value:

**'title_status'**: most of the vehicles are in clean state and there is very little variance

###Remark

To prevent losing a lot of data due to missing data, looked at the 'model' column and imputed the most common value to the following columns: 'cylinders', 'drive', 'size', 'type', 'fuel'. Used mode() to find the most common value for each of these columns based on 'model'. This is simplistic approach but went this for now

**This is simplistic. Explore if we can fix the data by distributing the values in the same proportion as their ratios**

### Final remark on data exploration
There are a lot of rows that do not have 'condition' value but this could be an important column as the condition of the car should be an important feature to predict car price. As a result kept this column even if it reduces the samples by a lot (almost 43%)


### Data Preparation

After our initial exploration and fine-tuning of the business understanding, it is time to construct our final dataset prior to modeling.  Here, we want to make sure to handle any integrity issues and cleaning, the engineering of new features, any transformations that we believe should happen (scaling, logarithms, normalization, etc.), and general preparation for modeling with `sklearn`. 

###Remark

- Clean the target 'price' column by dropping all rows which have 0 value as this is invalid
- Drop 'manufacturer' and 'model' columns to reduce complexity
- Transformed 'transmission', 'drive', 'size', 'type' and 'fuel' columns using OneHotEncoder as they really do not have any ordinal relation
- Transformed 'condition' and 'cylinders' using OrdinalEncoder
- The above transformation results in 32 columns (exclusing price column)

## Training and Test split

- Split the data with test_size=0.3
- Took the log of price as target y. This was done mainly for:
  - target price is large so taking log would reduce variance and make it less sensitive to outliers
  - the goal is the find the features affecting car prices and not necessarily to predict the car prices themselves
  - keeps the MSE's which is what we used to validate model correctness more manageable

## Modeling

Following modes were analyzed:

1. Model with PCA to form baseline MSE
    - manually find the ideal number of components by looking at singular values plot and find the 'elbow'
2. LinearRegression 
    - with all columns. Manually did a search to see which polynomial degree would yield best results. 
    - use SequentialFeatureSelector to select 6 columns
4. Ridge
    - use grid search to find optimal hyperparameter alpha (with polynomial degree fixed to 2) and 5 cross validation folds
    - use grid search to find optimal hyperparameters alpha for Ridge and degree for polynomial with 5 cross validation folds
5. Lasso
    - use grid search to find optimal alpha with degree=2 and 5 cross validation folds

Used StandardScaler with Pipeline for all of the analysis above

### Evaluation

With some modeling accomplished, we aim to reflect on what we identify as a high-quality model and what we are able to learn from this.  We should review our business objective and explore how well we can provide meaningful insight into drivers of used car prices.  Your goal now is to distill your findings and determine whether the earlier phases need revisitation and adjustment or if you have information of value to bring back to your client.

### Model evaluation

Summary table:

| Model      | Parameters | Results (MSE) |
| ----------- | ----------- | --------------------|
| PCA, LinearRegression      | 3 components, degree=4 |  MSE=1.165044104302023 |
| LinearRegression   | degree=2        | MSE=0.9219674281178614 |
| LinearRegression with SequentialSelector | degree=2, columns=6 | MSE=1.1457267976191932 |
| Ridge Regression | degree=2 | optimal alpha=0.1, MSE=0.9132654803829813 |
| Ridge Regression | Grid search for optimal alpha and degree | optimal alpha=0.1, degree=2, MSE=0.9132654803829813 |
| LASSO Regression | Grid search for optimal alpha, degree=2 | optimal alpha=0.1, MSE=0.9713301103156311 |


Discussion:

1. **PCA, LinearRegression**: Analyzed to find the optimal number of features by looking for the 'elbow' in the singular values plot. Found that 3 components was optimal. Applied LinearRegression with polynomial degree=3 to these 3 PCA's which resulted in **MSE = 1.165044104302023, degree = 4**. Used this as a baseline for further model evaluations
2. **LinearRegression**: was analyzed. Ran different models with varying polynomial degrees. Found optimal degree to be 2. LinearRegression with degree=2 polynomial resulted in: **Best mse = 0.9219674281178614, degree = 2**
3. **LinearRegression with SequentialSelector**:  Ran LinearRegression model with SequentialSelector to select 6 columns and polynomial degree=2. which resulted in: **Best mse = 1.1457267976191932, degree = 2**
4. **Ridge Regression**: GridSearch was done with 10 steps [10**-5 to 10**4) to search for optimal alpha for regularization. Ran with fixed degree=2 polynomial and cross validation with 5 folds. Results: **Best alpha = 0.1, best_mse = 0.9132654803829813.**
5. **Ridge Regression**: Did a grid search for finding an optimal degree and alpha with cross validation of 5 folds. Found that the **optimal degree=2 and alpha=0.1 as was analyzed in step 3 above.** This was a computationally intensive run and took almost 30 minutes on my Macbook PRO machine
6. **Lasso regression**: Lasso regression was done with grid search for optimal alpha and with cross validation with 5 folds. Results = **Best alpha = 1e-05 best mse = 0.9713301103156311**

Looking at these findings, the **best model** evaluated was **Ridge Regression with alpha=0.1, degree=2 with MSE= 0.91.** 
LinearRegression model with degree=2 was also very close to the optimal. As the plot for target price for test data above demonstrates
the LinearRegression model and the best_ridge_model tracks the actual target price better than other models

## Feature analysis with best model (Ridge with alpha=0.1, degree=2)

Best model in the analysis is the Ridge model, with alpha=0.1 and degree=2. We will do the following to analyze the model
1. Use sklearn's permutation_importance to determine the features impacting the target price the most
2. Look at the coefficients on the regression to detemine which features have the most positive and negative impacts 

### Deployment

Now that we've settled on our models and findings, it is time to deliver the information to the client.  You should organize your work as a basic report that details your primary findings.  Keep in mind that your audience is a group of used car dealers interested in fine-tuning their inventory.

**Findings/Recommendations**


**Summary**

The target price is most sensitive to the following features:

- gas fuel (more than diesel)
- full-size vehicles (more than other vehicle sizes)
- sedan (more than other types of vehicle)
- fwd drive (more than other drive types)

**Following features impact target price *positively*** (increases price):

- Condition of the vehicle (new condition is most favorable)
- hybrid fuel is preferred
- VAN, mini-van and SUV are the preferred type of vehicles
- vehicles with 6 cylinders
- automatic transmission
- 4wd


**Following features impact target prices *negatively*** (decreases price):

- how old the vehicle is (based on year column)
- rwd drive type
- manual transmission
- gas transmission
- sedan


Our **recommendation** is to stock up on vehicle which have 

- Automatic Transmission
- VAN, Minivan, SUV with 6 cylinders
- 4wd
- low mileage and in new/like-new condition
- low years

**Detailed Report**

The best model that predicted the target price looked at combination of 2 features, and also just the features by themselves

The model found following features also have most positive impact on the price of the vehicle

- New condition of the vehicle
- hybrid with less years
- VAN
- mini-van
- SUV
- 4wd
- 6 cylinders with automatic transmission
- automatic transmission
- gas fuel
- full-size with 6 cylinders
- electric vehicle
- rwd with 6 cylinders
- 4wd with 6 cylinders
- trucks with 6 cylinders
- wagon
- wagons with automatic transmission
- manual transmission with gas fuel type
- manual transmission with rwd
- fwd with 6 cylinders


The model found following features also have a negative impact on the price of the vehicle

- year of car
- 6 cylinders in general
- rwd
- manual transmission
- sub-compact
- bus
- gas cars
- sedans
- odometer
- hatchback
- compact

After evaluating multiple models, the following features of the car were found to be **statistically significant** with respect to the car price (in decreasing order of siginificance). This does not specify if the impact was positive or negative just that these features were important the the target price:

***Transmission***
- Automatic transmission is more important for the car price than manual transmission

***Size of car***
- full-size
- mid-size
- compact
- sub-compact

***Type of car***
- sedan
- SUV
- coupe
- truck
- pickup
- convertible
- wagon
- van
- mini-van

***Drive type***
- fwd
- 4wd
- rwd

***Condition***
- new, like new cars sold for more

***Odometer***
- number of miles was statistically significant

## Future work
- analyze the columns more and see if we can make use of them. For example manufacturer brand may have a bigger impact on car price
- Analyze different transformations of data - maybe there is some structure in the column values that can be utilized.
- Do a larger grid search to find optimal parameters for alpha and polynomial degree. This becomes quite computationally intensive so had to restrict the search to smaller space