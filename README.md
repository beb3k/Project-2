# Project2
Credit card fraud prediction machine learning model

## Overview
This project concerns the building a predictive model to detect fraud using a simulated dataset from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection). According to the author, this dataset is generated using Sparkov which is a data generation tool from [this github](https://github.com/namebrandon/Sparkov_Data_Generation). The dataset generated contains legitimate and fraud transaction from Jan 1st 2019 to Dec 31st 2020 covering 1000 customers from 800 merchants.

## EDA
- The dataset contains 21 features with 1296675 datapoints
- Initial feature selection is done from my own best judgement by dropping these columns using this code

```dropped_columns = [['Unnamed: 0', 'cc_num', 'merchant', 'first', 'last', 'gender', 'street', 'city', 'job', 'trans_num']]```

### Categorical features
* EDA for categorical features are done for 'category' and 'state'
  * .value_counts()
  * .nunique()
  * Pie-chart visualization of 'category' and 'state'
* Bivariate visualization (bar-chart)
  * 'category' vs 'isFraud' value of class 0
  * 'category' vs 'isFraud' value of class 1
  * 'state' vs 'isFraud' value of class 0
  * 'state' vs 'isFraud' value of class 0
From this categorical features EDA there are some conclusion that can be made
* The data is highly imbalanced between class 0 and 1 target variable 'isFraud'
* Highest frequency 'category' value is gas_transport with 10.2% of total transaction
* Highest frequency 'state' values is 'TX' which is the state of Texas with 7.3% of total transaction
* The highest fraud number in 'category' is 'grocery_pos' with 1743 fraud
* The highest fraud proportion in 'category' is 'shopping_net' with 1.8% of total transaction of that particular category
* The highest fraud number in 'state' is in 'TX' with 479 fraud
* The highest fraud proportion in 'state' is in 'DE' with 100% of total transaction in that state


  ## Feature engineering
