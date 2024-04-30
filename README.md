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
* Bivariate EDA
  * visualization (bar-chart)
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

### Numerical features
* The EDA for numerical features are done bivarietely between 'zip' and 'isFraud' as a part of feature engineering as it will be explained below

## Feature engineering

### 'long', 'lat', 'merch_long', 'merch_lat'
* This part combines the four aforementioned features above into one singular feature called 'dist' using the Haversine formula

### 'dist'
* The new feature 'dist' was re-engineered into a binned feature 'dist_bin'
 * The binning is based on interval binning by first calculating the mean and std of 'dist' and then defining the edges of the according to this code:

```bin_edges = [0, mean - 2*std, mean + std, float('inf')]```
 * The bin labels are assinged as such:

```bin_labels = ['near', 'medium', 'far']```
 * Lastly the binned features are one-hot encoded
  * Note: for the subsequent features, everytime there is binning, there is also encoding

### 'category', 'state'
* 'category' was binned based on the intrinsic category of the values. In this case the first step was to combine the _pos (point of sale/offline purchase) and _net categories (online purchase) into a 'combined_category'. Afterwards it was split again into two categories namely 'essential_categories' and 'non_essential_categories' before encoding. This is done to prevent overdimensionality of this feature caused by encoding too many values of 'category'

```essential_categories = ['grocery_combined', 'health_fitness', 'personal_care', 'home', 'food_dining']```

```non_essential_categories = ['misc_combined', 'entertainment', 'gas_transport', 'shopping_combined', 'travel', 'kids_pets']```

*  'state' was also engineered into bins based on US census regions as it refers to [here](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf) before encoding. This is also to prevent overdimensionality of the feature
