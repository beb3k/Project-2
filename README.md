# Project2
Credit card fraud prediction machine learning model

## Overview
This project concerns the building a predictive model to detect credit card fraud using a simulated dataset from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection). According to the author, this dataset is generated using Sparkov which is a data generation tool from [this github](https://github.com/namebrandon/Sparkov_Data_Generation). The dataset generated contains legitimate and fraud credit card transactions from Jan 1st 2019 to Dec 31st 2020 covering 1000 customers from 800 merchants.

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

`bin_edges = [0, mean - 2*std, mean + std, float('inf')]`

 * The bin labels are assinged as such:

`bin_labels = ['near', 'medium', 'far']`

 * Lastly the binned features are one-hot encoded
  * Note: for the subsequent features, everytime there is binning, there is also encoding

### 'category', 'state'
* 'category' was binned based on the intrinsic category of the values. In this case the first step was to combine the _pos (point of sale/offline purchase) and _net categories (online purchase) into a 'combined_category'. Afterwards it was split again into two categories namely 'essential_categories' and 'non_essential_categories' before encoding. This is done to prevent overdimensionality of this feature caused by encoding too many values of 'category'

```essential_categories = ['grocery_combined', 'health_fitness', 'personal_care', 'home', 'food_dining']```

```non_essential_categories = ['misc_combined', 'entertainment', 'gas_transport', 'shopping_combined', 'travel', 'kids_pets']```

*  'state' was also engineered into bins based on US census regions as it refers to [here](https://www2.census.gov/geo/pdfs/maps-data/maps/reference/us_regdiv.pdf) before encoding. This is also to prevent overdimensionality of the feature

### 'dob'
* the date of birth featured is used to infer a new feature 'age', and then 'age' was binned to 'age_group' with these labels

```bins = [18, 35, 58, 99]```

```labels = ['muda', 'paruhbaya', 'tua']```

The new feature 'age_group' was subsequently one-hot encoded

### 'zip'
This feature is arguably the most complicated to engineer as it involves multiple features to infer some information and to derive several new features

* First, some value and unique counting. It yields 970 unique values with zip value of 73754 appeared 3646 times
* Second, multivariate profiling of 'zip'. Aggregating 'amt' and 'isFraud' based on 'zip'. This aggregation yields several new features that is created on a separate dataset 'zip_profile'. For clarity please refer to this code snippet:

```# Step 1: Aggregasi data berdasar 'zip'
zip_profile = df.groupby('zip').agg(
    total_transactions=('amt', 'count'),
    total_amt=('amt', 'sum'),
    average_amt=('amt', 'mean'),
    fraud_transactions=('is_fraud', 'sum')
).reset_index()`
```

From this, the new feature 'fraud_rate' can be calculated

`zip_profile['fraud_rate'] = zip_profile['fraud_transactions'] / zip_profile['total_transactions']`

The value counts can of 'fraud_rate' can then be counted:

`value_counts_fraud_rate = zip_profile['fraud_rate'].value_counts().head(10)
print(value_counts_fraud_rate)`

A new feature called 'fraud_risk' can be created based on the calculation of 'fraud_rate' earlier. It was binned to the value of 0, >0 to <1, and 1. For clarity please refer to the code snipet below

```
def classify_fraud_risk(rate):
    if rate == 1:
        return 'high risk'
    elif rate == 0:
        return 'low risk'
    else:
        return 'medium risk'
```

This feature was also one-hot encoded

* Third, the value of 'fraud_risk' is mapped to every 'zip'. It was initially done on the 'zip_profile' datarame and then assigned to the original dataframe 'df'

```
# Mapping 'fraud_risk' untuk setiap 'zip' di dataframe 'zip_profile'
zip_profile['fraud_risk'] = zip_profile['fraud_rate'].apply(classify_fraud_risk)
zip_to_risk_mapping = zip_profile.set_index('zip')['fraud_risk'].to_dict()

# assign 'fraud_risk' dari dataframe 'zip_profile' ke dataframe awal 'df'
df['fraud_risk'] = df['zip'].map(zip_to_risk_mapping)
```

* Fourth, calculating the average transaction amount for every 'fraud_risk' that was assigned to every 'zip' into a new feature 'avg_trans_amt'. It was also initially done on 'zip_profile' and then assigned to 'df'. Please refer to the code snippet for brevity and clarity

```
# kalkulasi mean untuk bin fraud_rate (fraud_risk)
low_risk_avg_amt = zip_profile.loc[zip_profile['fraud_rate'] == 0, 'average_amt'].mean()
medium_risk_avg_amt = zip_profile.loc[(zip_profile['fraud_rate'] > 0) & (zip_profile['fraud_rate'] < 1), 'average_amt'].mean()
high_risk_avg_amt = zip_profile.loc[zip_profile['fraud_rate'] == 1, 'average_amt'].mean()

# mapping hasil kalkulasi ke setiap 'zip'
zip_profile['avg_trans_amt'] = zip_profile['fraud_rate'].apply(
    lambda x: high_risk_avg_amt if x == 1 else medium_risk_avg_amt if 0 < x < 1 else low_risk_avg_amt)

zip_avg_mapping = zip_profile.set_index('zip')['avg_trans_amt'].to_dict()

# Assign hasil mapping ke dataframe 'df'
df['avg_trans_amt'] = df['zip'].map(zip_avg_mapping)
```

* Fifth and final step is to calculate the average transaction count for every 'fraud_risk' into a new feature called 'trans_count_zip'. The same steps were taken as the above features

```
# Hitung mean untuk jumlah transaksi tiap kategori 'fraud_risk'
avg_trans_counts = zip_profile.groupby('fraud_risk')['total_transactions'].mean()

# Mapping hasil kalkulasi mean ke 'fraud_risk' untuk tiap 'zip'
zip_profile['trans_count_zip'] = zip_profile['fraud_risk'].map(avg_trans_counts)
```

## Modeling
