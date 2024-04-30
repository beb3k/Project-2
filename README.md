# Project2
Fraud prediction machine learning model

## Overview
This project concerns the building a predictive model to detect fraud using a simulated dataset from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection). According to the author this dataset is generated using Sparkov which is a data generation tool from [this github](https://github.com/namebrandon/Sparkov_Data_Generation). The dataset generated contains legitimate and fraud transaction from Jan 1st 2019 to Dec 31st 2020 covering 1000 customers from 800 merchants.

## EDA
- The dataset contains 21 features with 1296675 datapoints
- Initial feature selection is done from my own best judgement by dropping these columns

dropped_columns = [['Unnamed: 0', 'cc_num', 'merchant', 'first', 'last', 'gender', 'street', 'city', 'job', 'trans_num']]
