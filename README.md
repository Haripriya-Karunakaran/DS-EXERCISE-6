# DS-EXERCISE-6

# FEATURE TRANSFORMATION 

## AIM: 

To read the given data and perform Feature Transformation process and save the data to a file. 

## EXPLANATION:

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a
mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis. 

## ALGORITHM: 

Step 1: Read the given data. 

Step 2: Clean the Data Set using Data Cleaning Process.

Step 3: Apply Feature Transformation techniques to all the feature of the data set. 

Step 4: Save the data to the file.

## CODE:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

df = pd.read_csv("/content/Data_to_Transform.csv")

print(df)

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df1['Highly Positive Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df1['Highly Negative Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df1['Moderate Positive Skew'],fit=True,line='45')

plt.show()

sm.qqplot(df1['Moderate Negative Skew'],fit=True,line='45')

plt.show()

df1['HighlyPositiveSkew'] = np.log(df1['Highly Positive Skew'])

sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')

plt.show()

df2 = df.copy()

df2['Highly Positive Skew'] = 1/df2['Highly Positive Skew']

sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')

plt.show()

df3 = df.copy()

df3['Highly Positive Skew'] = df3['Highly Positive Skew']**(1/1.2)

sm.qqplot(df2['Highly Positive Skew'],fit=True,line='45')

plt.show()

df4 = df.copy()

df4['Moderate Positive Skew_1'],parameters =stats.yeojohnson(df4['Moderate Positive Skew'])

sm.qqplot(df4['Moderate Positive Skew_1'],fit=True,line='45')

plt.show()

from sklearn.preprocessing import PowerTransformer

trans = PowerTransformer("yeo-johnson")

df5 = df.copy()

df5['Moderate Negative Skew_1'] = pd.DataFrame(trans.fit_transform(df5[['Moderate Negative Skew']]))

sm.qqplot(df5['Moderate Negative Skew_1'],line='45')

plt.show()

from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(output_distribution = 'normal')

df5['Moderate Negative Skew_2'] = pd.DataFrame(qt.fit_transform(df5[['Moderate Negative Skew']]))

sm.qqplot(df5['Moderate Negative Skew_2'],line='45')

plt.show()

## OUTPUT:
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/a3b87712-4b54-47f5-8e91-85f4e27ecf0f)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/709b31f0-c761-42b3-93fe-eb3ca24136d5)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/5ab42e67-f59f-4ed0-9516-3f4269144cd7)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/39b4f483-9074-4579-ae72-d8c881471fd7)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/4c4da213-2b64-4656-9fb9-adf22d51323c)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/cd05efc8-c972-4b99-a527-d3e895ce5403)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/cb7229d9-e4c3-4607-8517-2cd60ca8e7cc)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/6e0af26f-49b4-417b-8d6b-29be99dcf524)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/945156ec-8bc8-4c75-a2cf-e844b00eaccd)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/a0dddac0-3ef7-429a-bb20-f6f69b5e9a95)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/09b7d2f2-c9e1-4716-bb36-f158b9fc79f7)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/bb9aacf0-6116-46ac-a20a-d6653f03a4b4)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/186d5e05-75ea-4c88-bc7b-4829e77ae353)
![image](https://github.com/Haripriya-Karunakaran/DS-EXERCISE-6/assets/126390051/8c39b76d-19a3-433e-8f52-f73816a8c96a)

## RESULT:

Thus, the Feature Transformation for the given datasets had been executed successfully

