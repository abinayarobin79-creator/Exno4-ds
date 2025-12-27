# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income.csv",na_values=[ " ?"])
data

<img width="1037" height="449" alt="Screenshot 2025-12-27 212953" src="https://github.com/user-attachments/assets/7c74a52a-7c4d-4cb7-b408-408a2594b64d" />

data.isnull().sum()

<img width="248" height="746" alt="Screenshot 2025-12-27 213017" src="https://github.com/user-attachments/assets/c8a98865-4ca9-4f4d-a93e-58843f0a61c7" />

missing=data[data.isnull().any(axis=1)]
missing

<img width="1025" height="440" alt="Screenshot 2025-12-27 213037" src="https://github.com/user-attachments/assets/1c6a9b7c-9d7b-4b6f-afa7-761913e9f2bc" />

data2=data.dropna(axis=0)
data2

<img width="1039" height="454" alt="Screenshot 2025-12-27 213054" src="https://github.com/user-attachments/assets/601b8afe-99e9-4dbc-93c8-9d81b19211a4" />

sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

<img width="504" height="315" alt="Screenshot 2025-12-27 213104" src="https://github.com/user-attachments/assets/b0573ad5-5584-4c9d-a5d2-a432bcb66736" />

sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs

<img width="503" height="634" alt="Screenshot 2025-12-27 213117" src="https://github.com/user-attachments/assets/c5b1fcd5-b3c6-4e51-b066-d753352a7aa9" />

data2

<img width="996" height="349" alt="Screenshot 2025-12-27 213130" src="https://github.com/user-attachments/assets/cb6fb64e-901c-4129-821d-8185a5d3b49c" />

new_data=pd.get_dummies(data2, drop_first=True)
new_data

<img width="1029" height="355" alt="Screenshot 2025-12-27 213142" src="https://github.com/user-attachments/assets/5d53838f-5f9b-47e9-bfbf-b85b4c0dc922" />

columns_list=list(new_data.columns)
print(columns_list)

<img width="1030" height="340" alt="Screenshot 2025-12-27 213151" src="https://github.com/user-attachments/assets/a5582fa0-8045-4ffc-9411-4c9b817ae236" />

features=list(set(columns_list)-set(['SalStat']))
print(features)

<img width="1034" height="345" alt="Screenshot 2025-12-27 213200" src="https://github.com/user-attachments/assets/d24d19a2-8899-49e8-a2f9-3db6be60fde7" />

y=new_data['SalStat'].values
print(y)

<img width="219" height="49" alt="Screenshot 2025-12-27 213207" src="https://github.com/user-attachments/assets/d1ce6bd7-40a2-42fd-a1f2-258f0dca55b7" />

x=new_data[features].values
print(x)

<img width="495" height="196" alt="Screenshot 2025-12-27 213228" src="https://github.com/user-attachments/assets/31986725-e7b5-47d0-9fa1-cfe00556e29c" />


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)

<img width="267" height="34" alt="Screenshot 2025-12-27 213235" src="https://github.com/user-attachments/assets/1c48d7a4-30a2-41d0-b0f1-6c2f0a174d5b" />

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)

<img width="161" height="63" alt="Screenshot 2025-12-27 213243" src="https://github.com/user-attachments/assets/d1fe4a93-64b0-42b0-b7c9-cf3c7f77fe32" />

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

<img width="231" height="30" alt="Screenshot 2025-12-27 213251" src="https://github.com/user-attachments/assets/ad79de86-6d6b-49f5-9719-deb9febaf674" />

print("Misclassified Samples : %d" % (test_y !=prediction).sum())

<img width="339" height="39" alt="Screenshot 2025-12-27 213257" src="https://github.com/user-attachments/assets/059f9058-fa2b-4cec-bcbd-247798104060" />

data.shape

<img width="143" height="38" alt="Screenshot 2025-12-27 213304" src="https://github.com/user-attachments/assets/b7f55036-9763-4ecb-b05d-6f5e648257d2" />

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)

<img width="412" height="58" alt="Screenshot 2025-12-27 213313" src="https://github.com/user-attachments/assets/a81aa298-ae5c-44a8-b9b9-e48650bf5ae2" />

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

<img width="512" height="234" alt="Screenshot 2025-12-27 213322" src="https://github.com/user-attachments/assets/f97b75df-ade9-4e01-9881-170f9ee810e5" />

tips.time.unique()

<img width="508" height="64" alt="Screenshot 2025-12-27 213330" src="https://github.com/user-attachments/assets/e4c1b043-2ebb-4f97-b021-f21134e4493b" />

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

<img width="258" height="103" alt="Screenshot 2025-12-27 213337" src="https://github.com/user-attachments/assets/75b9062e-2545-47d2-9094-5a0e8e76167a" />

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")

<img width="486" height="64" alt="Screenshot 2025-12-27 213345" src="https://github.com/user-attachments/assets/323ba2e9-0227-4b00-a53e-29bd1ec89107" />

# RESULT:

Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
