#Libraries
import pandas as pd
import numpy as np

#Training (891 Entries) & Testing (417 Entries) data
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')
all_data = [train_data, test_data]
passenger_id = test_data['PassengerId']

#Feature Engineering
for data in all_data:
    age_avg  = data['Age'].mean()
    age_std  = data['Age'].std()
    age_null = data['Age'].isnull().sum()

    random_list = np.random.randint(age_avg - age_std, age_avg + age_std , size = age_null)
    data['Age'][np.isnan(data['Age'])] = random_list
    data['Age'] = data['Age'].astype(int)

train_data['category_age'] = pd.cut(train_data['Age'], 5)
print( train_data[["category_age","Survived"]].groupby(["category_age"], as_index = False).mean() )
