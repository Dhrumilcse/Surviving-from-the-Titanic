#Libraries
import pandas as pd

#Training (891 Entries) & Testing (417 Entries) data
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')
all_data = [train_data, test_data]
passenger_id = test_data['PassengerId']

#Feature Engineering
for data in all_data:
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

train_data['category_fare'] = pd.qcut(train_data['Fare'], 4)
print( train_data[["category_fare","Survived"]].groupby(["category_fare"], as_index = False).mean() )
