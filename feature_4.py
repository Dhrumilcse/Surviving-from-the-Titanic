#Libraries
import pandas as pd

#Training (891 Entries) & Testing (417 Entries) data
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')
all_data = [train_data, test_data]
passenger_id = test_data['PassengerId']

#Feature Engineering
for data in all_data:
    data['Embarked'] = data['Embarked'].fillna('S')
print( train_data[["Embarked","Survived"]].groupby(["Embarked"], as_index = False).mean() )
