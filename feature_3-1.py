#Libraries
import pandas as pd

#Training (891 Entries) & Testing (417 Entries) data
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')
all_data = [train_data, test_data]
passenger_id = test_data['PassengerId']

#Feature 3: Family Size
for data in all_data:
    data['family_size'] = data['SibSp'] + data['Parch'] + 1

#Feature 3.1: Is alone?
for data in all_data:
    data['is_alone'] = 0
    data.loc[data['family_size'] == 1, 'is_alone'] = 1
print (train_data[['is_alone', 'Survived']].groupby(['is_alone'], as_index=False).mean())
