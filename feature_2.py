#Libraries
import pandas as pd

#Training (891 Entries) & Testing (417 Entries) data
train_data = pd.read_csv('Data/train.csv')
test_data = pd.read_csv('Data/test.csv')
all_data = [train_data, test_data]
passenger_id = test_data['PassengerId']

#Feature Engineering
print( train_data[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean() )
