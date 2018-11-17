import csv
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#extract accident_id, 1st_road_class, speed_limit, road_surface_conditions, target
accident_columns = ["accident_id", "1st_road_class", "speed_limit", "road_surface_conditions","time"]
vehicle_columns = ["accident_id","Vehicle_Type","Age_of_Driver"] #note, age of driver isn't always available

accidents_train = pd.read_csv("data/train.csv", usecols=accident_columns+["target"])
accidents_unseen = pd.read_csv("data/test.csv", usecols=accident_columns)
vehicles = pd.read_csv("data/vehicles.csv", usecols=vehicle_columns)

#split features and target
target = pd.factorize(accidents_train["target"])[0]
train = accidents_train.drop(axis=1,labels=["target"])

#---------------------preproccesing--------------------------#

#join and drop empty values (driver age)
print(accidents_unseen.shape)
train = train.merge(vehicles, on='accident_id',suffixes=('_accidents','_vehicles')).fillna(999)
unseen = accidents_unseen.merge(vehicles, on='accident_id',suffixes=('_accidents','_vehicles')).fillna(999)
print(train.shape)
#create 24 hour buckets
train['time'] = train['time'].apply(lambda x: int(x.split(":")[0]))
unseen['time'] = unseen['time'].apply(lambda x: int(x.split(":")[0]))
print(train.shape)

#create age buckets
train['age_unknown'] = train['Age_of_Driver'].apply(lambda x: 1 if x == 999 else 0)
train['age_0_20'] = train['Age_of_Driver'].apply(lambda x: 1 if int(x)<20 else 0)
train['age_20_30'] = train['Age_of_Driver'].apply(lambda x: 1 if int(x)<30 and int(x)>=20 else 0)
train['age_30_40'] = train['Age_of_Driver'].apply(lambda x: 1 if int(x)<40 and int(x)>=30 else 0)
train['age_40_50'] = train['Age_of_Driver'].apply(lambda x: 1 if int(x)<50 and int(x)>=40 else 0)
train['age_50_60'] = train['Age_of_Driver'].apply(lambda x: 1 if int(x)<60 and int(x)>=50 else 0)
train['age_60_70'] = train['Age_of_Driver'].apply(lambda x: 1 if int(x)<70 and int(x)>=60 else 0)
train['age_70_80'] = train['Age_of_Driver'].apply(lambda x: 1 if int(x)<80 and int(x)>=70 else 0)
train['age_80_90'] = train['Age_of_Driver'].apply(lambda x: 1 if int(x)<90 and int(x)>=80 else 0)
train['age_90_'] = train['Age_of_Driver'].apply(lambda x: 1 if int(x)>=90 else 0)
train = train.drop(axis=1,labels="Age_of_Driver")

unseen['age_unknown'] = unseen['Age_of_Driver'].apply(lambda x: 1 if x == 999 else 0)
unseen['age_0_20'] = unseen['Age_of_Driver'].apply(lambda x: 1 if int(x)<10 else 0)
unseen['age_20_30'] = unseen['Age_of_Driver'].apply(lambda x: 1 if int(x)<30 and int(x)>=20 else 0)
unseen['age_30_40'] = unseen['Age_of_Driver'].apply(lambda x: 1 if int(x)<40 and int(x)>=30 else 0)
unseen['age_40_50'] = unseen['Age_of_Driver'].apply(lambda x: 1 if int(x)<50 and int(x)>=40 else 0)
unseen['age_50_60'] = unseen['Age_of_Driver'].apply(lambda x: 1 if int(x)<60 and int(x)>=50 else 0)
unseen['age_60_70'] = unseen['Age_of_Driver'].apply(lambda x: 1 if int(x)<70 and int(x)>=60 else 0)
unseen['age_70_80'] = unseen['Age_of_Driver'].apply(lambda x: 1 if int(x)<80 and int(x)>=70 else 0)
unseen['age_80_90'] = unseen['Age_of_Driver'].apply(lambda x: 1 if int(x)<90 and int(x)>=80 else 0)
unseen['age_90_'] = unseen['Age_of_Driver'].apply(lambda x: 1 if int(x)>=90 else 0)
unseen = unseen.drop(axis=1,labels="Age_of_Driver")

#use one hot encoding for categorical variables
train = pd.get_dummies(train)
unseen = pd.get_dummies(unseen)

print(train.shape)
#group by accident_id

train = train.groupby(['accident_id']).agg('sum').reset_index()
print(train.shape)



unseen = unseen.groupby(['accident_id','speed_limit','time']).agg('sum').reset_index()
print(train.shape)

#split train and test
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=0)


#fit model
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(X_train, y_train)


#evaluate model
print(f1_score(y_test, clf.predict(X_test)))


for item in list(zip(train, clf.feature_importances_)):
	print("Feature: {0}			weight: {1}".format(item[0],item[1]))


#predict unseens
predictions = clf.predict(unseen)

print(predictions.shape)

with open("result/submission.csv","w") as submission:
	writer=csv.writer(submission)
	writer.writerow(["accident_id","target"])
	for idx, val in enumerate(unseen['accident_id']):
		writer.writerow([val,predictions[idx]])
