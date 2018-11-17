import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#extract accident_id, 1st_road_class, speed_limit, road_surface_conditions, target
accident_columns = ["accident_id", "1st_road_class", "speed_limit", "road_surface_conditions"]
vehicle_columns = ["accident_id","Vehicle_Type","Age_of_Driver"] #note, age of driver isn't always available

accidents_train = pd.read_csv("data/train.csv", usecols=accident_columns+["target"])
accidents_unknown = pd.read_csv("data/test.csv", usecols=accident_columns)
vehicles = pd.read_csv("data/vehicles.csv", usecols=vehicle_columns)

#join and drop empty values (driver age)
train = accidents_train.merge(vehicles, on='accident_id',suffixes=('_accidents','_vehicles')).dropna()
unknown = accidents_unknown.merge(vehicles, on='accident_id',suffixes=('_accidents','_vehicles')).dropna()

#split features and target
target = pd.factorize(train["target"])[0]
train = train.drop(axis=1,labels=["target","accident_id"])

#use one hot encoding for categorical variables
train = pd.get_dummies(train)
unknown = pd.get_dummies(unknown).drop(axis=1,labels="accident_id")

#split train and test
X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.3, random_state=0)

clf = RandomForestClassifier(n_jobs=2, random_state=0)

clf.fit(X_train, y_train)

print(clf.score(X_test,y_test))

#drop values where driver's age isn't available

for item in list(zip(train, clf.feature_importances_)):
	print("Feature: {0}			weight: {1}".format(item[0],item[1]))