import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Titanic dataset
titanic_data = sns.load_dataset('titanic')

# Clean the Titanic dataset
td = titanic_data.copy()
td.drop(['alive', 'who', 'adult_male', 'class', 'embark_town', 'deck'], axis=1, inplace=True)
td.dropna(inplace=True)
td['sex'] = td['sex'].apply(lambda x: 1 if x == 'male' else 0)
td['alone'] = td['alone'].apply(lambda x: 1 if x == True else 0)

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(td[['embarked']])
onehot = enc.transform(td[['embarked']]).toarray()
cols = ['embarked_' + val for val in enc.categories_[0]]
td[cols] = pd.DataFrame(onehot)
td.drop(['embarked'], axis=1, inplace=True)

# Train the Titanic dataset
X = td.drop('survived', axis=1)
y = td['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict survival probability for a new passenger
passenger = pd.DataFrame({
    'name': ['John Mortensen'],
    'pclass': [2],
    'sex': ['male'],
    'age': [64],
    'sibsp': [1],
    'parch': [1],
    'fare': [16.00],
    'embarked': ['S'],
    'alone': [False]
})

new_passenger = passenger.copy()
new_passenger['sex'] = new_passenger['sex'].apply(lambda x: 1 if x == 'male' else 0)
new_passenger['alone'] = new_passenger['alone'].apply(lambda x: 1 if x == True else 0)

onehot = enc.transform(new_passenger[['embarked']]).toarray()
cols = ['embarked_' + val for val in enc.categories_[0]]
new_passenger[cols] = pd.DataFrame(onehot, index=new_passenger.index)
new_passenger.drop(['name', 'embarked'], axis=1, inplace=True)

dead_proba, alive_proba = np.squeeze(logreg.predict_proba(new_passenger))

print('Death probability: {:.2%}'.format(dead_proba))  
print('Survival probability: {:.2%}'.format(alive_proba))

# Improve chances
importances = dt.feature_importances_
for feature, importance in zip(new_passenger.columns, importances):
    print(f'The importance of {feature} is: {importance}')

# Evaluate model accuracy
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print('DecisionTreeClassifier Accuracy: {:.2%}'.format(accuracy_dt))

y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print('LogisticRegression Accuracy: {:.2%}'.format(accuracy_logreg))
