# importing dataset
import pandas as pd
import numpy as np

# a list with all missing value formats
missing_value_formats = ["n.a.","?","NA","n/a", "na", "--"]
df = pd.read_csv("Covid_data.csv", na_values = missing_value_formats)

# making all values either integer or NaN
def make_int(i):
    try:
        return int(i)
    except:
        return np.nan

# applying make_int function to the entire series using map
df['Sp02'] = df['Sp02'].map(make_int)
df['HR'] = df['HR'].map(make_int)
df['T'] = df['T'].map(make_int)
df['State'] = df['State'].map(make_int)

# dropping all rows with all NaN
df = df.dropna(axis = 0, how ='all')

# filling a column from the column of its previous row
df['Sp02'].fillna(method = 'pad', inplace = True)
df['HR'].fillna(method = 'pad', inplace = True)
df['T'].fillna(method = 'pad', inplace = True)
df['State'].fillna(method = 'pad', inplace = True)

# replacing negative num with positive num
num = df._get_numeric_data()
num[num < 0] = (-1)*num

# dropping duplicates
df.drop_duplicates()

# creating the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error

X = df.drop(columns = ['State'])
y = df['State']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

predictions = decision_tree.predict(X_test)
score = accuracy_score(y_test, predictions)
model_r2_score = r2_score(y_true=y_test, y_pred=predictions)
model_MAE = mean_absolute_error(y_true=y_test, y_pred=predictions)
model_EVS = explained_variance_score(y_true=y_test, y_pred=predictions)

# saving the model
import pickle
pickle.dump(decision_tree, open('covid_model.pkl','wb'))

# loading the model that I just saved
model = pickle.load(open('covid_model.pkl','rb'))
print(model.predict([[98, 80, 37]]))