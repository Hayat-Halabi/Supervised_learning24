# Supervised_learning24
Demo solutions w/ downloadable link
``` python

#Import Pandas and NumPy libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#Step 2: Load the Dataset
#Load the titanic dataset and display the first few rows.
#Return a description of the dataset using the describe() method

df = pd.read_csv('titanic.csv', sep= ',')
df.head()

#Observation: Here, we have Pclass, Name, Sex, Age, SibSp(sibling and spouse), Parch(parent and child), Ticket.
df.describe()
#Observations: Here, the mean and median are not applicable for PassengerId. 29 is the mean age, and 80 is the maximum #age. Similarly, 32 is the mean, and 512 is the max for Fare.
df.info()

#Observation: Here, both non-null counts, as well as the data type, is printed.

#Step 3: Data Preprocessing

# if df['SibSp'] + df['Parch']) > 0, then the passenger is NOT traveling alone - travelalone value = 0 (False) 
# else the passenger is traveling alone - travelalone value = 1 (True) 

df['Travelalone'] =  np.where((df['SibSp'] + df['Parch']) > 0, 0 , 1).astype('uint8')

#Observation: Now that a Travelalone attribute is created, let's drop the unnecessary data columns and check if there # are any missing values.

df1 = df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis =1)
df.isna().sum()

#Observation: As we can see, there are missing values in age and cabin.

df1.isna().sum()

#Observation: There are missing values in age, so let's replace them.

df1['Age'].fillna(df1['Age'].median(skipna=True), inplace=True)
df1.head()
df1.isna().sum()

# Observation: There are categorical variables like Sex and Embarked.

df_titantic = pd.get_dummies(df1, columns= ['Pclass', 'Embarked', 'Sex'],drop_first= True)
df_titantic.head()
df.head()

# Observation:

# - Hence, the dummy values are created.
#   Categorical data cannot typically be directly handled by machine learning algorithms, as most algorithms are
#   primarily designed to operate with numerical data only.
# - Therefore, before categorical features can be used as inputs to machine learning algorithms, they must be encoded
#   as numerical values
# - Dummy variables are useful because they allow us to include categorical variables in our analysis.

#Step 4: Prepare Features and Target Variables
# - Let’s try to preprocess the data and create a scaler to standardize the data points

# - Create x and y values for the same

X = df_titantic.drop(['Survived'], axis =1)
y = df_titantic['Survived']

# Now, let's import the MinMaxScaler and StandardScaler.

# Then, transpose that and call it as MM or MinMaxScaler.

# Finally, transfer and print the data.

from sklearn.preprocessing import MinMaxScaler, StandardScaler
trans_MM = MinMaxScaler()
trans_SS = StandardScaler()

# normalizing our data with MinMaxScaler()

df_MM = trans_MM.fit_transform(X)
pd.DataFrame(df_MM)

#Observations:

# - For Age and Fare in MinScaler, the values have changed from 0 to 1.

# - Some of the attributes are also in the range of 0 to 1.

# - MinMaxScaler will be using the X values. For example: one value is equal to X minus Xmin divided by Xmax by Xmin.

# - For StandardScaler, we take the origin minus the mean of the distribution divided by the standard deviation of the distribution.

# Now, let's check the same for StandardScaler and see how the data varies.

# Standardizing a dataset involves rescaling the distribution of values so that the mean of observed values is 0 and 
# the standard deviation is 1

df_SS = trans_SS.fit_transform(X)
pd.DataFrame(df_SS)

# Observations:

# - As you can see, the data has negative values too, and it converted the dummies we created.

# - It always depends on the case whether you want to consider MinMaxScalar or Standardscalar.

# Now let's split the dataset, perform hyperparameter tuning, and build the model.

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
# Creating a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Define a dictionary named param_grid for specifying hyperparameter options.
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# Use Grid Search to tune the hyperparameters

# Performing grid search with cross-validation
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)


#Extract the ideal hyperparameters : Grid search will return the best hyperparameters for the model.
param_grid

# Getting the best hyperparameters from grid search
best_params = grid_search.best_params_ 
best_params

#Rebuild the model with best hyperparameters

# Creating a Decision Tree Classifier with the best hyperparameters
best_dt_classifier = DecisionTreeClassifier(**best_params, random_state=42)

# Fit the training data into the optimized model

# Fitting the model on the training data
best_dt_classifier.fit(X_train, y_train)

#Evaluate the model's performance (accuracy) on the testing dataset

# Evaluating the model on the testing data
accuracy = best_dt_classifier.score(X_test, y_test)
print("Accuracy of the Decision Tree Classifier:", accuracy)

```
Downloadable question file+ Demo solution above
```
https://vocproxy-1-21.us-west-2.vocareum.com/files/home/labsuser/Block_24_Demo_Student.ipynb?_xsrf=2%7C3db90129%7Cacba5a54690658c9a4998717e4af4654%7C1708812522
```
