import pandas as pd
import numpy as np
#import graphviz
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split,cross_val_score, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

"""
df = pd.read_csv('Wage.csv')

# Extract features and target variable
X = df['age']
y = df['wage']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
"""
"""
degrees = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
results = []
#X = np.array(X).reshape(-1, 1)
X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)
for x in degrees:
    model = make_pipeline(PolynomialFeatures(x), LinearRegression())
    model.fit(X_train, y_train)

    cv_scores = cross_val_score(model, X, y, cv=5)
    results.append(np.mean(cv_scores))

print(results)
best_degree = degrees[np.argmin(results)]
print("Best degree:", best_degree)

X_sorted = np.sort(X, axis=0)
model = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
model.fit(X, y)
y_train_pred = model.predict(X_sorted)
plt.scatter(X, y, label="training data")
plt.plot(X_sorted, y_train_pred, color='red', label='Fitted Line')
plt.xlabel('age')
plt.ylabel('wage')
#plt.show()
"""

"""
steps = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
#X = np.sort(X, axis=0)
scores = []
for i in steps:
    breaks = []
    step = 100/i
    for j in range(1, i):
        breaks.append(j * step)
    breakpoints = np.percentile(X, breaks)
    segments = np.digitize(X, breakpoints)
    segment_means = [np.mean(y[segments == k]) for k in range(1, len(breakpoints) + 2)]

    segments_test = np.digitize(X_test, breakpoints)
    y_pred = np.array([segment_means[i - 1] for i in segments_test])
    mae = np.mean(np.abs(y_test - y_pred))
    scores.append(mae)

print(scores)
bestSplit = steps[np.argmin(scores)]
print("Best Step:", bestSplit)
# Generate evenly spaced breakpoints
breakpoints = np.linspace(np.min(X), np.max(X), bestSplit + 1)

# Create step function segments
segments = np.digitize(X, breakpoints)

# Calculate mean for each segment
segment_means = [np.mean(y[segments == i]) for i in range(1, len(breakpoints) + 1)]

plt.scatter(X, y, label="Data")
plt.step(breakpoints, segment_means, where='post', color='red', label='Step Function')
plt.xlabel('age')
plt.ylabel('wage')
plt.legend()
plt.show()
"""
df = pd.read_csv('Carseats.csv')
df['Urban'] = df['Urban'].map({'Yes': 1, 'No': 0})
df['US'] = df['US'].map({'Yes': 1, 'No': 0})
df['ShelveLoc'] = df['ShelveLoc'].map({'Good': 2, 'Medium': 1, 'Bad': 0})
X = df[['CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'ShelveLoc', 'Age', 'Education', 'Urban', 'US']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
errors = []
for i in range(1, 15):
    tree_regressor = DecisionTreeRegressor(max_depth=i)
    tree_regressor.fit(X_train, y_train)
    y_pred = tree_regressor.predict(X_test)
    cv_scores = cross_val_score(tree_regressor, X, y, cv=10)
    errors.append(np.mean(cv_scores))
print(errors)
print(np.argmin(errors))

bagReg = BaggingRegressor(estimator=SVR(), n_estimators=8, random_state=0).fit(X_train, y_train)
y_BagPred = bagReg.predict(X_test)
mse = mean_squared_error(y_test, y_BagPred)
print(mse)

rf_reg = RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(X_train, y_train)
y_rfPred = rf_reg.predict(X_test)
mse = mean_squared_error(y_test, y_rfPred)
print(mse)
importance = rf_reg.feature_importances_
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i, v))

plt.bar([x for x in range(len(importance))], importance)
feature_names = X.columns
plt.xticks(range(len(importance)), feature_names, rotation='vertical')
plt.show()



"""
dot_data = export_graphviz(
    tree_regressor,
    out_file=None,
    feature_names=['feature_name'],  # Replace 'feature_name' with your actual feature name
    filled=True,
    rounded=True
)

graph = graphviz.Source(dot_data)
graph.render("regression_tree")  # This will save the tree visualization to a file called "regression_tree.pdf"
"""











