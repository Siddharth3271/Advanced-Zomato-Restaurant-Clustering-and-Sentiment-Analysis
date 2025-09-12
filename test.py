#Plotting graph
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

# Assume clf_grid and xgb_grid are your trained GridSearch objects
# and X_test, y_test are your test data
# Let's create some dummy data to make the example run
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression()
xgb = XGBClassifier()

clf_grid = GridSearchCV(clf, {'C': [1]}, cv=2)
xgb_grid = GridSearchCV(xgb, {'n_estimators': [100]}, cv=2)

clf_grid.fit(X_train, y_train)
xgb_grid.fit(X_train, y_train)

# finding the best parameters for all the models
log_reg_best = clf_grid.best_estimator_
xgbc_best = xgb_grid.best_estimator_

# predicting the sentiment by all models
y_preds_proba_lr = log_reg_best.predict_proba(X_test)[:, 1]
y_preds_proba_xgbc = xgbc_best.predict_proba(X_test)[:, 1]

classifiers_proba = [(log_reg_best, y_preds_proba_lr), (xgbc_best, y_preds_proba_xgbc)]

# Define a list to store the results
results = []

# Train the models and record the results
for pair in classifiers_proba:
    fpr, tpr, _ = roc_curve(y_test, pair[1])
    auc = roc_auc_score(y_test, pair[1])

    results.append({
        'classifiers': pair[0].__class__.__name__,
        'fpr': fpr,
        'tpr': tpr,
        'auc': auc
    })

# Create the DataFrame from the list of dictionaries
result_table = pd.DataFrame(results)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

# ploting the roc auc curve for all models
fig = plt.figure(figsize=(10,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], result_table.loc[i]['tpr'], label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

plt.plot([0,1], [0,1],'r--')

plt.xlabel("False Positive Rate", fontsize=15)
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC AUC Curve', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()