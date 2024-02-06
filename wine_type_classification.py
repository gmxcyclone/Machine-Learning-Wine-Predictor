import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import pickle as pkl

def train_evaluate(model, params, X_train, y_train, X_test, y_test, is_classifier=True):
    grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error' if not is_classifier else 'accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
    print(f'Cross-Validated Scores: {cv_scores}')
    print(f'Average CV Score: {np.mean(cv_scores):.4f}')

    predictions = best_model.predict(X_test)
    if is_classifier:
        accuracy = np.mean(predictions == y_test)
        print(f'Test Accuracy: {accuracy:.4f}')
    else:
        mse = mean_squared_error(y_test, predictions)
        print(f'Test MSE: {mse:.4f}')

    return best_model

# Function to prepare data
def prepare_data(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)



def prep(train, test, target):
    return train.drop(target,axis=1), test.drop(target,axis=1), train[target], test[target]


train = pd.read_csv('splits/combined_train.csv')
train.drop('quality', axis=1, inplace=True)
test = pd.read_csv('splits/combined_test.csv')
test.drop('quality', axis=1, inplace=True)

X_train, X_test, y_train, y_test = prep(train, test, 'red_wine')


#X_train, X_test, y_train, y_test = prepare_data(wine_data, 'red_wine')

rf_classifier_params = {'n_estimators': [25, 50, 75, 100, 150, 200], 'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 5]}
rfc = RandomForestClassifier(random_state=42)
best_model = train_evaluate(rfc, rf_classifier_params, X_train, y_train, X_test, y_test, is_classifier=True)

with open('model_objs/wine_type_classifier.pkl', 'wb') as f:
	pkl.dump(best_model, f)
