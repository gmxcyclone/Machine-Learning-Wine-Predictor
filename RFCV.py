import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix
import pickle as pkl


def train_evaluate(model, params, X_train, y_train, X_test, y_test, is_classifier=True,):
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
        mtx = confusion_matrix(y_test, predictions)


    else:
        mse = mean_squared_error(y_test, predictions)
        print(f'Test MSE: {mse:.4f}')
        fig,ax = plt.subplots()
        min_val = min([min(y_test), min(predictions)])
        max_val = max([max(y_test), max(predictions)])
        ax.plot([min_val-1, max_val+1], [min_val-1, max_val+1], 'k--')
        ax.scatter(y_test,predictions, c='b', marker='s', edgecolors='k')
        ax.set_xlim([min_val-1, max_val+1])
        ax.set_ylim([min_val-1, max_val+1])
        ax.set_xlabel('Actual Quality')
        ax.set_ylabel('Predicted Quality')
        ax.set_box_aspect(1)
        plt.show()

    return best_model

# Function to prepare data
def prepare_data(df, target):
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function to plot feature importance
def plot_feature_importance(model, X_train, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title(f"Feature importances for {title}")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()

'''

red_wine = pd.read_csv('Data/winequality-red_clean.csv')
white_wine = pd.read_csv('Data/winequality-white_clean.csv')
combined_wine = pd.read_csv('Data/winequality-combined_clean.csv')



combined_wine['red_wine'] = combined_wine['red_wine'].apply(lambda x: 1 if x else 0)


X_red_train, X_red_test, y_red_train, y_red_test = prepare_data(red_wine, 'quality')
X_white_train, X_white_test, y_white_train, y_white_test = prepare_data(white_wine, 'quality')
X_combined_train, X_combined_test, y_combined_train, y_combined_test = prepare_data(combined_wine, 'quality')
'''

def prep(train, test, target):
    return train.drop(target,axis=1), test.drop(target,axis=1), train[target], test[target]

red_train = pd.read_csv('splits/red_train.csv')
red_test = pd.read_csv('splits/red_test.csv')
X_red_train, X_red_test, y_red_train, y_red_test = prep(red_train, red_test, 'quality')

white_train = pd.read_csv('splits/white_train.csv')
white_test = pd.read_csv('splits/white_test.csv')
X_white_train, X_white_test, y_white_train, y_white_test = prep(white_train, white_test, 'quality')

combined_train = pd.read_csv('splits/combined_train.csv')
combined_test = pd.read_csv('splits/combined_test.csv')
X_combined_train, X_combined_test, y_combined_train, y_combined_test = prep(combined_train, combined_test, 'quality')


rf_classifier_params = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_split': [2, 5]}
rf_regressor_params = {'n_estimators': [100, 200], 'max_depth': [10, 20], 'min_samples_leaf': [1, 2]}

#rf_classifier_params = {'n_estimators': [25, 50, 75, 100, 150, 200], 'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 5]}
#rf_classifier_params = {'n_estimators': [25, 50, 75, 100, 150, 200], 'max_depth': [5, 10, 15, 20], 'min_samples_split': [1, 2, 5]}


# Initialize models
rf_classifier = RandomForestClassifier(random_state=42)
rf_regressor = RandomForestRegressor(random_state=42)

# Training and evaluation
print("\nRed Wine Classification:")
best_rf_classifier_red = train_evaluate(rf_classifier, rf_classifier_params, X_red_train, y_red_train, X_red_test, y_red_test)
with open('model_objs/red_classification.pkl', 'wb') as f:
    pkl.dump(best_rf_classifier_red, f)

print("\nWhite Wine Classification:")
best_rf_classifier_white = train_evaluate(rf_classifier, rf_classifier_params, X_white_train, y_white_train, X_white_test, y_white_test)
with open('model_objs/white_classification.pkl', 'wb') as f:
    pkl.dump(best_rf_classifier_white, f)

print("\nCombined Wine Classification:")
best_rf_classifier_combined = train_evaluate(rf_classifier, rf_classifier_params, X_combined_train, y_combined_train, X_combined_test, y_combined_test)
with open('model_objs/comb_classification.pkl', 'wb') as f:
    pkl.dump(best_rf_classifier_combined, f)

print("\nRed Wine Regression:")
best_rf_regressor_red = train_evaluate(rf_regressor, rf_regressor_params, X_red_train, y_red_train, X_red_test, y_red_test, is_classifier=False)
with open('model_objs/red_regressor.pkl', 'wb') as f:
    pkl.dump(best_rf_regressor_red, f)
    
print("\nWhite Wine Regression:")
best_rf_regressor_white = train_evaluate(rf_regressor, rf_regressor_params, X_white_train, y_white_train, X_white_test, y_white_test, is_classifier=False)
with open('model_objs/white_regressor.pkl', 'wb') as f:
    pkl.dump(best_rf_regressor_white, f)

print("\nCombined Wine Regression:")
best_rf_regressor_combined = train_evaluate(
    rf_regressor, rf_regressor_params, X_combined_train, y_combined_train, X_combined_test, y_combined_test, is_classifier=False
)
with open('model_objs/comb_regressor.pkl', 'wb') as f:
    pkl.dump(best_rf_regressor_combined, f)

plot_importances = False

if plot_importances:
    print("Red Wine Classification:")
    best_rf_classifier_red = train_evaluate(rf_classifier, rf_classifier_params, X_red_train, y_red_train, X_red_test, y_red_test)
    plot_feature_importance(best_rf_classifier_red, X_red_train, "Red Wine Classification")

    print("White Wine Classification:")
    best_rf_classifier_white = train_evaluate(rf_classifier, rf_classifier_params, X_white_train, y_white_train, X_white_test, y_white_test)
    plot_feature_importance(best_rf_classifier_white, X_white_train, "White Wine Classification")


    print("Combined Wine Classification:")
    best_rf_classifier_combined = train_evaluate(rf_classifier, rf_classifier_params, X_combined_train, y_combined_train, X_combined_test, y_combined_test)
    plot_feature_importance(best_rf_classifier_combined, X_combined_train, "Combined Wine Classification")


    print("Red Wine Regression:")
    best_rf_regressor_red = train_evaluate(rf_regressor, rf_regressor_params, X_red_train, y_red_train, X_red_test, y_red_test, is_classifier=False)
    plot_feature_importance(best_rf_regressor_red, X_red_train, "Red Wine Regression")


    print("White Wine Regression:")
    best_rf_regressor_white = train_evaluate(rf_regressor, rf_regressor_params, X_white_train, y_white_train, X_white_test, y_white_test, is_classifier=False)
    plot_feature_importance(best_rf_regressor_white, X_white_train, "White Wine Regression")


    print("Combined Wine Regression:")
    best_rf_regressor_combined = train_evaluate(rf_regressor, rf_regressor_params, X_combined_train, y_combined_train, X_combined_test, y_combined_test, is_classifier=False)
    plot_feature_importance(best_rf_regressor_combined, X_combined_train, "Combined Wine Regression")