import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# train function
def train_evaluate(model, X_train, y_train, X_test, y_test, is_classifier=True):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if is_classifier:
        accuracy = accuracy_score(y_test, predictions)
        print(f'Model Accuracy: {accuracy:.4f}')
    else:
        mse = mean_squared_error(y_test, predictions)
        print(f'Model MSE: {mse:.4f}')

# prepping features and target 
def prepare_data(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def plot_feature_importance(model, X_train, title):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title(f"Feature importances for {title}")
    plt.bar(range(X_train.shape[1]), importances[indices],asda
            color="r", align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.show()




# loading data 
red_wine = pd.read_csv('ML-Project\Data/winequality-red_clean.csv')
white_wine = pd.read_csv('ML-Project\Data/winequality-white_clean.csv')
combined_wine = pd.read_csv('ML-Project\Data\winequality-combined_clean.csv')


# setting types for combined dataset 
combined_wine['red_wine'] = combined_wine['red_wine'].apply(lambda x: 1 if x else 0)


# Splitting data 
X_red_train, X_red_test, y_red_train, y_red_test = prepare_data(red_wine)

X_white_train, X_white_test, y_white_train, y_white_test = prepare_data(white_wine)

X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(
    combined_wine.drop('quality', axis=1),
    combined_wine['quality'],
    test_size=0.2,
    stratify=combined_wine['red_wine'],
    random_state=42
)



# init models
rf_classifier = RandomForestClassifier(random_state=42)
rf_regressor = RandomForestRegressor(random_state=42)



# training and evaluation of classification models
print("Red Wine Classification:")
train_evaluate(rf_classifier, X_red_train, y_red_train, X_red_test, y_red_test)

print("White Wine Classification:")
train_evaluate(rf_classifier, X_white_train, y_white_train, X_white_test, y_white_test)

print("Combined Wine Classification:")
train_evaluate(rf_classifier, X_combined_train, y_combined_train, X_combined_test, y_combined_test)




# training and evaluation of regression models
print("Red Wine Regression:")
train_evaluate(rf_regressor, X_red_train, y_red_train, X_red_test, y_red_test, is_classifier=False)

print("White Wine Regression:")
train_evaluate(rf_regressor, X_white_train, y_white_train, X_white_test, y_white_test, is_classifier=False)

print("Combined Wine Regression:")
train_evaluate(rf_regressor, X_combined_train, y_combined_train, X_combined_test, y_combined_test, is_classifier=False)


# feature importances of 3 datasets 

# red
print("Red Wine Classification Feature Importance:")
rf_classifier.fit(X_red_train, y_red_train)
plot_feature_importance(rf_classifier, X_red_train, "Red Wine")
# white
print("White Wine Classification Feature Importance:")
rf_classifier.fit(X_white_train, y_white_train)
plot_feature_importance(rf_classifier, X_white_train, "White Wine")
# comb
print("Combined Wine Classification Feature Importance:")
rf_classifier.fit(X_combined_train, y_combined_train)
plot_feature_importance(rf_classifier, X_combined_train, "Combined Wine")