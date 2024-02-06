import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import pickle as pkl
import time
from itertools import product


validation = False

if validation:
	# Load in Pre-Split Test Data
	data = pd.read_csv('splits/combined_test.csv')
	data.rename(columns={'quality': "quality_true", 'red_wine': "red_wine_true"}, inplace=True)

else:
	# Generate a large hypothetical set of data
	no_data=False
	if no_data:
		print('generating data')
		
		fixed_acidity = np.linspace(3.5, 16, num=3)
		volatile_acidity = np.linspace(0.1, 2, num=2)
		citric_acid	= np.linspace(0.1, 2, num=2)
		residual_sugar = np.linspace(0.5, 70, num=3)
		chlorides = np.linspace(0, 0.7, num=3)
		free_sulfur_dioxide	= np.linspace(5, 450, num=3)
		total_sulfur_dioxide = np.linspace(0, 2, num=3)
		#density = np.linspace(0.9, 1.1, num=3)
		density = [1]
		pH = np.linspace(2.5, 4.5, num=3)
		sulphates = np.linspace(0.1, 2.5, num=3)
		alcohol = np.linspace(7, 16, num=3)

		lists = [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]
		time_start = time.time()
		all_data = []
		i = 0
		for item in product(*lists):
			i+=1
			all_data.append(np.array(item))
			final_array = np.array(all_data)
			df = pd.DataFrame(final_array, columns=['fixed_acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])
			if i%10000 == 0:
				print(f"{i}: {time.time()-time_start}")
		data = df
		data.to_csv('pipeline_predictions/candidates_short.csv', index=False)
		print('done generating data')
	else:
		data = pd.read_csv('pipeline_predictions/candidates_short.csv')



# Load in ML Models
with open('model_objs/wine_type_classifier.pkl', 'rb') as f:
    wine_type = pkl.load(f)
    print(wine_type)

with open('model_objs/red_regressor.pkl', 'rb') as f:
    red_regressor = pkl.load(f)

with open('model_objs/white_regressor.pkl', 'rb') as f:
	white_regressor = pkl.load(f)

# Classify if red wine or white wine

if validation:
	predictions = wine_type.predict(data.drop(columns=['red_wine_true', 'quality_true']))
	print(predictions)

	classified_data = data.copy()
	classified_data['red_wine_pred'] = predictions
	print(classified_data)

	red_data = classified_data.loc[classified_data['red_wine_pred'] == 1]
	white_data = classified_data.loc[classified_data['red_wine_pred'] == 0]

	red_data['quality_pred'] = red_regressor.predict(red_data.drop(columns=['red_wine_true', 'quality_true', 'red_wine_pred']))

	white_data['quality_pred'] = white_regressor.predict(white_data.drop(columns=['red_wine_true', 'quality_true', 'red_wine_pred']))


else:
	predictions = wine_type.predict(data)

	classified_data = data.copy()
	classified_data['red_wine_pred'] = predictions

	red_data = classified_data.loc[classified_data['red_wine_pred'] == 1]
	white_data = classified_data.loc[classified_data['red_wine_pred'] == 0]

	red_data['quality_pred'] = red_regressor.predict(red_data.drop(columns=['red_wine_pred']))

	white_data['quality_pred'] = white_regressor.predict(white_data.drop(columns=['red_wine_pred']))



# Predict value using red wine and white wine regression models



if validation:
	red_mse = mean_squared_error(red_data['quality_true'], red_data['quality_pred'])
	print(f'\nRed Wine MSE: {red_mse:.4f}')
	white_mse = mean_squared_error(white_data['quality_true'], white_data['quality_pred'])
	print(f'\nWhite Wine MSE: {white_mse:.4f}')
else:
	red_data.to_csv('pipeline_predictions/candidates_red_pred.csv', index=False)
	white_data.to_csv('pipeline_predictions/candidates_white_pred.csv', index=False)






