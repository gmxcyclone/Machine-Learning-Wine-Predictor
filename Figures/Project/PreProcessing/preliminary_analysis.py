import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px


def main():

	# Works best with 'Combined'
	# There is a current bug with 'Separate' where it appears that white wine plots both red and white

	red_wine, white_wine = read_in_data()
	combined_data = combined_df(False, red_wine, white_wine)

	pca_type = 'combined'

	#regular_pca(pca_type, red_wine.copy(), white_wine.copy(), combined_data.copy())
	normalized_pca(pca_type, red_wine.copy(), white_wine.copy(), combined_data.copy())


## Read in Data
def read_in_data():
	red_wine = pd.read_csv('../CS4641_Project/Data/winequality-red_clean.csv')
	white_wine = pd.read_csv('../CS4641_Project/Data/winequality-white_clean.csv')

	return red_wine, white_wine


## Combine Data and Create New File
def combined_df(export_file, red_wine, white_wine):
	# Creates the combined dataframe 
	# Exports this dataframe to a CSV file if 'export_file' is True

	red_wine_comb = red_wine.copy()
	red_wine_comb['red_wine'] = 1

	white_wine_comb = white_wine.copy()
	white_wine_comb['red_wine'] = 0


	combined_data = pd.concat([red_wine_comb, white_wine_comb], axis=0)

	if export_file:
		combined_data.to_csv('../Data/winequality-combined_clean.csv', index=False)

	return combined_data

def fit_pca(data):
	if 'red_wine' in data.keys():
		data_fit = data.drop(columns='red_wine')
	else:
		data_fit = data

	pca = PCA(n_components = 2)
	X_PCA = pca.fit_transform(data_fit)

	variance_explained = pca.explained_variance_ratio_ * 100
	variance_explained_cumsum = np.cumsum(variance_explained)

	X_head_PCA = ["PC0", "PC1"]
	df_PCA = pd.DataFrame(X_PCA, columns=X_head_PCA)

	data['PC0'] = df_PCA["PC0"]
	data['PC1'] = df_PCA["PC1"]

	return data, variance_explained

## Plot PCA
def regular_pca(pca_type, red, white, combined):
	colors = {'Red Wine': 'red', 'White Wine': 'blue'}
	if pca_type == 'seperate':
		red_data, variance_red = fit_pca(red)
		white_data, variance_white = fit_pca(white)
		variance = (variance_red, variance_white)
	else:
		combined_data, variance = fit_pca(combined)
	
	if pca_type == 'seperate':
		fig = px.scatter(combined, x='PC0', y='PC1', color='white_wine',
                         color_discrete_map={1: colors['Red Wine'], 0: colors['White Wine']},
                         facet_col='red_wine', facet_col_wrap=2,
                         labels={'PC0': f'PC0 (Variance: {round(variance[0],1)}%)',
                                 'PC1': f'PC1 (Variance: {round(variance[1],1)}%)'},
                         title='Principal Component Analysis')
	else: fig = px.scatter(combined, x='PC0', y='PC1', color='red_wine',
                         color_discrete_map={1: colors['Red Wine'], 0: colors['White Wine']},
                         labels={'PC0': f'PC0 (Variance: {round(variance[0],1)}%)',
                                 'PC1': f'PC1 (Variance: {round(variance[1],1)}%)'},
                         title='Principal Component Analysis')
	
	fig.show()
		
	fig.write_html("pca_plot.html")
	
	


## Plot Normalized PCA
def normalized_pca(pca_type, red, white, combined):
	# Creates a PCA plot after doing min-max scaling on the descriptors
	# Type: {combined, separate, red, white}
		# Combined: Plot both on a single plot
		# Separate: Plot both on separate subplots
		# Red: Plot only data for red wine
		# White: Plot only data for white wine

		desc_cols = [col for col in list(red.keys()) if col != 'quality']


		X_red = red[desc_cols]
		X_white = white[desc_cols] 

		desc_cols.append('red_wine')
		X_combined = combined[desc_cols]

		data = [X_red, X_white, X_combined]

		for x in range(3):
			X = data[x]
			X = X.fillna(value = 0)
			X = (X-X.min())/(X.max()-X.min())
			X = X.fillna(value = 0)
			data[x] = X


		regular_pca(pca_type, data[0], data[1], data[2])
		fig = px.scatter(data[2], x='PC0', y='PC1', color='red_wine',
                     color_discrete_map={0: 'blue', 1: 'red'},
                     labels={'PC0': 'PC0 (Normalized)', 'PC1': 'PC1 (Normalized)'},
                     title='Normalized Principal Component Analysis')
		fig.write_html("normalized_pca_plot.html")

if __name__ == '__main__':
	main()


