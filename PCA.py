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

	#normalized_pca(pca_type, red_wine.copy(), white_wine.copy(), combined_data.copy())
	
	for pca_type in ['combined']:
		normalized_pca(pca_type, red_wine.copy(), white_wine.copy(), combined_data.copy())
	

## Read in Data
def read_in_data():
	red_wine = pd.read_csv('Data/winequality-red_clean.csv')
	white_wine = pd.read_csv('Data/winequality-white_clean.csv')

	return red_wine, white_wine


## Combine Data and Create New File
def combined_df(export_file, red_wine, white_wine):
	# Creates the combined dataframe 
	# Exports this dataframe to a CSV file if 'export_file' is True

	red_wine_comb = red_wine.copy()
	red_wine_comb['red_wine'] = 1

	white_wine_comb = white_wine.copy()
	white_wine_comb['red_wine'] = 0


	combined_data = pd.concat([white_wine_comb, red_wine_comb], axis=0)

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
def regular_pca(pca_type, red, white, combined, quality_dict):
	# Creates a PCA plot without doing any scaling on the descriptors
	# Type: {combined, separate, red, white}
		# combined: Plot both on a single plot
		# separate: Plot both on separate subplots
		# red: Plot only data for red wine
		# white: Plot only data for white wine

	colors = ['r', 'b']

	if pca_type == 'separate':
		fig, ax = plt.subplots(1,2)

		# ONE OPTION:
		#red_data = fit_pca(red)
		#white_data = fit_pca(white)
		#ax[0].scatter(red_data['PC0'], red_data['PC1'], marker='+', color=colors[0])
		#ax[1].scatter(white_data['PC0'], white_data['PC1'], marker='+', color=colors[1])
		#############
		# SECOND OPTION
		combined_data, variance = fit_pca(combined)
		ax[0].scatter(
			combined_data.loc[combined_data['red_wine']==1,'PC0'], 
			combined_data.loc[combined_data['red_wine']==1,'PC1'], 
			marker='+', 
			color=colors[0]
		)
		ax[1].scatter(
			combined_data.loc[combined_data['red_wine']==0,'PC0'], 
			combined_data.loc[combined_data['red_wine']==0,'PC1'], 
			marker='+', 
			color=colors[1]
		)
		################
		

		ax[0].set_title('Red Wine')
		ax[1].set_title('White Wine')
		for x in range(2):
			ax[x].set(xlim=(-0.6, 0.7), ylim=(-0.6, 0.7))
			ax[x].set_xlabel(f'PC0 (Variance: {round(variance[0],1)}%)')
			ax[x].set_ylabel(f'PC1 (Variance: {round(variance[1],1)}%)')


	else:
		if pca_type == 'combined':
			data = combined
			quality = quality_dict['combined']
		elif pca_type == 'red':
			data = red
			quality = quality_dict['red']
		elif pca_type == 'white':
			data = white
			quality = quality_dict['white']

		data, variance = fit_pca(data)
		data['quality'] = quality

		if pca_type == 'combined':
			data['red_wine'] = data['red_wine'].astype(int)
			data['white_wine'] = 1 - data['red_wine']
			data.to_csv('temp.csv')
			fig = px.scatter(
				data,
				x="PC0",
				y="PC1",
				color='red_wine',
				labels={'PC0': f'PC0 (Variance: {round(variance[0],1)}%)',
						'PC1': f'PC1 (Variance: {round(variance[1],1)}%)'},
				title='Principal Component Analysis'
				)
			fig.update_traces(marker=dict(size=12))
			fig.write_html("PCA_combined.html")
			'''
			ax.scatter(
				data.loc[data['red_wine']==0,'PC0'], 
				data.loc[data['red_wine']==0,'PC1'], 
				marker='+', 
				color=colors[1],
				label='White Wine',
			)
			ax.scatter(
				data.loc[data['red_wine']==1,'PC0'], 
				data.loc[data['red_wine']==1,'PC1'], 
				marker='+', 
				color=colors[0], 
				label='Red Wine',
			)
			ax.legend()
			ax.set(xlim=(-0.6, 0.7), ylim=(-0.6, 0.7))
			'''

		elif pca_type == 'red':
			fig =px.scatter(
				data,
				x='PC0',
				y='PC1', 
				color='quality',
				labels={'PC0': f'PC0 (Variance: {round(variance[0],1)}%)',
						'PC1': f'PC1 (Variance: {round(variance[1],1)}%)'},
				title='Principal Component Analysis (Red)'
				)
			fig.update_traces(marker=dict(size=12))
			fig.write_html('PCA_red.html')
		elif pca_type == 'white':
			fig =px.scatter(
				data,
				x='PC0',
				y='PC1', 
				color='quality',
				labels={'PC0': f'PC0 (Variance: {round(variance[0],1)}%)',
						'PC1': f'PC1 (Variance: {round(variance[1],1)}%)'},
				title='Principal Component Analysis (White)'
				)
			fig.update_traces(marker=dict(size=12))
			fig.write_html('PCA_white.html')
		#ax.set_title('Principal Component Analysis')
		#ax.set_xlabel(f'PC0 (Variance: {round(variance[0],1)}%)')
		#ax.set_ylabel(f'PC1 (Variance: {round(variance[1],1)}%)')

	#plt.show()


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

		qualities = {
			'red':red['quality'],
			'white':white['quality'],
			'combined':combined['quality'],
		}

		for x in range(3):
			X = data[x]
			X = X.fillna(value = 0)
			X = (X-X.min())/(X.max()-X.min())
			X = X.fillna(value = 0)
			data[x] = X


		regular_pca(pca_type, data[0], data[1], data[2], qualities)

if __name__ == '__main__':
	main()


