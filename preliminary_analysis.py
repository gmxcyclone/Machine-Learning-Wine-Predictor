import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px

def main():
    red_wine, white_wine = read_in_data()
    combined_data = combined_df(False, red_wine, white_wine)

    pca_type = 'combined'

    # regular_pca(pca_type, red_wine.copy(), white_wine.copy(), combined_data.copy())
    plotly_regular_pca(pca_type, red_wine.copy(), white_wine.copy(), combined_data.copy())

    # normalized_pca(pca_type, red_wine.copy(), white_wine.copy(), combined_data.copy())
    plotly_normalized_pca(pca_type, red_wine.copy(), white_wine.copy(), combined_data.copy())

## Read in Data
def read_in_data():
    red_wine = pd.read_csv('../Data/winequality-red_clean.csv')
    white_wine = pd.read_csv('../Data/winequality-white_clean.csv')

    return red_wine, white_wine

## Combine Data and Create New File
def combined_df(export_file, red_wine, white_wine):
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

    pca = PCA(n_components=2)
    X_PCA = pca.fit_transform(data_fit)

    variance_explained = pca.explained_variance_ratio_ * 100
    variance_explained_cumsum = np.cumsum(variance_explained)

    X_head_PCA = ["PC0", "PC1"]
    df_PCA = pd.DataFrame(X_PCA, columns=X_head_PCA)

    data['PC0'] = df_PCA["PC0"]
    data['PC1'] = df_PCA["PC1"]

    return data, variance_explained

## Plot PCA
def plotly_regular_pca(pca_type, red, white, combined):
    colors = {'Red Wine': 'red', 'White Wine': 'blue'}  # Define colors for red and white wine

    if pca_type == 'separate':
        data, variance = fit_pca(combined)
        fig = px.scatter(data,
                         x='PC0', y='PC1', color='red_wine',
                         color_discrete_map={1: colors['Red Wine'], 0: colors['White Wine']},
                         facet_col='red_wine',
                         labels={'PC0': f'PC0 (Variance: {round(variance[0], 1)}%)',
                                 'PC1': f'PC1 (Variance: {round(variance[1], 1)}%)'},
                         title='PCA Analysis')
        fig.update_xaxes(range=[-0.6, 0.7])
        fig.update_yaxes(range=[-0.6, 0.7])

    else:
        if pca_type == 'combined':
            data, variance = fit_pca(combined)
            fig = px.scatter(data,
                             x='PC0', y='PC1', color='red_wine',
                             color_discrete_map={1: colors['Red Wine'], 0: colors['White Wine']},
                             labels={'PC0': f'PC0 (Variance: {round(variance[0], 1)}%)',
                                     'PC1': f'PC1 (Variance: {round(variance[1], 1)}%)'},
                             title='PCA Analysis')
            fig.update_xaxes(range=[-0.6, 0.7])
            fig.update_yaxes(range=[-0.6, 0.7])

    fig.show()

## Plot Normalized PCA
def plotly_normalized_pca(pca_type, red, white, combined):
    desc_cols = [col for col in list(red.keys()) if col != 'quality']

    X_red = red[desc_cols]
    X_white = white[desc_cols]

    desc_cols.append('red_wine')
    X_combined = combined[desc_cols]

    data = [X_red, X_white, X_combined]

    for x in range(3):
        X = data[x]
        X = X.fillna(value=0)
        X = (X - X.min()) / (X.max() - X.min())
        X = X.fillna(value=0)
        data[x] = X

    plotly_regular_pca(pca_type, data[0], data[1], data[2])

if __name__ == '__main__':
    main()