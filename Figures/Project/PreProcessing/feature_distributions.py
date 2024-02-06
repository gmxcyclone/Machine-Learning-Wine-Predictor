import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


filename = '../Data/winequality-combined_clean.csv'
data = pd.read_csv(filename)

cols = [col for col in list(data.keys()) if col != 'red_wine']


fig,ax = plt.subplots(2,len(cols), sharex='col', sharey=True, layout='constrained')
for x in range(len(cols)):
	for is_red in [0,1]:
		wine_col = data.loc[data['red_wine']==is_red, cols[x]]
		ax[is_red, x].hist(wine_col)
		if is_red == 1:
			ax[is_red, x].set_xlabel(cols[x])
		if x == 0 and is_red == 1:
			ax[is_red, x].set_ylabel('Frequency (Red Wine)')
		if x == 0 and is_red == 0:
			ax[is_red, x].set_ylabel('Frequency (White Wine)')

plt.show()


