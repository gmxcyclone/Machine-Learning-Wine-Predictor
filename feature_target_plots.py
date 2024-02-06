import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


filename = '../Data/winequality-combined_clean.csv'
data = pd.read_csv(filename)

cols = [col for col in list(data.keys()) if col != 'red_wine' and col != 'quality']

colors = ['b', 'r']

fig,ax = plt.subplots(2,len(cols), sharey=True, layout='constrained')
for x in range(len(cols)):
	for is_red in [0,1]:
		desc = data.loc[data['red_wine']==is_red, cols[x]]
		target = data.loc[data['red_wine']==is_red, 'quality']
		ax[is_red, x].scatter(desc, target, marker='+', color=colors[is_red])
		if is_red == 1:
			ax[is_red, x].set_xlabel(cols[x])
		if x == 0 and is_red == 1:
			ax[is_red, x].set_ylabel('Red Wine Quality')
		if x == 0 and is_red == 0:
			ax[is_red, x].set_ylabel('White Wine Quality')

plt.show()