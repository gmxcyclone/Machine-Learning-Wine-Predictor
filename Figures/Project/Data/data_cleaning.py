import numpy as np
import pandas as pd


for wine_type in ['red', 'white']:
	with open(f"winequality-{wine_type}.csv", 'r') as f:
		lines = f.readlines()

	for x in range(len(lines)):
		lines[x] = lines[x].replace('"', '')
		lines[x] = lines[x].replace(';', ',')

	with open(f"winequality-{wine_type}_clean.csv", 'w') as f:
		f.writelines(lines)
	
	