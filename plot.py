import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
pd.read_table('toplot.dat', sep=' ', header=None).plot()
plt.savefig('plot.png')