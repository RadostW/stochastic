import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.plot([float(x) for x in open('toplot.dat').readlines()])
plt.savefig('plot.png')