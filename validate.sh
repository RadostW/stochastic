g++ validate.cpp -o validate
./validate

python3 - << EOF
import pandas as pd
import matplotlib.pyplot as plt
pd.read_table('toplot.dat', sep=' ').interpolate().drop('t', axis=1).plot(); plt.savefig('foo.png')
plt.close()
EOF