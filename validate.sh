g++ validate.cpp -o validate
./validate

python3 - << EOF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.log(1e-3+pd.read_table('toplot.dat', sep=' ').set_index('steps')).plot()
#plt.xlabel('steps')
plt.savefig('foo.png')
EOF