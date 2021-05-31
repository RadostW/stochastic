import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')
df['error_sq'] = df.end_error**2
df['id'] = df['name']+df['dt'].astype(str)+df['e_terms'].astype(str)+df['rho'].astype(str)
print(df['id'])
for id_, df_ in df.groupby('id'):
    df_.plot.scatter(x='main_loop_time_ms', y='error_sq')
    plt.title(id_)
    plt.savefig(f'results/results_{id_}.png')
    plt.close()
