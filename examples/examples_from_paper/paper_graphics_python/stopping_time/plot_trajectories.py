# Copyright (c) Radost Waszkiewicz 2022
# Following code is lincensed under MIT license

import matplotlib.pyplot as plt
import numpy as np

traj_1 = np.genfromtxt('./sources/trajectory_1.csv', delimiter=',')
traj_2 = np.genfromtxt('./sources/trajectory_2.csv', delimiter=',')
traj_3 = np.genfromtxt('./sources/trajectory_3.csv', delimiter=',')
traj_4 = np.genfromtxt('./sources/trajectory_4.csv', delimiter=',')
traj_5 = np.genfromtxt('./sources/trajectory_5.csv', delimiter=',')

boundary_teo = np.genfromtxt('./sources/boundary_teo.csv', delimiter=',')

plt.rcParams.update({
    "legend.frameon": False,
    "text.usetex": True,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 15,
    "figure.subplot.left":   0.125,
    "figure.subplot.right":  0.95,
    "figure.subplot.bottom": 0.15,
    "figure.subplot.top":    0.95,
    "figure.subplot.wspace": 0.2,
    "figure.subplot.hspace": 0.2,
})

plt.plot(traj_3[:,0],traj_3[:,1], 'xkcd:dark orange')
plt.plot(traj_4[:,0],traj_4[:,1], 'xkcd:dark blue')
plt.plot(traj_1[:,0],traj_1[:,1], 'C0')
plt.plot(traj_2[:,0],traj_2[:,1], 'C1')

plt.plot([2],[0], 'ko')

plt.plot(boundary_teo[:,0],boundary_teo[:,1], '--k')

plt.ylim([-0.2,1.2])

plt.xlabel(r"Radial displacement ($r$)")
plt.ylabel(r"Azimuthal displacement ($\phi$)")

plt.tight_layout()

plt.savefig('trajectories.svg', format='svg')
plt.show()
