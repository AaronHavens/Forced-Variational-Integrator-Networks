import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text',usetex=True)




data = np.load('cost_5_pend_diff.npy')

## create discrete colormap
##cmap = colors.ListedColormap(['red', 'blue'])
##bounds = [0,10,20]
##norm = colors.BoundaryNorm(bounds, cmap.N)
#
extent = [-2,2,-np.pi,np.pi]
color = 'coolwarm'
ax1 = plt.subplot(231)
#ax11 = plt.subplot(221)

asp = 4/(2*np.pi)#'auto'
im1 = ax1.imshow(data, cmap=color,aspect=asp, vmin=-100, vmax=100, extent=extent)#, norm=norm)

ax1.set_title(r'$5$ trajectories')
#ax11.set_title(r'total cost', rotation='vertical',x=-0.1,y=0.5)
data = np.load('cost_grid.npy')
#ax1.set_xlabel(r'$\dot \theta$')
ax1.set_ylabel(r'$\theta$')
#ax1.set_ylabel(r'$x$')
#plt.colorbar(im1)



data = np.load('cost_10_pend_diff.npy')
ax2 = plt.subplot(232)
im2 = ax2.imshow(data, cmap=color, aspect=asp, vmin=-50, vmax= 50, extent=extent)#, norm=norm)
ax2.set_title(r'$10$ trajectories')
#ax2.set_title(r'total cost', rotation='vertical',x=-0.1,y=0.5)
#ax2.set_xlabel(r'$\dot \theta$')
#ax2.set_ylabel(r'$\theta$')

data = np.load('cost_20_pend_diff.npy')
ax3 = plt.subplot(233)
im3 = ax3.imshow(data, cmap=color, aspect=asp, vmin=-50, vmax= 50, extent=extent)#, norm=norm)
ax3.set_title(r'$20$ trajectories')
#ax2.set_title(r'total cost', rotation='vertical',x=-0.1,y=0.5)
#ax2.set_xlabel(r'$\dot \theta$')
#ax2.set_ylabel(r'$\theta$')

cbar3 = plt.colorbar(im3)
cbar3.set_label(r'total cost difference')

data = np.load('ucost_5_pend_diff.npy')
color = 'coolwarm'
ax4 = plt.subplot(234)
#ax33 = plt.subplot(223)
im4 = ax4.imshow(data, cmap=color, aspect=asp, vmin=-10, vmax=10,extent=extent)#, norm=norm)

ax4.set_xlabel(r'$\dot \theta$')
ax4.set_ylabel(r'$\theta$')

#ax4.set_xlabel(r'$\dot x$')
#ax4.set_ylabel(r'$x$')
#cbar = plt.colorbar(im3)



data = np.load('ucost_10_pend_diff.npy')
ax5 = plt.subplot(235)
im5 = ax5.imshow(data, cmap=color, aspect=asp, vmin=-10, vmax=10, extent=extent)#, norm=norm)
#ax4.set_title(r'total cost', rotation='vertical',x=-0.1,y=0.5)
#ax5.set_xlabel(r'$\dot x$')
ax5.set_xlabel(r'$\dot \theta$')
#ax4.set_ylabel(r'$\theta$')

data = np.load('ucost_20_pend_diff.npy')
ax6 = plt.subplot(236)
im6 = ax6.imshow(data, cmap=color, aspect=asp, vmin=-10, vmax=10, extent=extent)#, norm=norm)
# draw gridlines
#ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
#ax.set_xticks(np.linspace(-4, 4, 6));
#ax.set_yticks(np.arange(-np.pi, np.pi, 6));
#ax4.set_title(r'$20$ trajectories')
#ax4.set_title(r'total cost', rotation='vertical',x=-0.1,y=0.5)
#ax6.set_xlabel(r'$\dot x$')
ax6.set_xlabel(r'$\dot \theta$')
#ax4.set_ylabel(r'$\theta$')


cbar6 = plt.colorbar(im6)
cbar6.set_label(r'control cost difference')
plt.show()

