from utilities import post_analyzer
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
import numpy as np


tmp = post_analyzer(path = "/home/climate/xp53/nas_home/lds_wrf_output_new/k=0.02", k=0.02, T = 18)
tmp.var_read()
tmp.order_()
tmp.weight_est()
tmp.agg_weight()
tmp.collect_roots()
tmp.crop_domain()

fig, ax = plt.subplots(figsize=(12, 6))
tmp.traj_plotter(ax)
ax.set_xlabel('Time Elapsed [day]')
ax.set_ylabel('Cumulative Rainfall [mm]')
ax.grid(which='major', axis='x', linestyle='-', linewidth=1, color='grey', alpha=0.5)
ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='grey', alpha=0.25)
fig.savefig('test.pdf')
pause = 1

tmp.return_period()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), subplot_kw={'projection': ccrs.PlateCarree()})

vmin = np.min([tmp.rain_orderc[tmp.rank[0][0],-1,:,:], tmp.rain_orderc[tmp.rank[1][0],-1,:,:]])
vmax = np.max([tmp.rain_orderc[tmp.rank[0][0],-1,:,:], tmp.rain_orderc[tmp.rank[1][0],-1,:,:]])

tmp.map_plotter(data = tmp.rain_orderc[tmp.rank[0][0],-1,:,:], ax = ax[0], crange = (vmin, vmax))
tmp.map_plotter(data = tmp.rain_orderc[tmp.rank[1][0],-1,:,:], ax = ax[1], crange = (vmin, vmax))
fig.savefig('dry_map.pdf')


r1 = tmp.conditional_prob(rp_thre=100)[-1,:,:]
r2 = tmp.conditional_prob(rp_thre=1000)[-1,:,:]

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 6), subplot_kw={'projection': ccrs.PlateCarree()})

vmin = np.min([r1, r2])
vmax = np.max([r1, r2])

tmp.map_plotter(data = r1, ax = ax[0], crange = (vmin, vmax))
ax[0].set_title('Return Period > 100 years')
tmp.map_plotter(data = r2, ax = ax[1], crange = (vmin, vmax))
ax[1].set_title('Return Period > 1000 years')
tmax = max(np.abs(np.min(r2 - r1)), np.abs(np.max(r2 - r1)))
tmp.map_plotter(data = r2 - r1, ax = ax[2], crange = (-tmax, tmax), cmap = 'RdBu_r', align = 0)
fig.savefig('cond_rain_map.pdf')

pause = 1