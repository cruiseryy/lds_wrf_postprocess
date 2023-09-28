import numpy as np
import xarray as xr 
from matplotlib import pyplot as plt
import geopandas as gpd
import pandas as pd
import cartopy.crs as ccrs
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import matplotlib.ticker as mticker
import matplotlib as mpl
import numpy.matlib
from scipy.stats import genpareto as gp
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utilities import *

hpath = '~/nas_home/scratch/historical_temp/'
htemp = np.zeros((40, 90, 120, 160))
for j in range(40):
    file = hpath + str(j) + '_T2.nc'
    with xr.open_dataset(file) as ds:
        htemp[j, :, :, :] = ds['T2'].to_numpy()
        pause = 1
pause = 1

tmp = post_analyzer(path = "/home/climate/xp53/nas_home/lds_wrf_output_new/k=0.02", k=0.02, T = 18)
tmp.var_read()
tmp.var_read2()
tmp.order_()
tmp.order2_()

tmp.weight_est()
tmp.agg_weight()

tmp.correct()

tmp.return_period()

htemp_mu = np.mean(htemp, axis = 0) 

htemp_ts = np.nanmean(np.multiply(htemp_mu, tmp.mask), axis = (1, 2))

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
ax1.plot(htemp_ts - 273.15, color = 'black', label = 'Historical')

htemp_map = np.mean(htemp_mu, axis = 0) - 273.15
tmpmax = np.nanmax(np.multiply(htemp_map, tmp.mask))
tmpmin = np.nanmin(np.multiply(htemp_map, tmp.mask))

fig2, ax2 = plt.subplots(2, 2, figsize=(16, 12), subplot_kw=dict(projection=ccrs.PlateCarree()))
tmp.map_plotter(htemp_map, ax2[0, 0], var_name = 'T2', crange = [tmpmin, tmpmax], cmap = 'Oranges')
ax2[0, 0].set_title('Historical')


repidx = list(range(5,107,6))
temp = np.delete(tmp.t2_order, repidx, axis = 1)

ll = [128, 62, 16]
rp = [4, 100, 1000]
cc = ['blue', 'green', 'red']
for i, tl in enumerate(ll):
    ri, ci = (i + 1) // 2, (i + 1) % 2
    tmu = np.zeros((91, 120, 160))
    p0 = 0 
    for k in range(tl):
        idx = tmp.rank[k][0]
        tmp_t = temp[k, :, :, :]
        tmu += tmp_t * tmp.pq_ratio[idx] * 1 / 128
        p0 += tmp.pq_ratio[idx] * 1 / 128
    tmu /= p0
    tmu_ts = np.nanmean(np.multiply(tmu, tmp.mask), axis = (1, 2))
    ax1.plot(tmu_ts - 273.15, color = cc[i], label = 'RP >'  + str(rp[i]))

    tmu_map = np.mean(tmu, axis = 0) - 273.15
    tmpmax = np.nanmax(np.multiply(tmu_map, tmp.mask))
    tmpmin = np.nanmin(np.multiply(tmu_map, tmp.mask))
    tmp.map_plotter(tmu_map, ax2[ri, ci], var_name = 'T2', crange = [tmpmin, tmpmax], cmap = 'Oranges')
    ax2[ri, ci].set_title('RP >' + str(rp[i]))
ax1.legend()
fig2.tight_layout()
fig1.savefig('temp_ts.png', dpi = 300)
fig2.savefig('temp_map.png', dpi = 300)

pause = 1