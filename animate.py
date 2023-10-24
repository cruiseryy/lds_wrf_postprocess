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

tmp = post_analyzer(path = "/home/water/xp53/nas_home/lds_wrf_output_new/k=0.02", k=0.02, T = 18)
tmp.var_read()
tmp.var_read2()
tmp.order_()
tmp.order2_()

tmp.weight_est()
tmp.agg_weight()

tmp.correct()

tmp.return_period()

tr, tsm, tt, p = tmp.select_data(rp_thre=100)

rain_map = np.zeros((91, 120, 160))

for j in range(len(tr)):
    rain_map += tr[j, :, :, :] * p[j]

mu0 = mu1 = mu2 = 0
tr_traj = np.nanmean(np.multiply(tr, tmp.mask), axis = (2, 3))
pause = 1
for j in range(len(tr)):
    mu0 += p[j]
    mu1 += tr_traj[j, : ] * p[j]
    mu2 += tr_traj[j, : ] ** 2 * p[j]
mu1 /= mu0
mu2 /= mu0
sigma = np.sqrt(mu2 - mu1 ** 2)

rain_map /= mu0

bg_upp, bg_low = tmp.select_bg()

mmax = 399.1
mmin = 117.9

for j in range(92):

    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(3, 5, (1, 10), projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(3, 5, (11, 14))

    tmpmax = np.copy(bg_upp[:j])
    tmpmin = np.copy(bg_low[:j])
    base_xx = np.arange(0, j)
    base_yy = base_xx * 7.14
    tmpmax -= base_yy
    tmpmin -= base_yy
    tmpp1 = np.copy(tmpmax)
    tmpp1[tmpp1 < 0] = 0
    tmpp2 = np.copy(tmpmin)
    tmpp2[tmpp2 <0] = 0 
    tmpn1 = np.copy(tmpmax)
    tmpn1[tmpn1 > 0] = 0
    tmpn2 = np.copy(tmpmin)
    tmpn2[tmpn2 > 0] = 0
    ax2.fill_between(base_xx, tmpp2, tmpp1,
                    color='lightskyblue', alpha=0.25, linewidth=0)
    ax2.fill_between(base_xx, tmpn2, tmpn1,
                    color='orange', alpha=0.25, linewidth=0)
    xmin, xmax = -5, 95
    ax2.set_xlim((xmin, xmax))
    ax2.plot([xmin, xmax], [0, 0], color = 'black', dashes = (3, 1), linewidth = 1.5, label = 'Climatology')

    ax2.plot(base_xx, mu1[:j] - base_yy , color = 'darkred', linewidth = 2, label = 'RP > 100')
    tmp_sig = sigma[:j]
    tmp_low = mu1[:j] - 2*tmp_sig
    tmp_low[tmp_low < 0] = 0
    tmp_upp = mu1[:j] + 2*tmp_sig
    ax2.fill_between(base_xx, tmp_low - base_yy, tmp_upp - base_yy, color = 'darkred', alpha=0.15, linewidth=0)

    ax2.set_xticks(range(0, 90 + 1, 20))
    ax2.set_xticklabels(['12/01', '12/21', '01/10', '01/30', '02/19' ])
    ax2.set_xticks(range(0, 90 + 1, 5), minor=True)
    ax2.set_xlabel('Date (00:00:00)')
    ax2.set_ylabel('Rainfall Anomaly [mm]')
    ax2.grid(which='major', axis='x', linestyle='-', linewidth=1, color='grey', alpha=0.5)
    ax2.grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='grey', alpha=0.25)
    ax2.set_ylim((-480, 80))
    ax2.set_yticks([-400, -300, -200, -100, 0])
    ax2.legend()

    ax1.set_title('Days Elapsed: {}'.format(j-1))
    
    tmp.map_plotter(rain_map[max(0, j-1), :, :], ax1, crange = [mmin, mmax])
    fig.savefig('animated_plot/{}.jpg'.format(j), dpi = 300)
    pause = 1
pause = 1
