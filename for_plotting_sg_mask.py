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
from time import time
plt.rcParams['font.family'] = 'Myriad Pro'
pause = 1


class sg_mask_plotter():
    def __init__(self) -> None:
        self.path = "/home/water/xp53/nas_home/lds_wrf_output_new/k=0.02"
        tmpfile = self.path + '/RESULTS/' + '/0_RAINNC.nc'
        with xr.open_dataset(tmpfile) as ds:
            lats, lons = latlon_coords(ds)
            self.lats, self.lons = lats[0,:,:], lons[0,:,:]
        self.mask = np.loadtxt('mask.txt')
        self.coastline = gpd.read_file('/home/water/xp53/nas_home/coastlines-split-SGregion/lines.shp')
        self.reservoir = pd.read_csv('sta_loc.csv', header = None, skiprows=[0], usecols=[1, 2]).to_numpy()
        pause 
        return
        
    def map_plotter(self, ax):

        self.coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        
        ax.set_extent([103.4, 104.180, 1.105, 1.656], crs=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        # gl.top_labels = False
        # gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([103.6, 103.8, 104.0])
        gl.ylocator = mticker.FixedLocator([1.2, 1.4, 1.6])
        density = 3
        mask_domain = ax.contourf(
            self.lons, self.lats, self.mask == 1,
            transform=ccrs.PlateCarree(),
            colors='none',
            levels=[.5,1.5],
            hatches=[density*'/',density*'/'],
            label = 'SG mask'
        )
        sta_loc = ax.scatter(self.reservoir[:,0], self.reservoir[:,1], s = 25, facecolors='navy', marker='D', label = 'stations')
        ax.legend(handles=[sta_loc], loc='upper left')
        pause = 1
        return 

if __name__ == '__main__':
    tmp = sg_mask_plotter()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    tmp.map_plotter(ax)
    fig.savefig('sg_mask.pdf')