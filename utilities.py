import numpy as np
import xarray as xr 
from matplotlib import pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)

# /home/climate/xp53/nas_home/LDS_WRF_OUTPUT/K=0.05/0_RAINNC.nc

class post_analyzer:
    def __init__(self, 
                 path = '', 
                 var_list = ['RAINNC', 'SMCREL', 'SMOIS'], 
                 N = 128, 
                 T = 18,
                 dt = 5,
                 ref = 7.078,
                 cl_path = '/home/climate/xp53/for_plotting/cropped/coastlines.shp'
                 ) -> None:
        
        self.path = path
        self.var_listt = var_list
        self.N = N
        self.T = T
        self.dt = dt
        self.M = self.T * (self.dt + 1)

        # read the coastline
        self.coastline = gpd.read_file(cl_path)

        # read the lat and lon
        tmpfile = self.path + '/RESULTS/' + '/0_RAINNC.nc'
        with xr.open_dataset(tmpfile) as ds:
            lats, lons = latlon_coords(ds)
            self.lats, self.lons = lats[0,:,:], lons[0,:,:]

        # read the climatology for computing the rainfall deficit
        self.ref = ref

        # read importance sampiling weights
        self.weights = np.loadtxt(self.path + '/VARS/' + 'weights.txt')
        self.R = np.loadtxt(self.path + '/VARS/' + 'R.txt')

        # read the topological relationships for reconstructing the continous rainfall trajectories
        self.parent = np.loadtxt(self.path + '/VARS/' + 'topo.txt').astype(int)

        # read the initial condition (IC) years
        self.ic = np.loadtxt(self.path + '/VARS/' + 'ic.txt').astype(int)

        pause = 1
        return


    def var_read(self):
        self.rain_raw = np.zeros((self.N, self.M, self.lats.shape[0], self.lats.shape[1]))
        for j in range(self.N):
            file = self.path + '/RESULTS/' +  str(j) + '_RAINNC.nc'
            with xr.open_dataset(file) as ds:
                self.rain_raw[j, :, :, :] = ds['RAINNC'][:, :, :]
        for j in range(self.N):
            for i in range(self.T):
                ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
                rt = self.find_root(j, i)
                self.rain_raw[j, ts:te, :, :] -= self.rain_raw[rt, 0, :, :]
        return
    
    def find_root(self, j, t):
        while t > 0:
            j = self.parent[j, t]
            t -= 1
        return j
    
    def order_(self):
        self.rain_order = np.zeros((self.N, self.M, self.lats.shape[0], self.lats.shape[1]))
        with open(self.path + '/VARS/' + 'log.txt', 'a') as f:
            for j in range(self.N):
                tidx = j
                for i in range(self.T)[::-1]:
                    print('The IC year of the {}th ordered traj is {}'.format(j, self.ic[tidx, i]), file=f)
                    ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
                    self.rain_order[j, ts:te, :, :] = self.rain_order[tidx, ts:te, :, :]
                    tidx = self.parent[tidx, i]
                print('\n', file=f)

        return
    
    def traj_plotter(self, ax):
        # to plot the ordered traj as well as the raw traj (background)
        return


    def map_plotter(self, data, ax, var_name = 'Rainfall [mm]'):
        self.coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        basemap = ax.contourf(self.lons, self.lats, data, 10, 
                                transform=ccrs.PlateCarree(), cmap="jet")
        ax.set_extent([np.min(self.lons), np.max(self.lons), np.min(self.lats), np.max(self.lats)], crs=ccrs.PlateCarree())
        cbar = plt.colorbar(basemap, ax=ax, orientation='vertical', shrink=.8)
        cbar.set_label(var_name)
        return

    def agg_weight(self):
        # to compute the aggregated weight of each ordered traj 
        return
    
    def return_period(self):
        # to compute the return period of each ordered traj (having total rainfall smaller than the corresponding traj rainfall)
        return

        
if __name__ == '__main__':
    tmp = post_analyzer(path = "/home/climate/xp53/nas_home/LDS_WRF_OUTPUT/K=0", T = 10)
    tmp.var_read()
    tmp.order_()
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    jj = 99
    ii = 1
    tmp.map_plotter(tmp.rain_raw[jj, ii, :, :], ax)
    fig.savefig('test.pdf')
    pause = 1
    