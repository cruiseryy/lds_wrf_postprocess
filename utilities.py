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

        pause = 1
        return


    def var_read(self):
        self.rain_raw = np.zeros((self.N, self.T, self.lats.shape[0], self.lats.shape[1]))
        self.rain_order = np.zeros((self.N, self.T, self.lats.shape[0], self.lats.shape[1]))
        for j in range(self.N):
            file = self.path + '/RESULTS/' +  str(j) + '_RAINNC.nc'
            with xr.open_dataset(file) as ds:
                rain[j, :, :, :] = ds['RAINNC'][:, :, :]


                rain = ds['RAINNC']
                data = rain[5, :, :] - rain[6, :, :]
                if np.max(np.abs(data)) > 10: continue
                lats, lons = latlon_coords(rain)
                lats, lons = lats[0,:,:], lons[0,:,:]
                pause = 1
                fig, ax = plt.subplots(figsize=(8, 6),subplot_kw={'projection': ccrs.PlateCarree()})
                self.coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
                
                basemap = ax.contourf(to_np(lons), to_np(lats), to_np(data), 10, 
                                        transform=ccrs.PlateCarree(), cmap="jet")
                ax.set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)], crs=ccrs.PlateCarree())
                cbar = plt.colorbar(basemap, ax=ax, orientation='vertical', shrink=.9)
                cbar.set_label('Rainfall [mm]')
                fig.savefig(str(j))
                pause = 1
            

        return
    
    def map_plotter(self, data, ax, var_name = 'Rainfall [mm]'):
        self.coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        basemap = ax.contourf(self.lons, self.lats, data, 10, 
                                transform=ccrs.PlateCarree(), cmap="jet")
        ax.set_extent([np.min(self.lons), np.max(self.lons), np.min(self.lats), np.max(self.lats)], crs=ccrs.PlateCarree())

        # add a colorbar and adjust is size to match the plot
        cbar = plt.colorbar(basemap, ax=ax, orientation='vertical', shrink=.9)
        cbar.set_label(var_name)
        
        return
    
    def topo_read(self, flag = 1):
        if flag == 1:
            # 
            return
        


        
if __name__ == '__main__':
    tmp = post_analyzer(path = "/home/climate/xp53/nas_home/LDS_WRF_OUTPUT/K=0")

    subplot_kw={'projection': ccrs.PlateCarree ()}
    tmp.var_read()