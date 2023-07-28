import numpy as np
import xarray as xr 
from matplotlib import pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)

# /home/climate/xp53/nas_home/LDS_WRF_OUTPUT/K=0.05/0_RAINNC.nc
#  
class post_analyzer:
    def __init__(self, path = '', 
                 var_list = ['RAINNC', 'SMCREL', 'SMOIS'], 
                 N = 128, 
                 T = 18,
                 dt = 5) -> None:
        self.path = path
        self.var_listt = var_list
        self.N = N
        self.T = T
        self.dt = dt

        self.coastline = gpd.read_file('/home/climate/xp53/for_plotting/cropped/coastlines.shp')

        return
    
    def var_read(self):
        for j in range(self.N):
            file = self.path + str(j) + '_RAINNC.nc'
            with xr.open_dataset(file) as ds:
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
    
    def topo_read(self, flag = 1):
        if flag == 1:
            # read topo
            return
        


        
if __name__ == '__main__':
    tmp = post_analyzer(path = "/home/climate/xp53/nas_home/LDS_WRF_OUTPUT/K=0.05/")
    tmp.var_read()