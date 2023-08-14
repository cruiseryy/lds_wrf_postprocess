import numpy as np
import xarray as xr 
from matplotlib import pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)

# /home/climate/xp53/nas_home/LDS_WRF_OUTPUT/K=0.05/0_RAINNC.nc
# ref = 8.398613461383452 for the whole WRF simulation domain <- from a incorrect K=0 run

class post_analyzer:
    def __init__(self, 
                 path = '', 
                 var_list = ['RAINNC', 'SMCREL', 'SMOIS'], 
                 N = 128, 
                 T = 18,
                 dt = 5,
                 ref = 8.398613461383452,
                 k = 0,
                 cl_path = '/home/climate/xp53/for_plotting/cropped/coastlines.shp'
                 ) -> None:
        
        self.path = path
        self.var_listt = var_list
        self.N = N
        self.T = T
        self.dt = dt
        self.M = self.T * (self.dt + 1)

        self.k = k

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
        base_rain = np.zeros((self.N, self.lats.shape[0], self.lats.shape[1]))
        for j in range(self.N):
            file = self.path + '/RESULTS/' +  str(j) + '_RAINNC.nc'
            with xr.open_dataset(file) as ds:
                self.rain_raw[j, :, :, :] = ds['RAINNC'][:self.M, :, :]
                # self.rain_raw[j, :, :, :] = ds['RAINNC'][:, :, :] # uncomment this 
                base_rain[j, :, :] = ds['RAINNC'][0, :, :]
        for j in range(self.N):
            for i in range(self.T)[::-1]:
                ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
                rt = self.find_root(j, i)
                self.rain_raw[j, ts:te, :, :] -= base_rain[rt, :, :]
                pause = 1
        return
    
    def find_root(self, j, t):
        while t > 0:
            j = self.parent[j, t]
            t -= 1
        return j
    
    def order_(self):

        self.rain_order = np.zeros((self.N, self.M, self.lats.shape[0], self.lats.shape[1]))

        for j in range(self.N):
            tidx = j
            for i in range(self.T)[::-1]:
                ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
                self.rain_order[j, ts:te, :, :] = self.rain_raw[tidx, ts:te, :, :]
                tidx = self.parent[tidx, i]

        # this part is for debugging
        # with open(self.path + '/VARS/' + 'log.txt', 'w') as f:
        #     for j in range(self.N):
        #         tidx = j
        #         res1 = []
        #         res2 = []
        #         for i in range(self.T)[::-1]:
        #             res1.append(self.ic[tidx, i])
        #             res2.append(self.find_root(tidx, i))
        #             ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
        #             self.rain_order[j, ts:te, :, :] = self.rain_raw[tidx, ts:te, :, :]
        #             tidx = self.parent[tidx, i]
        #         print('The IC year of the {}th ordered traj is ({}, {}) & ({}, {})'.format(j, np.mean(res1), np.std(res1), np.mean(res2), np.std(res2)), file=f)
        # pause = 1

        train = np.mean(self.rain_order, axis = (2, 3))
        for j in range(self.N):
            res = []
            for i in range(self.T - 1):
                res.append(train[j, (i+1)*6-1] - train[j, (i+1)*6])
            print('The mean rainfall deficit of the {}th ordered traj is {}'.format(j, np.mean(np.abs(res))))
        pause = 1

        return
    
    def traj_plotter(self, ax):
        # to plot the ordered traj as well as the raw traj (background)
        # fill color between the max and min of the raw traj between ts and te
        idx = [0, 1, 2, 3, 12, 13]
        for i in range(self.T):
            ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
            tmp_traj_raw = np.mean(self.rain_raw[:, ts:te, :, :], axis = (2, 3))
            ax.fill_between(np.arange(ts - i, te - i), 
                                np.min(tmp_traj_raw, axis = 0),
                                np.max(tmp_traj_raw, axis = 0),
                                color='grey', alpha=0.25, linewidth=0)
            for j in idx:
                ind_traj_order = np.mean(self.rain_order[j, ts:te, :, :], axis = (1, 2))
                # ax.plot(np.arange(ts - i, te - i), ind_traj_order, color = 'red', linewidth=1)
                ax.plot(np.arange(ts - i, te - i), ind_traj_order, color = 'blue' if j == 1 else 'red', linewidth=1)
        ax.set_xticks(range(0, self.dt * self.T + 1, 20))
        ax.set_xticks(range(0, self.dt * self.T + 1, 5), minor=True)
        pause = 1
        return
    

    def collect_roots(self):
        self.roots = set([])
        for j in range(self.N):
            self.roots.add(self.find_root(j, self.T-1))
        self.roots = list(self.roots)
        return

    def map_plotter(self, data, ax, var_name = 'Rainfall [mm]'):
        self.coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        basemap = ax.contourf(self.lons, self.lats, data, 10, 
                                transform=ccrs.PlateCarree(), cmap="jet")
        ax.set_extent([np.min(self.lons), np.max(self.lons), np.min(self.lats), np.max(self.lats)], crs=ccrs.PlateCarree())
        cbar = plt.colorbar(basemap, ax=ax, orientation='vertical', shrink=.8)
        cbar.set_label(var_name)
        return
    
    def weight_est(self):
        # Iterate over all dt, for each dt, compute the weight of each traj 
        self.weight_ = np.zeros((self.N, self.T))
        self.R_ = np.zeros((self.T,))
        for i in range(self.T-1):
            for j in range(self.N):
                ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
                rain_start = self.rain_raw[j, ts, :, :]
                rain_end = self.rain_raw[j, te-1, :, :]
                rain_diff = np.mean(rain_end - rain_start)
                self.weight_[j, i] = np.exp(self.k * (self.ref * self.dt - rain_diff))
            self.R_[i] = np.mean(self.weight_[:, i])
            self.weight_[:, i] /= self.R_[i]
        # pause = 1 # check if re-estimated weights and Rs are correct
        return

    def agg_weight(self):
        # to compute the aggregated weight of each ordered traj 
        self.pq_ratio = np.ones((self.N, ))
        for j in range(self.N):
            for i in range(self.T - 1):
                ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
                rain_start = self.rain_order[j, ts, :,  :]
                rain_end = self.rain_order[j, te-1, :, :]
                rain_diff = np.mean(rain_end - rain_start)
                self.pq_ratio[j] *= (np.exp(-self.k * (self.ref * self.dt - rain_diff)) * self.R_[i])
                pause = 1
        
        # this is for checking consistency of pq_ratio estimations (compute by each sub interval or compute by the whole traj)
        self.pq_ratio2 = np.zeros((self.N, ))
        agg_R = 1
        for i in range(self.T - 1):
            agg_R *= self.R_[i]
        for j in range(self.N):
            ts, te = 0, (self.T - 1) * (self.dt + 1)
            rain_start = self.rain_order[j, ts, :,  :]
            rain_end = self.rain_order[j, te-1, :, :]
            rain_diff = np.mean(rain_end - rain_start)
            self.pq_ratio2[j] = np.exp(-self.k * (self.ref * self.dt * (self.T - 1) - rain_diff)) * agg_R
            pause = 1
        
        pause = 1 # check if pq_ratio is correct
        return
    
    def return_period(self):
        # to compute the return period of each ordered traj (having total rainfall smaller than the corresponding traj rainfall)
        return

        
if __name__ == '__main__':
    # /home/climate/xp53/nas_home/lds_wrf_output_new/k=0.02
    tmp = post_analyzer(path = "/home/climate/xp53/nas_home/lds_wrf_output_new/k=0.02", T = 5)
    tmp.var_read()
    tmp.order_()
    tmp.weight_est()
    tmp.agg_weight()
    tmp.collect_roots()
    fig, ax = plt.subplots(figsize=(8, 6))
    tmp.traj_plotter(ax)
    ax.set_xlabel('Time Elapsed [day]')
    ax.set_ylabel('Cumulative Rainfall [mm]')
    ax.grid(which='major', axis='x', linestyle='-', linewidth=1, color='grey', alpha=0.5)
    ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='grey', alpha=0.25)
    fig.savefig('test.pdf')
    pause = 1
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    jj = 99
    ii = 1
    tmp.map_plotter(tmp.rain_raw[jj, ii, :, :], ax)
    fig.savefig('test.pdf')
    pause = 1
