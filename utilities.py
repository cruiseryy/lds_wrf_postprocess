import numpy as np
import xarray as xr 
from matplotlib import pyplot as plt
import geopandas as gpd
import cartopy.crs as ccrs
from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim, cartopy_ylim, latlon_coords)
import matplotlib.ticker as mticker
import matplotlib as mpl
import numpy.matlib
from scipy.stats import genpareto as gp
from scipy.optimize import curve_fit

class gpd_fit:
    def __init__(self, X) -> None:
        self.X = X
        pass

    def fit_gpd2(self, qthre = 75):
        threshold = np.percentile(self.X, qthre)
        X = self.X[self.X > threshold]
        X.sort()
        ecdf = np.arange(X.shape[0]) / X.shape[0] 
        para0 = gp.fit(X, loc = threshold)
        pars, cov = curve_fit(lambda x, ksi, sigma: gp.cdf(X, c = ksi, loc=threshold, scale=sigma), X, ecdf, p0 = [para0[0], para0[2]], maxfev = 10000)
        dist = gp(c = pars[0], loc = threshold, scale = pars[1])
        return dist
    
# /home/climate/xp53/nas_home/LDS_WRF_OUTPUT/K=0.05/0_RAINNC.nc
# ref = 8.398613461383452 for the whole WRF simulation domain <- from a incorrect K=0 run
# ref = 9.097700159102828 for the cropped WRF simulation domain <- from the linear correction model

class post_analyzer:
    def __init__(self, 
                 path = '', 
                 var_list = ['RAINNC', 'SMCREL', 'SMOIS'], 
                 N = 128, 
                 T = 18,
                 dt = 5,
                 ref = 8.3986,
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
        self.bc_record = np.loadtxt(self.path + '/VARS/' + 'bc_record.txt').astype(int)

        self.mask = np.loadtxt('mask.txt')

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
        pause = 1
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
        idx = np.array(range(0, 128, 10))
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
        ax.plot([0, self.dt * self.T], [0, self.dt * self.T *9.097700159102828], color = 'black', linestyle = 'dashed', linewidth=1)
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

    def map_plotter(self, data, ax, var_name = 'Rainfall [mm]', crange = [0, 999], cmap = 'jet', align = 1):

        self.coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        if align:
            # a trick to set the colorbar range
            v = np.linspace(crange[0], crange[1], 20, endpoint=True)
            basemap = ax.contourf(self.lonc, self.latc, data, v, 
                                    transform=ccrs.PlateCarree(), cmap=cmap)
        else:
            basemap = ax.contourf(self.lonc, self.latc, data, 10, 
                                    transform=ccrs.PlateCarree(), cmap=cmap)
        basemap.set_clim(crange[0], crange[1])
        # ax.set_extent([np.min(self.lonc), np.max(self.lonc), np.min(self.latc), np.max(self.latc)], crs=ccrs.PlateCarree())
        ax.set_extent([103.599, 104.101, 1.15, 1.501], crs=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = mticker.FixedLocator([103.5, 103.6, 103.7, 103.8, 103.9, 104.0, 104.1])
        gl.ylocator = mticker.FixedLocator([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
        if align:
            cbar = plt.colorbar(basemap, ax=ax, orientation='vertical', shrink=0.8, pad=0.05, ticks = v)
        else:
            cbar = plt.colorbar(basemap, ax=ax, orientation='vertical', shrink=0.8, pad=0.05)
        cbar.set_label(var_name)
        pause = 1
        return 

    def conditional_prob(self, rp_thre = 1000):
        ls = np.where(self.return_period > rp_thre)[0]
        _, _, X, Y = self.rain_orderc.shape
        con_rain = np.zeros((self.M, X, Y))
        con_prob = 0
        for i in ls:
            idx = self.rank[i][0]
            con_rain += self.rain_orderc[idx, :, :, :] * self.pq_ratio[idx] * 1 / self.N
            con_prob += self.pq_ratio[idx] * 1 / self.N
        print([con_prob, 1 / con_prob])
        con_rain /= con_prob
        return con_rain
    
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
        pause = 1
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
    
    def crop_domain(self):
        self.latc = self.lats[5:-5,5:-5]
        self.lonc = self.lons[5:-5,5:-5]
        self.rain_orderc = self.rain_order[:, :, 5:-5, 5:-5]
        return
    
    def return_period(self):

        # read hisotrical era run rainfall data for comparison
        rainh2 = np.loadtxt('/home/climate/xp53/wrf_lds_post/rain.txt')
        rainh = np.loadtxt('/home/climate/xp53/wrf_lds_post/rainc.txt')
        pause = 1

        rainh.sort()
        # rainh[0] = rainh[1] = 800
        # rainh.sort()
        rainh = rainh[2:]
        cdfh = np.arange(1, rainh.shape[0]+1) / (rainh.shape[0] + 1)
        rph = 1 / cdfh

        rainh2.sort()
        # rainh2[0] = rainh2[1] = 800
        # rainh2.sort()
        rainh2 = rainh2[2:]
        cdfh2 = np.arange(1, rainh2.shape[0]+1) / (rainh2.shape[0] + 1)
        rph2 = 1 / cdfh2

        idx2 = np.arange(self.N)
        accu_rain2 = np.mean(self.rain_order[:, :, :, :], axis = (2, 3))
        rank2 = list(zip(idx2, accu_rain2[:, -1]))
        rank2.sort(key = lambda x: x[1])

        # to compute the return period of each ordered traj (having total rainfall smaller than the corresponding traj rainfall)
        idx = np.arange(self.N)
        accu_rain = np.mean(self.rain_orderc[:, :, :, :], axis = (2, 3))
        rank = list(zip(idx, accu_rain[:, -1]))
        rank.sort(key = lambda x: x[1])

        # fit a linear model to translate any things averaged on the whole domain to their equivalence averaged on the cropped domain
        y = accu_rain[:, -1]
        x = accu_rain2[:, -1]
        XX = np.vstack([x, np.ones(len(x))]).T
        slope, intercept = np.linalg.lstsq(XX, y, rcond=None)[0]

        fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(14, 6))
        ax[0].scatter(accu_rain2[:, -1], accu_rain[:, -1], color = 'red', linewidth=1)
        ax[0].plot(accu_rain2[:, -1], slope * accu_rain2[:, -1] + intercept, color = 'blue', linewidth=1)
        ax[0].set_xlabel('Original Accumulative Rainfall [mm]')
        ax[0].set_ylabel('Cropped Accumulative Rainfall [mm]')
        # grid
        ax[0].grid(which='major', axis='x', linestyle='-', linewidth=1, color='grey', alpha=0.5)
        ax[0].grid(which='major', axis='y', linestyle='-', linewidth=1, color='grey', alpha=0.5)

        sg_mask_rain = np.multiply(self.rain_order[:, :, :, :], self.mask)
        accu_rain3 = np.nanmean(sg_mask_rain[:, :, :, :], axis = (2, 3))
        y2 = accu_rain3[:, -1]
        slope2, intercept2 = np.linalg.lstsq(XX, y2, rcond=None)[0]
        ax[1].scatter(accu_rain2[:, -1], accu_rain3[:, -1], color = 'red', linewidth=1)
        ax[1].plot(accu_rain2[:, -1], slope2 * accu_rain2[:, -1] + intercept2, color = 'blue', linewidth=1)
        ax[1].set_xlabel('Original Accumulative Rainfall [mm]')
        ax[1].set_ylabel('SG masked Accumulative Rainfall [mm]')
        # grid
        ax[1].grid(which='major', axis='x', linestyle='-', linewidth=1, color='grey', alpha=0.5)
        ax[1].grid(which='major', axis='y', linestyle='-', linewidth=1, color='grey', alpha=0.5)
        fig.savefig('rain_comp.pdf')
        pause = 1
        
        cdf2 = np.zeros((self.N, ))
        cdf2[0] = 1 / self.N * self.pq_ratio[rank2[0][0]]
        for j in range(1, self.N):
            cdf2[j] = cdf2[j-1] + 1 / self.N * self.pq_ratio[rank2[j][0]]
        return_period2 = 1 / cdf2

        cdf = np.zeros((self.N, ))
        cdf[0] = 1 / self.N * self.pq_ratio[rank[0][0]]
        for j in range(1, self.N):
            cdf[j] = cdf[j-1] + 1 / self.N * self.pq_ratio[rank[j][0]]
        return_period = 1 / cdf

        fig, ax = plt.subplots(figsize=(14, 6), nrows = 1, ncols = 2)
        ax[0].scatter(return_period, [rain[1] for rain in rank], s = 75, color = 'red', linewidth=1)
        ax[0].scatter(rph, rainh, marker = '+', s = 75, color = 'blue', linewidth=3)
        ax[0].set_xlabel('Return Period [year]')
        ax[0].set_ylabel('Total Rainfall [mm]')
        ax[0].set_xscale('log')
        ax[0].set_title('cropped domain')
        # grid
        ax[0].grid(which='major', axis='x', linestyle='-', linewidth=1, color='grey', alpha=0.5)
        ax[0].grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='grey', alpha=0.25)
        ax[0].grid(which='major', axis='y', linestyle='-', linewidth=1, color='grey', alpha=0.5)

        ax[1].scatter(return_period2, [rain[1] for rain in rank2], color = 'red', linewidth=1)
        ax[1].scatter(rph2, rainh2, marker = '+', color = 'blue', linewidth=1)
        ax[1].set_xlabel('Return Period [year]')
        ax[1].set_ylabel('Total Rainfall [mm]')
        ax[1].set_xscale('log')
        ax[1].set_title('original domain')
        # grid
        ax[1].grid(which='major', axis='x', linestyle='-', linewidth=1, color='grey', alpha=0.5)
        ax[1].grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='grey', alpha=0.25)
        ax[1].grid(which='major', axis='y', linestyle='-', linewidth=1, color='grey', alpha=0.5)

        gp_ = gpd_fit(-rainh)
        qthre = 75
        genp = gp_.fit_gpd2(qthre = qthre)
        xxf = np.linspace(np.percentile(-rainh, 75), -500, 1000)
        yyf = 1 - genp.cdf(xxf)
        ax[0].plot(1/(1-qthre/100)/yyf, -xxf, color='black', label='GPD Fit c != 0')
        ax[0].set_xlim([0.7, 35000])

        fig.savefig('test_rp.pdf')

        self.rank = rank
        self.cdf = cdf
        self.return_period = return_period

        pause = 1
        return

        
if __name__ == '__main__':
    # /home/climate/xp53/nas_home/lds_wrf_output_new/k=0.02
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
    tmp.return_period()
    pause = 1
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    tmp.map_plotter(tmp.rain_orderc[tmp.rank[0][0],-1,:,:], ax[0])
    tmp.map_plotter(tmp.rain_orderc[tmp.rank[1][0],-1,:,:], ax[1])
    density = 7
    ax[0][0].contourf(
        tmp.lons, tmp.lats, tmp.mask == 1,
        transform=ccrs.PlateCarree(),
        colors='none',
        levels=[.5,1.5],
        hatches=[density*'/',density*'/'],
    )
    

    fig.savefig('dry_map.pdf')
    pause = 1
