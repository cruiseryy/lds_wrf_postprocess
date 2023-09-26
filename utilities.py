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
plt.rcParams['font.family'] = 'Myriad Pro'
pause = 1

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
# ref = 11.52 for the SG masked climatology <- from the ERA5-forced run 
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
    
    def correct(self):
        for j in range(self.N):
            end_rain = np.nanmean(np.multiply(self.rain_order[j, -1, :, :], self.mask))
            rescale = (end_rain - 233.612) / 1.384 / end_rain
            self.rain_order[j, :, :, :] *= rescale
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
    
    def traj_plotter_bg(self, ax):
        # to plot the ordered traj as well as the raw traj (background)
        # fill color between the max and min of the raw traj between ts and te
        for i in range(self.T):
            ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
            tmp_traj_raw = np.nanmean(np.multiply(self.rain_raw[:, ts:te, :, :], self.mask), axis = (2, 3))
            tmpmin = np.quantile(tmp_traj_raw, 0.05, axis = 0)
            tmpmax = np.quantile(tmp_traj_raw, 0.95, axis = 0)
            if i == 0:
                ax.fill_between(np.arange(ts - i, te - i), tmpmin, tmpmax,
                                    color='grey', alpha=0.25, linewidth=0, label='Raw Traj')
            else:
                ax.fill_between(np.arange(ts - i, te - i), tmpmin, tmpmax,
                                    color='grey', alpha=0.25, linewidth=0)

        ax.plot([0, self.dt * self.T], [0, self.dt * self.T * 11.52], color = 'black', linestyle = '--',
                 dashes = (3, 1), linewidth = 1.5, label = 'Climatology')
        ax.set_xticks(range(0, self.dt * self.T + 1, 20))
        ax.set_xticks(range(0, self.dt * self.T + 1, 5), minor=True)
        return
    
    def traj_plotter(self, ax, mu = [], sigma = [], c = 'red', label = ''):
        for i in range(self.T):
            ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
            tmprain = mu[ts:te]
            if i == 0:
                ax.plot(np.arange(ts - i, te - i), tmprain, color = c, linewidth = 2, label = label)
            else:
                ax.plot(np.arange(ts - i, te - i), tmprain, color = c, linewidth = 2)
            
            if sigma.any():
                tmp_sig = sigma[ts:te]
                tmpmin = tmprain - 2 * tmp_sig
                tmpmin[tmpmin < 0] = 0
                tmpmax = tmprain + 2 * tmp_sig
                ax.fill_between(np.arange(ts - i, te - i), 
                                tmpmin, tmpmax,
                                color= c, alpha=0.15, linewidth=0)
        return
    
    def dtraj_plotter_bg(self, ax):
        # to plot the ordered traj as well as the raw traj (background)
        # fill color between the max and min of the raw traj between ts and te
        for i in range(self.T):
            ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
            tmp_traj_raw = np.nanmean(np.multiply(self.rain_order[:, ts:te, :, :], self.mask), axis = (2, 3))
            tmpmin = np.min(tmp_traj_raw, axis = 0)
            tmpmax = np.max(tmp_traj_raw, axis = 0)
            # tmpmin, tmpmax = np.min(tmp_traj_raw, axis = 0),  np.max(tmp_traj_raw, axis = 0)
            base_xx = np.arange(ts - i, te - i) 
            base_yy = 6.44 * base_xx
            if i == 0:
                ax.fill_between(base_xx, tmpmin - base_yy, tmpmax - base_yy,
                                color='grey', alpha=0.25, linewidth=0, label='Raw Traj')
            else:
                ax.fill_between(base_xx, tmpmin - base_yy, tmpmax - base_yy,
                                color='grey', alpha=0.25, linewidth=0)
            pause = 1
        xmin, xmax = -5, 95
        ax.set_xlim((xmin, xmax))
        ax.plot([xmin, xmax], [0, 0], color = 'black', linestyle = '--', dashes = (3, 1), linewidth = 1.5, label = 'Climatology')
        ax.set_xticks(range(0, self.dt * self.T + 1, 20))
        ax.set_xticks(range(0, self.dt * self.T + 1, 5), minor=True)
        return
    
    def dtraj_plotter(self, ax, mu = [], sigma = [], c = 'red', label = ''):
        for i in range(self.T):
            ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
            tmprain = mu[ts:te]
            base_xx = np.arange(ts - i, te - i)
            base_yy = 6.44 * base_xx
            if i == 0:
                ax.plot(base_xx, tmprain - base_yy, color = c, linewidth = 2, label = label)
            else:
                ax.plot(base_xx, tmprain - base_yy, color = c, linewidth = 2)
            
            if sigma.any():
                tmp_sig = sigma[ts:te]
                tmpmin = tmprain - 2 * tmp_sig
                tmpmin[tmpmin < 0] = 0
                tmpmax = tmprain + 2 * tmp_sig
                ax.fill_between(np.arange(ts - i, te - i), 
                                tmpmin - base_yy, tmpmax - base_yy,
                                color= c, alpha=0.15, linewidth=0)
        return

    def collect_roots(self):
        self.roots = set([])
        for j in range(self.N):
            self.roots.add(self.find_root(j, self.T-1))
        self.roots = list(self.roots)
        return

    def map_plotter(self, data, ax, var_name = 'Rainfall [mm]', crange = [0, 999], cmap = 'Blues', align = 1):

        self.coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        if align:
            # a trick to set the colorbar range
            v = np.linspace(crange[0], crange[1], 10, endpoint=True)
            basemap = ax.contourf(self.lons, self.lats, data, v, 
                                    transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        else:
            basemap = ax.contourf(self.lons, self.lats, data, 10, 
                                    transform=ccrs.PlateCarree(), cmap=cmap, extend='both')
        basemap.set_clim(crange[0], crange[1])
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
    
    def return_period(self):

        # read hisotrical era run rainfall data for comparison
        pause = 1
        rainh0 = np.loadtxt('/home/climate/xp53/wrf_lds_post/rain.txt')
        rainh0 = (rainh0 - 233.612) / 1.384
        rainh0.sort()
        outlier0 = rainh0[0]
        rainh0 = rainh0[1:]
        cdfh0 = np.arange(1, rainh0.shape[0]+1) / (rainh0.shape[0] + 1)
        rph0 = 1 / cdfh0

        rainhm = np.loadtxt('/home/climate/xp53/wrf_lds_post/rainm.txt')
        rainhm = (rainhm - 233.612) / 1.384
        rainhm.sort()
        outlierm = rainhm[0]
        rainhm = rainhm[1:]
        cdfhm = np.arange(1, rainhm.shape[0]+1) / (rainhm.shape[0] + 1)
        rphm = 1 / cdfhm

        accu_rain0 = np.mean(self.rain_order[:, :, :, :], axis = (2, 3))
        accu_rainm = np.nanmean(np.multiply(self.rain_order[:, :, :, :], self.mask), axis = (2, 3))
        idx = np.arange(self.N)
        rank0, rankm = list(zip(idx, accu_rain0[:, -1])), list(zip(idx, accu_rainm[:, -1]))
        rank0.sort(key = lambda x: x[1])
        rankm.sort(key = lambda x: x[1])

        cdf0, cdfm = np.zeros((self.N, )), np.zeros((self.N, ))
        cdf0[0], cdfm[0] = 1 / self.N * self.pq_ratio[rank0[0][0]], 1 / self.N * self.pq_ratio[rankm[0][0]]
        for j in range(1, self.N):
            cdf0[j] = cdf0[j-1] + 1 / self.N * self.pq_ratio[rank0[j][0]]
            cdfm[j] = cdfm[j-1] + 1 / self.N * self.pq_ratio[rankm[j][0]]
        rp0, rpm = 1 / cdf0, 1 / cdfm

        # plotting
        fig, ax = plt.subplots(figsize=(14, 6), nrows = 1, ncols = 2)
        ax[0].scatter(rp0, [rain[1] for rain in rank0], s = 75, color = 'red', linewidth=1, label = 'LD Runs')
        ax[0].scatter(rph0, rainh0, marker = '+', s = 75, color = 'blue', linewidth=3, label = 'Historical Runs')
        ax[0].set_xlabel('Return Period [year]')
        ax[0].set_ylabel('Total Rainfall [mm]')
        ax[0].set_xscale('log')
        ax[0].set_title('(a) original domain')
        ax[1].scatter(rpm, [rain[1] for rain in rankm], s = 75, color = 'red', linewidth=1, label = 'LD Runs')
        ax[1].scatter(rphm, rainhm, marker = '+', s = 75, color = 'blue', linewidth=3, label = 'Historical Runs')
        ax[1].set_xlabel('Return Period [year]')
        ax[1].set_ylabel('Total Rainfall [mm]')
        ax[1].set_xscale('log')
        ax[1].set_title('(b) SG masked domain')
        # grid
        ax[0].grid(which='major', axis='x', linestyle='-', linewidth=1, color='grey', alpha=0.5)
        ax[0].grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='grey', alpha=0.25)
        ax[0].grid(which='major', axis='y', linestyle='-', linewidth=1, color='grey', alpha=0.5)
        ax[1].grid(which='major', axis='x', linestyle='-', linewidth=1, color='grey', alpha=0.5)
        ax[1].grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='grey', alpha=0.25)
        ax[1].grid(which='major', axis='y', linestyle='-', linewidth=1, color='grey', alpha=0.5)

        gp_ = gpd_fit(-rainh0)
        qthre = 75
        genp = gp_.fit_gpd2(qthre = qthre)
        xxf = np.linspace(np.percentile(-rainh0, 75), -100, 1000)
        yyf = 1 - genp.cdf(xxf)
        ax[0].plot(1/(1-qthre/100)/yyf, -xxf, color='black', label='GPD Fit')

        gp_ = gpd_fit(-rainhm)
        qthre = 75
        genp = gp_.fit_gpd2(qthre = qthre)
        xxf = np.linspace(np.percentile(-rainhm, 75), -100, 1000)
        yyf = 1 - genp.cdf(xxf)
        ax[1].plot(1/(1-qthre/100)/yyf, -xxf, color='black', label='GPD Fit')

        ax[0].plot([0.6, 20000], [outlier0, outlier0], color = 'grey', linestyle = 'dashed', linewidth=1)
        ax[1].plot([0.6, 20000], [outlierm, outlierm], color = 'grey', linestyle = 'dashed', linewidth=1)
        ax[0].set_xlim([0.6, 20000])
        ax[1].set_xlim([0.6, 20000])
        ax[0].legend()
        ax[1].legend()

        fig.savefig('fig_new/test_rp.pdf')
        pause = 1

        # check if the relationship between masked and unmasked rainfall is linear
        # comment this later
        # y = accu_rainm[:, -1]
        # x = accu_rain0[:, -1]
        # XX = np.vstack([x, np.ones(len(x))]).T
        # slope, intercept = np.linalg.lstsq(XX, y, rcond=None)[0]
        # fig, ax = plt.subplots()
        # ax.scatter(x, y, s = 50, color = 'red', marker = '.', alpha = 0.7, label = 'LD runs')
        # ax.plot(x, slope * x + intercept, color = 'black')

        # y = np.loadtxt('/home/climate/xp53/wrf_lds_post/rainm.txt')
        # x = np.loadtxt('/home/climate/xp53/wrf_lds_post/rain.txt')
        # XX = np.vstack([x, np.ones(len(x))]).T
        # slope, intercept = np.linalg.lstsq(XX, y, rcond=None)[0]
        # ax.scatter(x, y, s = 50, color = 'blue', marker = '+', alpha = 0.7, label = 'historical runs', linewidths=1.5)
        # ax.plot(x, slope * x + intercept, color = 'black')

        # ax.legend()
        # ax.set_xlabel('Original Accumulative Rainfall [mm]')
        # ax.set_ylabel('Masked Accumulative Rainfall [mm]')
        # fig.savefig('rain_comp_o_vs_mask.pdf')

        self.rp = rpm
        self.rank = rankm
        return
    
    def conditional_expectation(self, rp_thre = 1000):
        ls = np.where(self.rp > rp_thre)[0]
        _, _, X, Y = self.rain_order.shape
        cond_rain = np.zeros((self.M, X, Y))
        cond_prob = 0
        mu2 = np.zeros((self.M, X, Y))
        for j in ls:
            idx = self.rank[j][0]
            cond_rain += self.rain_order[idx, :, :, :] * self.pq_ratio[idx] * 1 / self.N
            mu2 += self.rain_order[idx, :, :, :] ** 2 * self.pq_ratio[idx] * 1 / self.N
            cond_prob += self.pq_ratio[idx] * 1 / self.N
        print([cond_prob, 1 / cond_prob])
        cond_rain /= cond_prob
        mu2 /= cond_prob
        cond_var = mu2 - cond_rain ** 2
        cond_std = np.sqrt(cond_var)
        return cond_rain, cond_std
    
    def conditional_traj(self, rp_thre = 1000):
        ls = np.where(self.rp > rp_thre)[0]
        _, _, X, Y = self.rain_order.shape
        cond_rain = np.zeros((self.M,))
        cond_prob = 0
        mu2 = np.zeros((self.M,))
        for j in ls:
            idx = self.rank[j][0]
            tmprain = np.nanmean(np.multiply(self.rain_order[idx, :, :, :], self.mask), axis = (1, 2))
            cond_rain += tmprain * self.pq_ratio[idx] * 1 / self.N
            mu2 += tmprain ** 2 * self.pq_ratio[idx] * 1 / self.N
            cond_prob += self.pq_ratio[idx] * 1 / self.N
        print([cond_prob, 1 / cond_prob])
        cond_rain /= cond_prob
        mu2 /= cond_prob
        cond_var = mu2 - cond_rain ** 2
        cond_std = np.sqrt(cond_var)
        return cond_rain, cond_std

if __name__ == '__main__':
    plt.rcParams['font.family'] = 'Myriad Pro'
    # /home/climate/xp53/nas_home/lds_wrf_output_new/k=0.02
    tmp = post_analyzer(path = "/home/climate/xp53/nas_home/lds_wrf_output_new/k=0.02", k=0.02, T = 18)
    tmp.var_read()
    tmp.order_()
    tmp.weight_est()
    tmp.agg_weight()
    tmp.correct()
    tmp.collect_roots()

    tmp.return_period()
    

    fig, ax = plt.subplots(figsize=(12, 6))
    tmp.dtraj_plotter_bg(ax)
    traj_mu1, traj_sigma1 = tmp.conditional_traj(rp_thre=100)
    traj_mu2, traj_sigma1 = tmp.conditional_traj(rp_thre=1000)
    tmp.dtraj_plotter(ax, traj_mu1, traj_sigma1, c = 'red', label = 'RP > 105')
    tmp.dtraj_plotter(ax, traj_mu2, traj_sigma1, c = 'blue', label = 'RP > 1139')
    ax.set_xlabel('Time Elapsed [day]')
    ax.set_ylabel('Rainfall Anomaly [mm]')
    ax.grid(which='major', axis='x', linestyle='-', linewidth=1, color='grey', alpha=0.5)
    ax.grid(which='minor', axis='x', linestyle='--', linewidth=0.5, color='grey', alpha=0.25)
    ax.legend()
    fig.savefig('fig_new/test.pdf')
    
    pause = 1

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), subplot_kw={'projection': ccrs.PlateCarree()})
    r_mu1, r_sigma1 = tmp.conditional_expectation(rp_thre=100)
    r_mu1, r_sigma1 = r_mu1[-1,:,:], r_sigma1[-1,:,:]
    r_mu2, r_sigma2 = tmp.conditional_expectation(rp_thre=1000)
    r_mu2, r_sigma2 = r_mu2[-1,:,:], r_sigma2[-1,:,:]
    # rain_clim = np.loadtxt('rain_clim.txt')
    # r_mu1 = rain_clim - r_mu1
    # r_mu2 = rain_clim - r_mu2
    tmpmax = np.max([np.nanmax(np.multiply(r_mu1, tmp.mask)), np.nanmax(np.multiply(r_mu2, tmp.mask))])
    tmpmin = np.min([np.nanmin(np.multiply(r_mu1, tmp.mask)), np.nanmin(np.multiply(r_mu2, tmp.mask))])
    tmp.map_plotter(r_mu1, ax[0][0], crange = [tmpmin, tmpmax])
    tmp.map_plotter(r_mu2, ax[0][1], crange = [tmpmin, tmpmax])
    ax[0][0].set_title('(a) E[rainfall | RP > 105]')
    ax[0][1].set_title('(b) E[rainfall | RP > 1139]')

    tmpmax = np.max([np.nanmax(np.multiply(r_sigma1, tmp.mask)), np.nanmax(np.multiply(r_sigma2, tmp.mask))])
    tmpmin = np.min([np.nanmin(np.multiply(r_sigma1, tmp.mask)), np.nanmin(np.multiply(r_sigma2, tmp.mask))])
    tmp.map_plotter(r_sigma1, ax[1][0], crange = [tmpmin, tmpmax])
    tmp.map_plotter(r_sigma2, ax[1][1], crange = [tmpmin, tmpmax])
    ax[1][0].set_title('(c) Sigma[Rainfall | RP > 105]')
    ax[1][1].set_title('(d) Sigma[Rainfall | RP > 1139]')
    # density = 7
    # ax[0][0].contourf(
    #     tmp.lons, tmp.lats, tmp.mask == 1,
    #     transform=ccrs.PlateCarree(),
    #     colors='none',
    #     levels=[.5,1.5],
    #     hatches=[density*'/',density*'/'],
    # )
    fig.tight_layout()
    fig.savefig('fig_new/dry_map.pdf')
    pause = 1


