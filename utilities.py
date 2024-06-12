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

# i intentionally hardcoded this script to read only rainfall (cus we need this estimate probabilities) and T2.
# this code only works for LD coupled with WRF and when rainfall is used as the score function A for computing the importance sampling weights
class GenPareto:
    def __init__(self, X, alpha = 0.5, beta = 0.5) -> None:
        self.X = np.array(sorted(X))
        # they are parameters for estimating ecdfs, see Sample quantiles in statistical packages
        self.alpha = alpha
        self.beta = beta
        pass

    # fitting a generalized pareto distribution minimizing the L-2 norm of CDF errors
    # essentially you can regard this as least squares estimates for matching quantiles (conditional on the threshold)
    def fit(self, qthre = 75):
        threshold = np.percentile(self.X, qthre)
        X = self.X[self.X >= threshold]
        # note in the Peak-Over-Threshold analysis, we typically use the survival probability P(X > x|X > threshold)
        # but here using the CDF P(X < x|X > threshold) is more convenient for fitting
        # the ecdf is given by (k - alpha) / (n - alpha - beta + 1) 
        ecdf = (np.arange(1, X.shape[0] + 1) - self.alpha) / (X.shape[0] - self.beta - self.alpha + 1)
        # use the MLE estimates as initial guesses
        para0 = gp.fit(X, loc = threshold) 
        pars, cov = curve_fit(lambda x, ksi, sigma: gp.cdf(X, c=ksi, loc=threshold, scale=sigma), X, ecdf, p0=[para0[0], para0[2]], maxfev=10000)
        dist = gp(c = pars[0], loc = threshold, scale = pars[1])
        # linear approximation around the optimal parameters to estimate the uncertainty, see curve_fit documentation for details
        # and i simply ignore the correlation between the parameters here
        pars1 = pars - np.sqrt(np.diag(cov))
        pars2 = pars + np.sqrt(np.diag(cov))
        # see WWA protocal paper S Philip & GJ van Oldenborgh et al. 2020, ksi is suggested to be bounded by [-0.4, 0.4]
        dist1 = gp(c = np.max([-0.4, pars1[0]]), loc = threshold, scale = pars1[1])
        dist2 = gp(c = np.min([0.4, pars2[0]]), loc = threshold, scale = pars2[1])
        return dist, dist1, dist2
   
class post_analyzer:
    def __init__(self, 
                 path = '', 
                 N = 128, 
                 T = 18,
                 dt = 5,
                 ref = 10,
                 k = 0,
                 cl_path = '/home/mizu_home/xp53/nas_home/coastlines-split-SGregion/lines.shp'
                 ) -> None:
        # path is where you store the results
        self.path = path
        # N is the number of trajectories/ensemble members
        self.N = N
        # T is the number of subintervals
        self.T = T
        # dt is the length of the subinterval
        self.dt = dt
        # M is the total length of the simulation horizon
        # +1 for overlapping days between subintervals
        self.M = self.T * (self.dt + 1)
        # k is the penalty parameter in the importance sampling step
        self.k = k
        # coastline
        self.coastline = gpd.read_file(cl_path)
        # read one RAINNC file for domain lat and lon
        tmpfile = self.path + '/RESULTS/' + '/0_RAINNC.nc'
        with xr.open_dataset(tmpfile) as ds:
            lats, lons = latlon_coords(ds)
            self.lats, self.lons = lats[0,:,:], lons[0,:,:]
        # climatology for computing the rainfall deficit
        self.ref = ref
        # resampling weights and their normalization constants
        self.weights = np.loadtxt(self.path + '/VARS/' + 'weights.txt')
        self.R = np.loadtxt(self.path + '/VARS/' + 'R.txt')
        # trajecotory topology 
        self.parent = np.loadtxt(self.path + '/VARS/' + 'topo.txt').astype(int)
        # the initial condition years
        self.ic = np.loadtxt(self.path + '/VARS/' + 'ic.txt').astype(int)
        # the boundary condition years
        self.bc_record = np.loadtxt(self.path + '/VARS/' + 'bc_record.txt').astype(int)
        # an arbitrary mask (by me) of Singapore
        self.mask = np.loadtxt('mask.txt')
        # all reservoir locations within Singapore
        self.reservoir = pd.read_csv('sg_reservoir_loc.csv', header = None, skiprows=[0], usecols=[1, 2]).to_numpy()
        # i hardcoded the time resolution of wrfout here
        self.t_res = 4 # 4 snapshots per day when foreced by CMIP6
        return
    
    # trace back to its root ic year at subinterval 0 
    def find_root(self, j, t):
        while t > 0:
            j = self.parent[j, t]
            t -= 1
        return j
    
    ########################################################################
    # Step 1: read raw variables
    ########################################################################
    def var_read(self):
        self.rain_raw = np.zeros((self.N, self.M, self.lats.shape[0], self.lats.shape[1]))
        base_rain = np.zeros((self.N, self.lats.shape[0], self.lats.shape[1]))
        for j in range(self.N):
            file = self.path + '/RESULTS/' +  str(j) + '_RAINNC.nc'
            with xr.open_dataset(file) as ds:
                self.rain_raw[j, :, :, :] = ds['RAINNC'][:self.M, :, :]
                base_rain[j, :, :] = ds['RAINNC'][0, :, :]
        # set the accumulated rainfall to be zero at the start for all trajectories
        for j in range(self.N):
            for i in range(self.T)[::-1]:
                ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
                rt = self.find_root(j, i)
                self.rain_raw[j, ts:te, :, :] -= base_rain[rt, :, :]
        
        # # for possible future expansion, i isolated the part for reading T2 here
        # self.t2_raw = np.zeros((self.N, self.M, self.lats.shape[0], self.lats.shape[1]))
        # for j in range(self.N):
        #     file = self.path + '/RESULTS/' +  str(j) + '_T2.nc'
        #     with xr.open_dataset(file) as ds:
        #         self.t2_raw[j, :, :, :] = ds['T2'][:self.M, :, :]
        pause = 1
        return
    
    def var_read_subdaily(self):
        # self.T*self.dt*(self.t_res+1): number of total days * (# of snapshots per day + 1 for overlapping)
        self.rain_raw_sd = np.zeros((self.N, self.T*self.dt*(self.t_res+1), self.lats.shape[0], self.lats.shape[1]))
        base_rain = np.zeros((self.N, self.lats.shape[0], self.lats.shape[1]))
        for j in range(self.N):
            file = self.path + '/RESULTS_SUBDAILY/' +  str(j) + '_RAINNC.nc'
            with xr.open_dataset(file) as ds:
                self.rain_raw_sd[j, :, :, :] = ds['RAINNC']
                base_rain[j, :, :] = ds['RAINNC'][0, :, :]
        # set the accumulated rainfall to be zero at the start for all trajectories
        for j in range(self.N):
            for i in range(self.T)[::-1]:
                ts, te = i * self.dt * (self.t_res+1), (i + 1) * self.dt * (self.t_res+1)
                rt = self.find_root(j, i)
                self.rain_raw_sd[j, ts:te, :, :] -= base_rain[rt, :, :]

        # # for possible future expansion, i isolated the part for reading T2 here
        # self.t2_raw_sd = np.zeros((self.N, self.T*self.dt*(self.t_res+1), self.lats.shape[0], self.lats.shape[1]))
        # for j in range(self.N):
        #     file = self.path + '/RESULTS_SUBDAILY/' +  str(j) + '_T2.nc'
        #     with xr.open_dataset(file) as ds:
        #         self.t2_raw_sd[j, :, :, :] = ds['T2']
        return
    
    ########################################################################
    # Step 2: sort out the trajectories given the recorded topology
    ########################################################################
    def order_traj(self):
        self.rain_order = np.zeros((self.N, self.M, self.lats.shape[0], self.lats.shape[1]))
        for j in range(self.N):
            tidx = j
            for i in range(self.T)[::-1]:
                ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
                self.rain_order[j, ts:te, :, :] = self.rain_raw[tidx, ts:te, :, :]
                tidx = self.parent[tidx, i]
        
        # this part is for debugging
        # first check if the topology is consistent with the recorded IC years
        with open(self.path + '/VARS/' + 'log.txt', 'w') as f:
            for j in range(self.N):
                tidx = j
                res1 = []
                res2 = []
                for i in range(self.T)[::-1]:
                    res1.append(self.ic[tidx, i])
                    res2.append(self.find_root(tidx, i))
                    tidx = self.parent[tidx, i]
                print('The IC year of the {}th ordered traj is ({}, {}) & ({}, {})'.format(j, np.mean(res1), np.std(res1), np.mean(res2), np.std(res2)), file=f)
        # then check if the overlapping rainfalls are consistent
            train = np.mean(self.rain_order, axis = (2, 3))
            for j in range(self.N):
                res = []
                for i in range(self.T - 1):
                    res.append(train[j, (i+1)*(self.dt+1)-1] - train[j, (i+1)*(self.dt+1)])
                print('The mean rainfall deficit of the {}th ordered traj is {}'.format(j, np.mean(np.abs(res))), file=f)
        
        # # for possible future expansion, i isolated the part for ordering temperature and other climate vars here
        # self.t2_order = np.zeros((self.N, self.M, self.lats.shape[0], self.lats.shape[1]))
        # for j in range(self.N):
        #     tidx = j
        #     for i in range(self.T)[::-1]:
        #         ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
        #         self.t2_order[j, ts:te, :, :] = self.t2_raw[tidx, ts:te, :, :]
        #         tidx = self.parent[tidx, i]
        pause = 1
        return
    
    def order_sd(self):
        # note i do not distinguish between rainfall and other climat vars here cus the ordering and reweighting noly rely on daily rainfall
        self.rain_order_sd = np.zeros(self.N, self.T*self.dt*(self.t_res+1), self.lats.shape[0], self.lats.shape[1])
        self.t2_order_sd = np.zeros(self.N, self.T*self.dt*(self.t_res+1), self.lats.shape[0], self.lats.shape[1])
        for j in range(self.N):
            tidx = j
            for i in range(self.T)[::-1]:
                ts, te = i * self.dt * (self.t_res+1), (i + 1) * self.dt * (self.t_res+1)
                self.t2_order_sd[j, ts:te, :, :] = self.t2_raw_sd[tidx, ts:te, :, :]
                self.rain_order_sd[j, ts:te, :, :] = self.rain_raw_sd[tidx, ts:te, :, :]
                tidx = self.parent[tidx, i]
        return

    ########################################################################
    # Step 3: estimate the weights and re-weight the trajectories
    ########################################################################
    # IMPORTANT: when re-weighting the trajectories, make sure you use the consistent score function A
    def weight_est(self):
        # iterate over all dt for all trajectories, estimate the resampling weights
        # check if the post-estimated weights are consistent with the recorded weights
        self.weight_ = np.zeros((self.N, self.T))
        self.R_ = np.zeros((self.T,))
        for i in range(self.T-1):
            for j in range(self.N):
                ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
                rain_start = self.rain_raw[j, ts, :, :]
                rain_end = self.rain_raw[j, te-1, :, :]
                rain_df = rain_end - rain_start
                # make consistent with the weighting functions used in the sampling algorithm
                # i used the rainfall deficit over a masked Singapore domain in the sampling algorithm
                delta_r = np.nanmean(np.multiply(rain_df, self.mask))
                self.weight_[j, i] = np.exp(self.k * (self.ref * self.dt - delta_r))
            self.R_[i] = np.mean(self.weight_[:, i])
            self.weight_[:, i] /= self.R_[i]
        pause = 1
        return
    
    def agg_weight(self):
        # reweighting coefficients are saved in pq_ratio
        self.pq_ratio = np.ones((self.N, ))
        # check if the pq_ratios are consistent computed using the original definition and the pre-computed weights
        self.pq_ratio2 = np.ones((self.N, ))
        for j in range(self.N):
            tidx = j 
            for i in range(self.T)[::-1]:
                # no resampling at the last subinterval
                if i == self.T - 1:
                    self.pq_ratio[j] *= 1
                    self.pq_ratio2[j] *= 1
                else:
                    ts, te = i * (self.dt + 1), (i + 1) * (self.dt + 1)
                    rain_start = self.rain_order[j, ts, :,  :]
                    rain_end = self.rain_order[j, te-1, :, :]
                    rain_df = rain_end - rain_start
                    delta_r = np.nanmean(np.multiply(rain_df, self.mask))
                    self.pq_ratio[j] *= (np.exp(-self.k * (self.ref * self.dt - delta_r)) * self.R_[i])
                    self.pq_ratio2[j] *= (1 / self.weight_[tidx, i])
                tidx = self.parent[tidx, i]
        pause = 1
        return
    
    ########################################################################
    # Step 4: estimate trajectory-associated return periods
    ########################################################################
    # i embedded this step of post-processing in the class for QoL in later risk analysis and visualization
    # but i got rid of a lot of unnecessary code for the sake of simplicity and expansibility 
    def return_period_est(self):
        accu_rain = np.nanmean(np.multiply(self.rain_order, self.mask), axis = (2, 3))
        idx = np.arange(self.N)
        # associate traj id with total rainfall
        self.rank = list(zip(idx, accu_rain[:, -1]))
        # sort the trajs by total rainfall
        self.rank.sort(key = lambda x: x[1])
        # estimate the cdfs associated with each traj in terms of total rainfall
        self.cdf = np.zeros((self.N, ))
        self.cdf[0] = 1 / self.N * self.pq_ratio[self.rank[0][0]]
        for j in range(1, self.N):
            self.cdf[j] = self.cdf[j-1] + 1 / self.N * self.pq_ratio[self.rank[j][0]]
        self.rp = 1 / self.cdf
        return
    # to select data given a return period threshold
    def select_data(self, rp_thre = 100):
        return
    # to select subdaily data given a return period threshold
    def select_data_sd(self, rp_thre = 100):
        return