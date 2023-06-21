import numpy as np
import os 
from collections import Counter
from time import time

from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.cm import get_cmap
import matplotlib.ticker as mticker

import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature
import cartopy.io.shapereader as shpreader

# import geopandas as gpd
# from netCDF4 import Dataset
# from wrf import (to_np, getvar, smooth2d, get_cartopy, cartopy_xlim,
#                  cartopy_ylim, latlon_coords)


class lds_plotter:
    def __init__(self, n = 128) -> None:
        self.path = os.getcwd()
        self.ic = np.loadtxt(self.path + '/vars/ic.txt').astype(int)
        self.bc_record = np.loadtxt(self.path + '/vars/bc_record.txt').astype(int)
        self.parent = np.loadtxt(self.path + '/vars/topo.txt').astype(int)
        self.R = np.loadtxt(self.path + '/vars/R.txt')
        self.weights = np.loadtxt(self.path + '/vars/weights.txt')
        self.n = n
        self.ref = 7.078
        return
    
    def freq_year_plot(self, nr=2, nc=3, T=6, tit = 'freq_year.pdf'):
        tmpmax = 0
        mp = {}
        for i in range(T):
            mp[i] = Counter(self.ic[:,i])
            tmpmax = max(tmpmax, max(list(mp[i].values())))
        
        fig, ax = plt.subplots(nrows = nr, ncols = nc, figsize=[nc*4, nr*4])
        for i in range(T):
            ri, ci = i // nc, i % nc
            ax[ri][ci].bar(list(mp[i].keys()), list(mp[i].values()))
            ax[ri][ci].set_title('({}) interval {}'.format(chr(ord('a') + i), i))
            ax[ri][ci].set_xlabel('Year')
            ax[ri][ci].set_ylabel('Freq')
            ax[ri][ci].set_ylim(0, 1.2*tmpmax)
            ax[ri][ci].xaxis.set_minor_locator(MultipleLocator(1))
        plt.tight_layout()
        plt.savefig(self.path + '/figures/' + tit)
        return
    
    def freq_traj_plot(self, nr=2, nc=3, T=6, tit = 'freq_traj.pdf'):
        tmpmax = 0
        mp = {}
        for i in range(T):
            tmp_ = [0]*self.n
            for j in range(self.n):
                tmp_[j] = self.find_root(j, i)
            mp[i] = Counter(tmp_)
            tmpmax = max(tmpmax, max(list(mp[i].values())))
        
        fig, ax = plt.subplots(nrows = nr, ncols = nc, figsize=[nc*4, nr*4])
        for i in range(T):
            ri, ci = i // nc, i % nc
            ax[ri][ci].bar(list(mp[i].keys()), list(mp[i].values()))
            ax[ri][ci].set_title('({}) interval {}'.format(chr(ord('a') + i), i))
            ax[ri][ci].set_xlabel('Traj #')
            ax[ri][ci].set_ylabel('Freq')
            ax[ri][ci].set_ylim(0, 1.2*tmpmax)
            # ax[ri][ci].xaxis.set_minor_locator(MultipleLocator(1))
        plt.tight_layout()
        plt.savefig(self.path + '/figures/' + tit)
        return
    
    def find_root(self, j, i):
        if i == 0: return j
        return self.find_root(self.parent[j, i], i-1)
    
    def traj_plot2(self, T = 5, tit = 'traj_alter.pdf'):
        # mp = np.zeros([self.n, 6*T])
        # for j in range(self.n): 
        #     mp[j, :] = np.mean(np.loadtxt(self.path + '/wrf_output/rainfall_' + str(j) + '.txt'), axis=1)
        mp = np.loadtxt('avg_rain.txt')
        pause = 1
        fig, ax = plt.subplots(figsize=[1.5*T+1,5])
        upp = np.zeros([25,])
        low = np.zeros([25,])
        for i in range(T):
            
            tmp_ = np.zeros([128, 5])

            for j in range(self.n):
                tmp_[j, :] = mp[j, 6*i:6*i+5] - mp[self.find_root(j, i), 0]
            
            upp[5*i:5*i+5] = np.quantile(tmp_, 1, axis=0)
            low[5*i:5*i+5] = np.quantile(tmp_, 0, axis=0)
        
        pause = 1

        return
    
    def traj_plot(self, T = 9, tit = 'traj_new.pdf'):
        mp = np.zeros([self.n, 6*T])
        for j in range(self.n): 
            mp[j, :] = np.mean(np.loadtxt(self.path + '/wrf_output/rainfall_' + str(j) + '.txt'), axis=1)
            print(j)
        # mp = np.loadtxt('avg_rain.txt')
        # pause = 1
        fig, ax = plt.subplots(figsize=[1.5*T+1,5])
        for i in range(T):
            xx = list(range((i+1)*5))
            zz = 1
            for j in range(self.n):
                tmprain = np.zeros([(i+1)*5, ])
                cur = j
                for i_ in range(i, -1, -1):
                    tmprain[i_*5:i_*5+5] = mp[cur, i_*6:i_*6+5]
                    if i_ != 0:
                        cur = self.parent[cur, i_]
                    pause = 1
                tmprain -= tmprain[0]
                if i != (T-1):
                    if j == 0 and i == 0:
                        ax.plot(xx, tmprain, color='lightskyblue', linewidth=5, alpha = 0.2, label='all trajs')
                    else:
                        ax.plot(xx, tmprain, color='lightskyblue', linewidth=5, alpha = 0.2)

                else:
                    ax.plot(xx, tmprain, color='lightskyblue', linewidth=5, alpha = 0.2)

        mark = 19
        legend_flag = 0
        for i in range(T):
            xx = list(range((i+1)*5))
            zz = 1
            for j in range(self.n):
                if self.find_root(j, i) != mark: continue

                tmprain = np.zeros([(i+1)*5, ])
                cur = j
                for i_ in range(i, -1, -1):
                    tmprain[i_*5:i_*5+5] = mp[cur, i_*6:i_*6+5]
                    if i_ != 0:
                        cur = self.parent[cur, i_]
                    pause = 1
                tmprain -= tmprain[0]
                if legend_flag == 0:
                    ax.plot(xx, tmprain, color='darkred', linewidth = 1.5, label='most cloned traj #19')
                    legend_flag = 1
                else:
                    ax.plot(xx, tmprain, color='darkred', linewidth = 1.5)
        
        ax.plot(xx, np.array(xx)*self.ref, 'k--', linewidth = 2, label='climatology')
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True)
        ax.set_xlabel('Days Elapsed')
        ax.set_ylabel('Cumulative rainfall [mm]')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.path + '/figures/' + tit)
        pause = 1
        return 
    
    
    # def traj_plot(self, T = 5, tit = 'traj.pdf'):
    #     mp = np.zeros([self.n, 6*T])
    #     for j in range(self.n): 
    #         mp[j, :] = np.mean(np.loadtxt(self.path + '/wrf_output/rainfall_' + str(j) + '.txt'), axis=1)
    #         # print(j)
    #     # mp2 = np.loadtxt('avg_rain.txt')
    #     # pause = 1
    #     fig, ax = plt.subplots(figsize=[1.5*T+1,5])
    #     for i in range(T):
    #         xx = list(range((i+1)*5))
    #         zz = 1
    #         for j in range(self.n):
    #             tmprain = np.zeros([(i+1)*5, ])
    #             cur = j
    #             for i_ in range(i, -1, -1):
    #                 tmprain[i_*5:i_*5+5] = mp[cur, i_*6:i_*6+5]
    #                 if i_ != 0:
    #                     cur = self.parent[cur, i_]
    #                 pause = 1
    #             tmprain -= tmprain[0]
    #             tmpcolor = 'r' if i == (T-1) else 'b' 
    #             if i != (T-1):
    #                 if j == 0 and i == 0:
    #                     ax.plot(xx, tmprain, color=tmpcolor, linewidth=5, alpha = 0.2, label='prev')
    #                 else:
    #                     ax.plot(xx, tmprain, color=tmpcolor, linewidth=5, alpha = 0.2)

    #             else:
    #                 if j == 0: 
    #                     ax.plot(xx, tmprain, color=tmpcolor, linewidth=2.5, alpha = 0.6, label='ending')
    #                 else:
    #                     ax.plot(xx, tmprain, color=tmpcolor, linewidth=2.5, alpha = 0.6)
                
    #             pause = 1
    #         pause = 1
    #     ax.plot(xx, np.array(xx)*self.ref, 'k-', linewidth = 3, label='climatology')
    #     ax.xaxis.set_minor_locator(MultipleLocator(1))
    #     ax.grid(True)
    #     ax.set_xlabel('Days Elapsed')
    #     ax.set_ylabel('Cumulative rainfall [mm]')
    #     ax.legend()
    #     plt.tight_layout()
    #     plt.savefig(self.path + '/figures/' + tit)
    #     pause = 1
    #     return 


# class plotter_sg:
#     def __init__(self) -> None:
#         ncfile = Dataset("/home/climate/xp53/testrun_4node/4node_testrun/wrfout_d01_1987-12-01_03:00:00")
#         rain0 = getvar(ncfile, 'RAINNC')
#         self.lats, self.lons = latlon_coords(rain0)
#         self.cart_proj = get_cartopy(rain0)
#         self.coastline = gpd.read_file('/home/climate/xp53/for_plotting/cropped/coastlines.shp')
#         return
    
#     def plotmap(self, ax, data, title):
#         self.coastline.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
#         ax.contour(to_np(self.lons), to_np(self.lats), to_np(data), 10, colors="none",
#                     transform=crs.PlateCarree())
#         basemap = ax.contourf(to_np(self.lons), to_np(self.lats), to_np(data), 10,
#                     transform=crs.PlateCarree(), cmap="jet")
#         ax.set_extent([np.min(self.lons), np.max(self.lons), np.min(self.lats), np.max(self.lats)], crs=crs.PlateCarree())
#         cbar = plt.colorbar(basemap, ax=ax, orientation='vertical', shrink=.5)
#         cbar.set_label('Rainfall [mm]')
#         cbar.mappable.set_clim(vmin=0, vmax=400) 
#         gl = ax.gridlines(crs=crs.PlateCarree(), draw_labels=True, linewidth=1.5, color='gray', alpha=0.5, linestyle='--')
#         gl.top_labels = False
#         gl.right_labels = False
#         gl.xlocator = mticker.FixedLocator([103.5, 103.7, 103.9, 104.1])
#         gl.ylocator = mticker.FixedLocator([1.1, 1.3, 1.5])
#         ax.set_title(title)
#         return


    # ncfile = Dataset("/home/climate/xp53/testrun_4node/4node_testrun/wrfout_d01_1987-12-01_03:00:00")
    # rain0 = getvar(ncfile, 'RAINNC')

    # lats, lons = latlon_coords(rain0)
    # cart_proj = get_cartopy(rain0)

    # coastline = gpd.read_file('/home/climate/xp53/for_plotting/cropped/coastlines.shp')

    # fig, ax = plt.subplots(nrows=1,ncols=2,figsize=(12, 6),subplot_kw={'projection': crs.PlateCarree()})

    # coastline.plot(ax=ax[0], facecolor='none', edgecolor='black', linewidth=.5)
    # ax[0].contour(to_np(lons), to_np(lats), to_np(rain0), 10, colors="none",
    #             transform=crs.PlateCarree())
    # ax[0].contourf(to_np(lons), to_np(lats), to_np(rain0), 10,
    #              transform=crs.PlateCarree(),
    #              cmap=get_cmap("jet"))
    # ax[0].set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)], crs=crs.PlateCarree())

    # ax[1].coastlines('10m', linewidth=0.5)
    # ax[1].contour(to_np(lons), to_np(lats), to_np(rain0), 10, colors="none",
    #             transform=crs.PlateCarree())
    # ax[1].contourf(to_np(lons), to_np(lats), to_np(rain0), 10,
    #              transform=crs.PlateCarree(),
    #              cmap=get_cmap("jet"))
    # ax[1].set_extent([np.min(lons), np.max(lons), np.min(lats), np.max(lats)], crs=crs.PlateCarree())


class plotter_world:
    def __init__(self) -> None:
        pass


if __name__ == '__main__':
    test = lds_plotter() 
    # test.freq_year_plot()
    # test.freq_traj_plot()
    test.traj_plot(tit = 'test.pdf')

    # pp = plotter_sg()
    pause = 1