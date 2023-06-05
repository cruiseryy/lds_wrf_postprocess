import numpy as np
from matplotlib import pyplot as plt
import os 
from collections import Counter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


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
        plt.tight_layout()
        plt.savefig(self.path + '/figures/' + tit)
        return
    
    def freq_traj_plot(self, nr=2, nc=3, T=6, tit = 'freq_traj.pdf'):
        tmpmax = 0
        mp = {}
        for i in range(T):
            if i == 0:
                mp[i] = Counter(list(range(self.n)))
            else:
                mp[i] = Counter(self.parent[:,i])
            tmpmax = max(tmpmax, max(list(mp[i].values())))
        
        fig, ax = plt.subplots(nrows = nr, ncols = nc, figsize=[nc*4, nr*4])
        for i in range(T):
            ri, ci = i // nc, i % nc
            ax[ri][ci].bar(list(mp[i].keys()), list(mp[i].values()))
            ax[ri][ci].set_title('({}) interval {}'.format(chr(ord('a') + i), i))
            ax[ri][ci].set_xlabel('Traj #')
            ax[ri][ci].set_ylabel('Freq')
            ax[ri][ci].set_ylim(0, 1.2*tmpmax)
        plt.tight_layout()
        plt.savefig(self.path + '/figures/' + tit)
        return
    
    def find_root(self, j, i):
        if i == 0: return j
        return self.find_root(self.parent[j, i], i-1)
    
    def traj_plot(self, T = 5, tit = 'traj.pdf'):
        mp = np.zeros([self.n, 6*T])
        for j in range(self.n): 
            mp[j, :] = np.mean(np.loadtxt(self.path + '/wrf_output/rainfall_' + str(j) + '.txt'), axis=1)
            # print(j)
        # mp2 = np.loadtxt('avg_rain.txt')
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
                tmpcolor = 'r' if i == (T-1) else 'b' 
                if i != (T-1):
                    if j == 0 and i == 0:
                        ax.plot(xx, tmprain, color=tmpcolor, linewidth=5, alpha = 0.2, label='prev')
                    else:
                        ax.plot(xx, tmprain, color=tmpcolor, linewidth=5, alpha = 0.2)

                else:
                    if j == 0: 
                        ax.plot(xx, tmprain, color=tmpcolor, linewidth=2.5, alpha = 0.6, label='ending')
                    else:
                        ax.plot(xx, tmprain, color=tmpcolor, linewidth=2.5, alpha = 0.6)
                
                pause = 1
            pause = 1
        ax.plot(xx, np.array(xx)*self.ref, 'k-', linewidth = 3, label='climatology')
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True)
        ax.set_xlabel('Days Elapsed')
        ax.set_ylabel('Cumulative rainfall [mm]')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.path + '/figures/' + tit)
        pause = 1
        return 

if __name__ == '__main__':
    test = lds_plotter() 
    # test.freq_year_plot()
    # test.freq_traj_plot()
    test.traj_plot()
    pause = 1