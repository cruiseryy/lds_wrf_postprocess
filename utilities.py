import numpy as np
import xarray as xr 

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
        return
    
    def var_read(self):
        for j in range(self.N):
            file = self.path + str(j) + '_RAINNC.nc'
            with xr.open_dataset(file) as ds:

                pause = 1
            

        return
    
    def topo_read(self, flag = 1):
        if flag == 1:
            # read topo
            return
        


        
if __name__ == '__main__':
    tmp = post_analyzer(path = "/home/climate/xp53/nas_home/LDS_WRF_OUTPUT/K=0.05/")
    tmp.var_read()