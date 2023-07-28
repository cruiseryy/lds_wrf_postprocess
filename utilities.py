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

# /home/climate/xp53/nas_home/LDS_WRF_OUTPUT/K=0.05/0_RAINNC.nc
#  
class post_analyzer:
    def __init__(self, path = '', var_list = ['RAINNC', 'SMCREL', 'SMOIS']) -> None:
        self.path = path

        return

    def var_read(self, var = ['RAINNC']):

        return

        
