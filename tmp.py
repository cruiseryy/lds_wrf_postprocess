import xarray as xr

ds = xr.open_dataset('wrfrst_d01_2002-12-06')

var_ = ds.variables
for i in var_:
    print(i)
    pause = 1
pause = 1