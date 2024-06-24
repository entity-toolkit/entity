import nt2.read as nt2r
import matplotlib as mpl

data = nt2r.Data("magnetosphere.h5")

def plot (ti, data):
    (data.N_2 - data.N_1).isel(t=ti).polar.pcolor(
        norm=mpl.colors.SymLogNorm(vmin=-5, vmax=5, linthresh=1e-2, linscale=1), 
        cmap="RdBu_r")