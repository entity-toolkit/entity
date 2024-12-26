import nt2.read as nt2r
import matplotlib as mpl

data = nt2r.Data("magnetar.h5")

def plot (ti, data):
       (data.Bph*(data.r*np.sin(data.th))).isel(t=ti).polar.pcolor(
        norm=mpl.colors.Normalize(vmin=-0.075, vmax=0.075), 
        cmap="PuOr")