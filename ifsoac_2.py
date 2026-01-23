from numba import jit, njit
from sklearn import preprocessing
import numpy as np, pandas as pd, datashader as ds
from datashader import transfer_functions as tf
from datashader.colors import inferno, viridis, Hot
from datashader.utils import export_image
from colorcet import palette

palette["viridis"] = viridis
palette["inferno"] = inferno
palette["Hot"] = Hot


"n-sided regular polygon, rotated an angle."
@jit(nopython=True)
def p_regular(nsides, angle=0):
    res = np.empty((nsides,2))
    i = 0
    for d in range(nsides):
        a = d * (2 * np.pi) / nsides + angle
        res[i] = np.cos(a), np.sin(a)
        i += 1
    return res


class Ifsoac:
    def __init__(self, series=None, op=None):
        """Series can be a single series if length > 100. Otherwise a list of series is assumed.
        op is a dict with options."""
        self.op = {"nsides": 500, #polygon sides
                   "p": 0.5, # prob of choosing vertex instead of prev point, 0.5=avg, ~0=random walk/ts
                   "tam_punto": .1, #point size (matplotlib)
                   "ventana_plt": 9, # window size (matplotlib)
                   "rotate": np.pi / 2, # ccw polygon rotation angle, 0=first vertex to the right
                   "nrandom": 100000, # number of random choices of vertices
                   "ventana_ds": 700, # window size (datashader)
                   "cols_ds": 2, # on multi-IFSs, display them using this number of columns (datashader)
                   "background_ds": "black", # bg color (datashader)
                   "transform": None, # transformation to apply (a function name, see below)
                   "argv": None, # list of parameters (if needed) for the transformation
                   "cmap_ds": "inferno",
                  "fatpoints": False, # big fat points
                   "x_range": None, # ranges for values
                   "y_range": None,
                  }
        if op is not None:
            self.op.update(op)
        series = [np.random.rand(self.op["nrandom"])] if series is None else series
        if len(series) > 100:
            series = [series]
        poligono = p_regular(self.op["nsides"], self.op["rotate"])
        scaler = preprocessing.MinMaxScaler(feature_range=(0, self.op["nsides"]))
        self.ifs = []
        for serie in series:
            m = np.max(serie) # to uniformly distribute series values along feat_range, we append m+epsilon to series,
            serie = np.append(serie, m+1e-10) # so it gets assigned "nsides" and then ...
  
            indices = scaler.fit_transform(serie.reshape(-1, 1)).flatten().astype(int)
            indices = indices[indices>=0]
            self.ifs.append(poligono[indices[:-1]]) # ... we discart it so it doesn't get out of bounds

    def jDC(self, serie=None):
        if serie is None:
            serie = self.ifs[0]
        _jdc = jdc(serie, self.op["p"])
        if self.op["transform"] is None:
            return _jdc
        else:
            return self.op["transform"](_jdc, self.op["argv"])
            
    def _images_ds(self):
        "Called by plot(), return a list of produced images (with same op)."
        ds.transfer_functions.Image.border = 0
        res = []
        cols = list("xy")
        vent = self.op["ventana_ds"]
        for serie in self.ifs:
            if self.op["x_range"]:
                cv = ds.Canvas(plot_width=vent, plot_height=vent, x_range=self.op["x_range"], 
                               y_range=self.op["y_range"])
            else:
                cv = ds.Canvas(plot_width=vent, plot_height=vent)
            df = pd.DataFrame(self.jDC(serie), columns=cols)
            if self.op["transform"] == escher:
                df = pd.concat([df, pd.DataFrame(p_regular(10000), columns=cols)])
            agg = cv.points(df, *cols)
            r = tf.shade(agg, cmap=palette[self.op["cmap_ds"]])
            if self.op["fatpoints"]:
                r = tf.spread(r)
            if self.op["background_ds"] is not None:
                r = tf.set_background(r, self.op["background_ds"])
            res.append(r)
        return res

    def plot(self, res=None):
        "Colaboratory & others, to show an image call this func at the last line in a cell ."
        imgs = self._images_ds() if (res is None) else res
        self.plotted = True
        return tf.Images(*imgs).cols(self.op["cols_ds"])

    def export_images(self, filename_prefix="filename_", fmt=".png", export_path="./"):
        imgs = self._images_ds()
        for i, img in enumerate(imgs):
            export_image(img=img, filename=filename_prefix + str(i), fmt=fmt, export_path=export_path)
        return self.plot(imgs)

    def __repr__(self):
        return str(self.op)
    

    
    
@jit(nopython=True)
def jdc(series, p):
    "Uniformly distributed random series produce regular Chaos Game."
    x, y = 0, 0
    res = np.empty_like(series)
    i = 0
    for a, b in series:
        #x, y = (a + x) * p, (b + y) * p
        x, y = a * p + x * (1 - p), b * p + y * (1 - p)
        res[i] = x, y
        i += 1
    return res


## TRANSFORM

def norm(p):
    x, y = p.T
    return np.sqrt(x * x + y * y)


def estereografica(p, argv):
    return (p.T / norm(p) ** 2).T


def escher(p, argv):
    # return p/norm(p/norm(p)-p)
    u, v = p.T
    k = 1 + u * u + v * v
    return np.array([(2 * u) / k, (2 * v) / k]).T


def antipolar(p, argv):
    argv = argv if argv else {"delta_x": np.pi, "delta_y": np.pi}
    x, y = p.T
    alfa = np.arctan2(y + argv["delta_y"], x + argv["delta_x"])
    r = norm(p)
    return np.array((alfa, r)).T


## MAPEOS toman y devuelven array de puntos

def logistica(x):
    return 4 * x * (1 - x)

def iterar(funcion=logistica, x0=0.3, N=1000000, N0=1000):
    funcion = njit(funcion)
    @njit
    def inner(funcion, x0, N, N0):
        y = x0
        for i in range(N0):
            y = funcion(y)
        # ya se generó una x0 (y) "caótica"
        ll = list()
        for _ in range(N):
            y = funcion(y)
            ll.append(y)
        return np.array(ll)
    return inner(funcion, x0, N, N0)


def tent(x):
    _lambda = 0.999
    return 2 * x * _lambda if x < 0.5 else (2 - 2 * x) * _lambda

# 1D
def lorenz_array(N=500000):
    return lorenz_array3d(N).T[0]

# 3D
@jit(nopython=True)
def lorenz_array3d(N=500000):
    x0 = 1
    y0 = 1
    z0 = 3
    h = 0.01
    sigma = 10.0    # a
    beta = 8 / 3.0  # c
    ro = 28.0       # b
    res = np.empty((N, 3))
    i = 0
    for i in range(N):
        x1 = h * (sigma * (y0 - x0)) + x0
        y1 = h * (x0 * (ro - z0) - y0) + y0
        z1 = h * (x0 * y0 - beta * z0) + z0
        res[i] = x0, y0, z0 = x1, y1, z1
        i += 1
    return res


# 1D
def rossler_array(N=500000):
    return rossler_array3d(N).T[0]

# 3D
@jit(nopython=True)
def rossler_array3d(N=500000):
    x0 = 1
    y0 = 1
    z0 = 3
    h = 0.01
    sigma = 0.1    # a
    ro = 0.1       # b
    beta = 14      # c
    res = np.empty((N, 3))
    i = 0
    for i in range(N):
        x1 = h * (-y0 - z0) + x0
        y1 = h * (x0 + sigma * y0) + y0
        z1 = h * (ro + z0 * (x0 - beta)) + z0
        res[i] = x0, y0, z0 = x1, y1, z1
        i += 1
    return res


## INITS
"""The data is "stored" in bins and each bin corresponds to a vertex in the nsides-side polygon.
This func. makes sure that each bin has the same number of data points.
In reality, the returned bins contain data indexes (bin index) having a uniform frequency and order 
similar to that of the original data. If data length is not a multiple of nbins, extra data places are
filled with 0."""
def same_amount_bins(df, nbins=500):
    df = pd.DataFrame(df)
    lo = df.sort_values(0)  # sorted by values
    times = len(lo) / nbins  # number of times a bin label will be repeated
    s = pd.Series(np.arange(nbins).repeat(times))
    return pd.Series(s, index=lo.index).fillna(0.).to_numpy()


# detects if a path from the list exists and changes to it (for running inside Colaboratory)
import sys, os
for spec in ["/content/drive/MyDrive/Projects/my_path"]:
    if os.path.isdir(os.path.expanduser(spec)):
        os.chdir(spec)
        break