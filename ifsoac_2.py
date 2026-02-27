# Acelera funciones numericas (loops pesados)
from numba import jit, njit
# Escalar datos
from sklearn import preprocessing
# Calculo y manejo de datos (datashader: renderizar millones de puntos)
import numpy as np, pandas as pd, datashader as ds
# Funciones que convierten conteos de datos en imagenes visibles.
from datashader import transfer_functions as tf
# Paletas de color predefinidas
from datashader.colors import inferno, viridis, Hot
# Funcion para exportar imagenes a disco
from datashader.utils import export_image
# Diccionario global de paletas de color
from colorcet import palette

# Se registran paletas de color para que Datashader las reconozca
palette["viridis"] = viridis
palette["inferno"] = inferno
palette["Hot"] = Hot


"n-sided regular polygon, rotated an angle."
@jit(nopython=True)
# (numero de lados del poligono, rotacion del poligono)
def p_regular(nsides, angle=0):
    # Crea un arreglo vacio (nsides, 2) : coordenadas (x, y)
    res = np.empty((nsides,2))
    i = 0
    # Itera por cada vertice del poligono
    for d in range(nsides):
        # Calcula el angulo
        a = d * (2 * np.pi) / nsides + angle
        # Convierte angulo en coordenadas cartesianas
        res[i] = np.cos(a), np.sin(a)
        # Incrementa indice
        i += 1
        # Devuelve los vertices del poligono
    return res


class Ifsoac:
    def __init__(self, series=None, op=None):
        """Series can be a single series if length > 100. Otherwise a list of series is assumed.
        op is a dict with options."""
        self.op = {"nsides": 500, # Numero de vertices del poligono
                   # Peso del vertice en el Juego del Caos
                   "p": 0.5, # prob of choosing vertex instead of prev point, 0.5=avg, ~0=random walk/ts
                   "tam_punto": .1, # Tamaño de punto (solo matplotlib)
                   "ventana_plt": 9, # Tamaño ventana matplotlib
                   # Rotacion del poligono
                   "rotate": np.pi / 2, # ccw polygon rotation angle, 0=first vertex to the right
                   # Longitud de la serie aleatoria
                   "nrandom": 100000, # number of random choices of vertices
                   # Resolucion (el tamaño en pixeles) de la imagen final que Datashader va a generar
                   "ventana_ds": 700, # window size (datashader)
                   # Numero de columnas al mostrar imagenes
                   "cols_ds": 2, # on multi-IFSs, display them using this number of columns (datashader)
                   "background_ds": "black", # Color de fondo
                   "transform": None, # transformation to apply (a function name, see below)
                   "argv": None, # list of parameters (if needed) for the transformation
                   "cmap_ds": "inferno", # Mapa de color
                  "fatpoints": False, # Engrosar puntos
                   "x_range": None, # ranges for values
                   "y_range": None,
                  }
        if op is not None:
            self.op.update(op) # Si el usuario pasa opciones, sobrescribe las default
        #Si no hay datos:genera serie aleatoria
        series = [np.random.rand(self.op["nrandom"])] if series is None else series 
        # Si es una serie larga, la encapsula en una lista
        if len(series) > 100:
            series = [series]
        # Genera los vertices del poligono
        poligono = p_regular(self.op["nsides"], self.op["rotate"])
        # Escala valores al rango [0, nsides]
        scaler = preprocessing.MinMaxScaler(feature_range=(0, self.op["nsides"]))
        # Lista donde se guardan las trayectorias IFS
        self.ifs = [] 
        for serie in series:
            # Itera por cada serie
            m = np.max(serie) # to uniformly distribute series values along feat_range, we append m+epsilon to series,
            # Agrega un valor ligeramente mayor al maximo
            serie = np.append(serie, m+1e-10) # so it gets assigned "nsides" and then ...
  
            # Escala la serie, convierte a enteros y cada entero: vertice
            indices = scaler.fit_transform(serie.reshape(-1, 1)).flatten().astype(int)
            # Elimina indices negativos
            indices = indices[indices>=0]
            # Usa los indices para seleccionar vertices y se elimina el ultimo para no salir del rango
            self.ifs.append(poligono[indices[:-1]]) # ... we discart it so it doesn't get out of bounds

    # Ejecuta el juego del caos
    def jDC(self, serie=None):
        # Si no se especifica una serie, se usa automaticamente la primera serie almacenada en el objeto.
        if serie is None:
            serie = self.ifs[0]
        # Llama a la funcion jdc
        _jdc = jdc(serie, self.op["p"])
        # Si no hay transformacion: regresa puntos
        if self.op["transform"] is None:
            return _jdc
        # Aplica transformacion geometrica
        else:
            return self.op["transform"](_jdc, self.op["argv"])
            
    def _images_ds(self):
        "Called by plot(), return a list of produced images (with same op)."
        # Le quita el borde a las imagenes generadas por Datashader
        ds.transfer_functions.Image.border = 0
        # Crea una lista vacia donde se guardaran las imagenes resultantes
        res = []
        # Crea una lista
        cols = list("xy")
        # Guarda el tamaño de la ventana de la imagen Datashader
        vent = self.op["ventana_ds"]
        # Recorre todas las series ya procesadas y guardadas en self.ifs
        for serie in self.ifs:
            # Pregunta si hay un rango para x
            if self.op["x_range"]:
                # Crea una hoja donde se va a dibujar el fractal (ancho, alto, limites del grafico)
                cv = ds.Canvas(plot_width=vent, plot_height=vent, x_range=self.op["x_range"], 
                               y_range=self.op["y_range"])
            # Si no se definio rango, Datashader calcula automaticamente el rango adecuado
            else:
                cv = ds.Canvas(plot_width=vent, plot_height=vent)
            # Genera los puntos finales del juego del caos y se convierten a DataFrame con columnas x y y
            df = pd.DataFrame(self.jDC(serie), columns=cols)
            # Si la transformacion elegida es la funcion escher, se hace un paso extra.
            if self.op["transform"] == escher:
                # Agrega al DataFrame los puntos de un poligono regular con 10000 vertices
                df = pd.concat([df, pd.DataFrame(p_regular(10000), columns=cols)])
            # Datashader no dibuja punto por punto como matplotlib (cuenta cuantos puntos caen en cada pixel)
            agg = cv.points(df, *cols) 
            # Convierte esa matriz agg en una imagen real
            r = tf.shade(agg, cmap=palette[self.op["cmap_ds"]])
            # Si se activo fatpoints=True, los puntos se hacen mas grandes
            if self.op["fatpoints"]:
                r = tf.spread(r)
            # Si el usuario eligio un color de fondo, lo aplica
            if self.op["background_ds"] is not None:
                r = tf.set_background(r, self.op["background_ds"])
            # Guarda la imagen generada en la lista res
            res.append(r)
        return res

    # Metodo para mostrar las imagenes
    def plot(self, res=None):
        "Colaboratory & others, to show an image call this func at the last line in a cell ."
        # Si no se pasan imagenes (res=None), se generan automaticamente
        imgs = self._images_ds() if (res is None) else res
        # Marca que ya se grafico algo
        self.plotted = True
        # Muestra las imágenes usando Datashader
        return tf.Images(*imgs).cols(self.op["cols_ds"])

    # Funcion para exportar las imagenes a archivos
    def export_images(self, filename_prefix="filename_", fmt=".png", export_path="./"):
        # Genera las imagenes
        imgs = self._images_ds()
        # Recorre cada imagen con su indice
        for i, img in enumerate(imgs):
            # Guarda la imagen como archivo
            export_image(img=img, filename=filename_prefix + str(i), fmt=fmt, export_path=export_path)
        return self.plot(imgs)

    def __repr__(self):
        return str(self.op)
    

    

@jit(nopython=True)
def jdc(series, p):
    "Uniformly distributed random series produce regular Chaos Game."
    # Punto inicial en el origen
    x, y = 0, 0
    # Crea un arreglo vacío del mismo tamaño que series para guardar resultados
    res = np.empty_like(series)
    i = 0
    # Recorre cada punto (vértice) de la serie
    for a, b in series:
        #x, y = (a + x) * p, (b + y) * p
        x, y = a * p + x * (1 - p), b * p + y * (1 - p)
        # Guarda el nuevo punto en el arreglo
        res[i] = x, y
        # Incrementa contador
        i += 1
    return res


## TRANSFORM
# p.T separa columnas x,y.
def norm(p):
    x, y = p.T
    return np.sqrt(x * x + y * y) # Calcula la norma euclidiana

# Es una transformación estereográfica (cambia la geometría del fractal)
def estereografica(p, argv):
    return (p.T / norm(p) ** 2).T

# Hace una transformación tipo proyección conforme
def escher(p, argv):
    # return p/norm(p/norm(p)-p)
    u, v = p.T
    k = 1 + u * u + v * v
    return np.array([(2 * u) / k, (2 * v) / k]).T # Genera un efecto tipo “círculo / disco” estilo Escher

# Si no le dan parámetros, usa desplazamientos por defecto
def antipolar(p, argv):
    argv = argv if argv else {"delta_x": np.pi, "delta_y": np.pi}
    # Separa x,y
    x, y = p.T
    # Calcula el ángulo polar del punto desplazado
    alfa = np.arctan2(y + argv["delta_y"], x + argv["delta_x"])
    # calcula el radio
    r = norm(p)
    # Regresa puntos en coordenadas polares: (ángulo, radio)
    return np.array((alfa, r)).T


## MAPEOS toman y devuelven array de puntos
# Mapa logístico clásico (caótico si el parámetro es 4)
def logistica(x):
    return 4 * x * (1 - x)

# Genera una serie caótica iterando una función.
def iterar(funcion=logistica, x0=0.3, N=1000000, N0=1000):
    # Compila la función con numba
    funcion = njit(funcion)
    # Función interna compilada
    @njit
    def inner(funcion, x0, N, N0):
        # inicializa
        y = x0
        # Corre N0 iteraciones para “entrar al régimen caótico”
        for i in range(N0):
            y = funcion(y)
        # ya se generó una x0 (y) "caótica"
        ll = list() # lista vacía para guardar resultados
        # Genera N valores y los guarda
        for _ in range(N):
            y = funcion(y)
            ll.append(y)
        # Convierte lista a array
        return np.array(ll)
    # Devuelve el array final
    return inner(funcion, x0, N, N0)

# Mapa tent: otro sistema dinámico caótico
def tent(x):
    _lambda = 0.999
    return 2 * x * _lambda if x < 0.5 else (2 - 2 * x) * _lambda

# Estos generan trayectorias de atractores caóticos
# 1D
def lorenz_array(N=500000):
    return lorenz_array3d(N).T[0]

# 3D
@jit(nopython=True)
def lorenz_array3d(N=500000):
    # Condiciones iniciales
    x0 = 1
    y0 = 1
    z0 = 3
    # Paso de integración
    h = 0.01
    # Parámetros del sistema Lorenz
    sigma = 10.0    # a
    beta = 8 / 3.0  # c
    ro = 28.0       # b
    # Matriz donde se guardan los puntos
    res = np.empty((N, 3))
    i = 0
    for i in range(N):
        # Ecuaciones discretizadas del Lorenz
        x1 = h * (sigma * (y0 - x0)) + x0
        y1 = h * (x0 * (ro - z0) - y0) + y0
        z1 = h * (x0 * y0 - beta * z0) + z0
        # Guarda el punto y actualiza x0,y0,z0
        res[i] = x0, y0, z0 = x1, y1, z1
        i += 1
    return res


# 1D
def rossler_array(N=500000):
    # genera x,y,z con ecuaciones de Rössler
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
    # Convierte entrada en DataFrame
    df = pd.DataFrame(df)
    # Ordena los valores
    lo = df.sort_values(0)  # sorted by values
    # Cuántas veces debe repetirse cada bin
    times = len(lo) / nbins  # number of times a bin label will be repeated
    # Crea etiquetas de bins repetidas
    s = pd.Series(np.arange(nbins).repeat(times))
    # Regresa la serie de bins reordenada al orden original
    return pd.Series(s, index=lo.index).fillna(0.).to_numpy()


# detects if a path from the list exists and changes to it (for running inside Colaboratory)
# Importa librerías
import sys, os
# Revisa rutas posibles
for spec in ["/content/drive/MyDrive/Projects/my_path"]:
    # Si la carpeta existe
    if os.path.isdir(os.path.expanduser(spec)):
        # Cambia el directorio de trabajo y termina el ciclo
        os.chdir(spec)
        break