# -*- coding: utf-8 -*-
"""
FITS LAB - OPERAZIONI
=====================
Ogni operazione è un blocco con i suoi parametri e una funzione che riceve
un array float32 e restituisce l'array nuovo. Il registro OPERAZIONI è il solo
punto da toccare per aggiungerne di nuove (versioni successive: maschere,
livelli, deconvoluzione, rimozione delle stelle, colore...).

Due famiglie:
  - "science": lavorano sui dati lineari e li lasciano misurabili
    (ritaglio, binning, pixel difettosi, fondo, rumore);
  - "display": trasformano i dati per la visione (stiramento, curve, tavolozza).
    Dal primo passo "display" in poi il risultato è solo estetico.

Unità dei parametri in pixel: pixel dell'immagine che si otterrebbe a piena
qualità in quel punto della ricetta (così la ricetta vale identica sull'anteprima
ridotta e sul file intero). Coordinate disegnate (rettangoli, punti): pixel del
file originale intero.

© 2026 Edoardo Battistel - FITS Lab
"""

import math

import numpy as np
from scipy import ndimage as ndi

DTYPE = np.float32


# =============================================================================
# GEOMETRIA (dove si trova l'array rispetto al file originale)
# =============================================================================
class Geometria:
    """x_orig = s * x_arr + ox + (s - 1) / 2 (pixel 0-based).
    s = pixel originali per pixel dell'array (riduzione reale);
    sn = la stessa cosa nella lavorazione a piena qualità (riduzione «nominale»)."""

    def __init__(self, oy=0.0, ox=0.0, s=1.0, sn=1.0):
        self.oy, self.ox, self.s, self.sn = float(oy), float(ox), float(s), float(sn)

    def copia(self):
        return Geometria(self.oy, self.ox, self.s, self.sn)

    def orig_da_arr(self, x, y):
        return (self.s * np.asarray(x) + self.ox + (self.s - 1) / 2,
                self.s * np.asarray(y) + self.oy + (self.s - 1) / 2)

    def arr_da_orig(self, x, y):
        return ((np.asarray(x) - self.ox - (self.s - 1) / 2) / self.s,
                (np.asarray(y) - self.oy - (self.s - 1) / 2) / self.s)

    def px(self, valore):
        """Parametro in pixel «nominali» -> pixel dell'array reale."""
        return float(valore) * self.sn / self.s

    def as_dict(self):
        return dict(oy=self.oy, ox=self.ox, s=self.s, sn=self.sn)


class Contesto:
    """Informazioni che viaggiano lungo la ricetta."""

    def __init__(self, geom, meta=None, lineare=True, progresso=None, anteprima=False):
        self.geom = geom
        self.meta = meta or {}
        self.lineare = lineare
        self.progresso = progresso
        self.anteprima = anteprima
        self.modello = None          # ultimo modello del fondo (da mostrare)
        self.note = []               # righe per HISTORY
        self.avvisi = set()          # es. "denoised"
        self.norm = None             # (lo, hi) usati dallo stiramento


# =============================================================================
# STATISTICHE ROBUSTE
# =============================================================================
def campione(a, n=400_000):
    """Al massimo n valori finiti di a (presi a passo regolare), in float64."""
    a = np.asarray(a)
    if a.ndim == 2 and a.size > n * 2:
        p = max(1, int(math.sqrt(a.size / n)))
        flat = a[::p, ::p].ravel()
    else:
        flat = a.ravel()
        if flat.size > n * 2:
            flat = flat[:: flat.size // n]
    v = flat[np.isfinite(flat)]
    return v.astype(np.float64)


def statistiche(a):
    """(mediana, sigma robusta) del fondo, con un taglio a 3 sigma ripetuto."""
    v = campione(a)
    if v.size == 0:
        return 0.0, 1.0
    m = float(np.median(v))
    s = float(1.4826 * np.median(np.abs(v - m)))
    for _ in range(4):
        if s <= 0:
            break
        v = v[np.abs(v - m) < 3 * s]
        if v.size < 50:
            break
        m = float(np.median(v))
        s = float(1.4826 * np.median(np.abs(v - m)))
    if not s > 0:
        s = float(np.std(v)) if v.size else 1.0
        s = s if s > 0 else 1.0
    return m, s


def rumore_pixel(a):
    """Rumore da pixel a pixel (differenze tra vicini): non risente di gradienti e nebulose."""
    b = a if a.size <= 4_000_000 else a[: 2000, : 2000]
    d = (b[:, 1:] - b[:, :-1]).ravel()
    d = d[np.isfinite(d)].astype(np.float64)
    if d.size < 20:
        return statistiche(a)[1]
    med = np.median(d)
    s = 1.4826 * np.median(np.abs(d - med)) / math.sqrt(2)
    return float(s) if s > 0 else statistiche(a)[1]


def _riempi_nan(a, valore):
    return np.where(np.isfinite(a), a, valore)


def gauss_nan(a, sigma):
    """Sfocatura gaussiana che ignora i NaN (convoluzione normalizzata)."""
    ok = np.isfinite(a)
    num = ndi.gaussian_filter(np.where(ok, a, 0).astype(np.float64), sigma, mode="nearest")
    den = ndi.gaussian_filter(ok.astype(np.float64), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~ok] = np.nan
    return out.astype(DTYPE)


# =============================================================================
# PARAMETRI
# =============================================================================
class P:
    """Definizione di un parametro (il testo viene da fitslab_testi)."""

    def __init__(self, id, tipo, default, minimo=None, massimo=None, passo=None, scelte=None, px=False,
                 visibile_se=None):
        self.id, self.tipo, self.default = id, tipo, default
        self.minimo, self.massimo, self.passo = minimo, massimo, passo
        self.scelte = scelte or []
        self.px = px
        self.visibile_se = visibile_se        # (id_parametro, valori ammessi)


class Operazione:
    id = ""
    famiglia = "science"
    parametri = []

    def predefiniti(self):
        return {p.id: (list(p.default) if isinstance(p.default, list) else p.default) for p in self.parametri}

    def descrivi(self, par):
        """Riga breve (inglese) per HISTORY e per l'elenco dei passi."""
        return self.id

    def applica(self, a, par, ctx):
        raise NotImplementedError


# =============================================================================
# RITAGLIO
# =============================================================================
def riquadro_dati(a, soglia_vuoto=0.0):
    """(y0, y1, x0, x1) del rettangolo che contiene tutti i pixel con dati."""
    ok = np.isfinite(a)
    righe = np.flatnonzero(ok.any(axis=1))
    col = np.flatnonzero(ok.any(axis=0))
    if righe.size == 0 or col.size == 0:
        return 0, a.shape[0], 0, a.shape[1]
    return int(righe[0]), int(righe[-1]) + 1, int(col[0]), int(col[-1]) + 1


def rettangolo_pieno(a, max_lato=600):
    """Rettangolo più grande senza NaN (su una versione ridotta, poi riportato)."""
    ny, nx = a.shape
    f = max(1, int(math.ceil(max(ny, nx) / max_lato)))
    ok = np.isfinite(a[: ny // f * f, : nx // f * f]).reshape(ny // f, f, nx // f, f).all(axis=(1, 3))
    h = np.zeros(ok.shape[1], dtype=np.int64)
    migliore = (0, 0, 0, 0, 0)
    for i in range(ok.shape[0]):
        h = np.where(ok[i], h + 1, 0)
        pila = []
        for j in range(len(h) + 1):
            altezza = h[j] if j < len(h) else 0
            inizio = j
            while pila and pila[-1][1] >= altezza:
                inizio, hh = pila.pop()
                area = hh * (j - inizio)
                if area > migliore[0]:
                    migliore = (area, i - hh + 1, i + 1, inizio, j)
            pila.append((inizio, altezza))
    _, y0, y1, x0, x1 = migliore
    if migliore[0] == 0:
        return riquadro_dati(a)
    return y0 * f, y1 * f, x0 * f, x1 * f


class Ritaglio(Operazione):
    id = "crop"
    famiglia = "science"
    parametri = [
        P("mode", "choice", "rect", scelte=["rect", "data", "full"]),
        P("rect", "rect", [0, 0, 0, 0], visibile_se=("mode", ["rect"])),   # x0, y0, x1, y1 originali
    ]

    def descrivi(self, par):
        if par["mode"] == "data":
            return "crop: trim empty borders"
        if par["mode"] == "full":
            return "crop: largest rectangle with data everywhere"
        x0, y0, x1, y1 = par["rect"]
        return f"crop: x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f} (original pixels)"

    def applica(self, a, par, ctx):
        ny, nx = a.shape
        if par["mode"] == "data":
            y0, y1, x0, x1 = riquadro_dati(a)
        elif par["mode"] == "full":
            y0, y1, x0, x1 = rettangolo_pieno(a)
        else:
            rx0, ry0, rx1, ry1 = par["rect"]
            if rx1 <= rx0 or ry1 <= ry0:
                return a
            ax0, ay0 = ctx.geom.arr_da_orig(min(rx0, rx1), min(ry0, ry1))
            ax1, ay1 = ctx.geom.arr_da_orig(max(rx0, rx1), max(ry0, ry1))
            x0, y0 = int(max(0, math.floor(ax0 + 0.5))), int(max(0, math.floor(ay0 + 0.5)))
            x1, y1 = int(min(nx, math.floor(ax1 + 0.5) + 1)), int(min(ny, math.floor(ay1 + 0.5) + 1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return a
        g = ctx.geom
        g.ox += g.s * x0
        g.oy += g.s * y0
        return np.ascontiguousarray(a[y0:y1, x0:x1])


# =============================================================================
# BINNING (riduzione della risoluzione che conserva il flusso)
# =============================================================================
def modo_binning_auto(unita):
    u = str(unita or "").upper().replace(" ", "")
    if any(k in u for k in ("/SR", "BEAM", "ARCSEC", "MAG", "K_CMB", "KCMB", "K_RJ")) or u in ("K", "MK", "UK"):
        return "mean"
    if any(k in u for k in ("COUNT", "CT", "DN", "ADU", "ELECTRON", "E-", "PHOTON", "/PIX", "JY/PIXEL")):
        return "sum"
    return "mean"


def riduci_blocchi(a, f, modo="mean"):
    f = int(f)
    if f <= 1:
        return a
    ny, nx = a.shape[0] // f * f, a.shape[1] // f * f
    b = a[:ny, :nx].reshape(ny // f, f, nx // f, f)
    with np.errstate(invalid="ignore"), _silenzio():
        m = np.nanmean(b, axis=(1, 3))
        if modo == "sum":
            m = m * (f * f)
    return m.astype(DTYPE)


class _silenzio:
    def __enter__(self):
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *a):
        self._w.__exit__(*a)


class Binning(Operazione):
    id = "bin"
    famiglia = "science"
    parametri = [
        P("factor", "choice", 2, scelte=[2, 3, 4, 5, 6, 8]),
        P("mode", "choice", "auto", scelte=["auto", "mean", "sum"]),
    ]

    def descrivi(self, par):
        return f"bin {par['factor']}x{par['factor']} ({par['mode']})"

    def applica(self, a, par, ctx):
        g = ctx.geom
        nominale = g.sn * int(par["factor"])
        f = max(1, int(round(nominale / g.s)))
        modo = par["mode"]
        if modo == "auto":
            modo = modo_binning_auto(ctx.meta.get("unit"))
        g.sn = nominale
        if f > 1:
            a = riduci_blocchi(a, f, modo if not ctx.anteprima else "mean")
            g.s *= f
        ctx.meta["bin_mode"] = modo
        return a


# =============================================================================
# PIXEL DIFETTOSI (caldi/freddi, raggi cosmici, buchi)
# =============================================================================
def pixel_caldi(a, k, sigma, frazione=0.6):
    """Maschere dei pixel isolati: superano TUTTI gli 8 vicini di k sigma e di più
    della 'frazione' della loro altezza sul fondo (il picco di una stella no: i vicini
    sono quasi alti come lui). Lo stesso al contrario per i pixel freddi."""
    fondo = np.nanmedian(a) if np.isfinite(a).any() else 0
    pieno = _riempi_nan(a, fondo)
    imp = np.ones((3, 3), bool)
    imp[1, 1] = False
    vmax = ndi.maximum_filter(pieno, footprint=imp, mode="nearest")
    vmin = ndi.minimum_filter(pieno, footprint=imp, mode="nearest")
    caldi = ((pieno - vmax) > k * sigma) & ((pieno - vmax) > frazione * (pieno - fondo))
    freddi = ((vmin - pieno) > k * sigma) & ((vmin - pieno) > frazione * (fondo - pieno))
    return caldi & np.isfinite(a), freddi & np.isfinite(a)


def raggi_cosmici(a, sigma, sigclip=5.0, objlim=3.0):
    """L.A.Cosmic semplificato (van Dokkum 2001): laplaciano dell'immagine
    ingrandita 2x, confrontato con il rumore e con la struttura fine."""
    pieno = _riempi_nan(a, np.nanmedian(a) if np.isfinite(a).any() else 0).astype(np.float64)
    grande = np.repeat(np.repeat(pieno, 2, axis=0), 2, axis=1)
    lap = ndi.convolve(grande, np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], float), mode="nearest")
    lap = np.clip(lap, 0, None)
    ny, nx = pieno.shape
    lap = lap.reshape(ny, 2, nx, 2).mean(axis=(1, 3))
    s = lap / (2 * sigma)
    s = s - ndi.median_filter(s, size=5, mode="nearest")
    m3 = ndi.median_filter(pieno, size=3, mode="nearest")
    fine = m3 - ndi.median_filter(m3, size=7, mode="nearest")
    fine = np.maximum(fine, 0.01 * sigma)
    cr = (s > sigclip) & ((lap / fine) > objlim) & np.isfinite(a)
    if cr.any():
        cr = ndi.binary_dilation(cr, iterations=1) & (s > sigclip * 0.3)
    return cr


def sostituisci_con_vicini(a, maschera, lato=5):
    """Pixel della maschera -> mediana dei vicini buoni (finestra lato x lato)."""
    if not maschera.any():
        return a
    b = a.astype(DTYPE, copy=True)
    ny, nx = b.shape
    r = lato // 2
    ys, xs = np.nonzero(maschera)
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    dy, dx = dy.ravel(), dx.ravel()
    lavoro = b.copy()
    lavoro[maschera] = np.nan
    for i in range(0, ys.size, 200_000):
        yy = np.clip(ys[i:i + 200_000, None] + dy, 0, ny - 1)
        xx = np.clip(xs[i:i + 200_000, None] + dx, 0, nx - 1)
        with _silenzio():
            med = np.nanmedian(lavoro[yy, xx], axis=1)
        b[ys[i:i + 200_000], xs[i:i + 200_000]] = med
    return b


def buchi_interni(a, lato_max):
    """NaN che non toccano il bordo esterno vuoto e sono più piccoli di lato_max."""
    nan = ~np.isfinite(a)
    if not nan.any():
        return nan
    lab, n = ndi.label(nan)
    if n == 0:
        return nan & False
    bordo = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    oggetti = ndi.find_objects(lab)
    tieni = np.zeros(n + 1, bool)
    for i, sl in enumerate(oggetti, start=1):
        if i in bordo or sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) <= lato_max:
            tieni[i] = True
    return tieni[lab]


def riempi_buchi(a, maschera):
    """Riempie i buchi con un'interpolazione che va dai bordi verso l'interno."""
    if not maschera.any():
        return a
    b = a.astype(np.float64, copy=True)
    restano = maschera.copy()
    for _ in range(200):
        if not restano.any():
            break
        ok = np.isfinite(b)
        num = ndi.gaussian_filter(np.where(ok, b, 0), 1.5)
        den = ndi.gaussian_filter(ok.astype(float), 1.5)
        nuovi = restano & (den > 0.05)
        if not nuovi.any():
            break
        b[nuovi] = num[nuovi] / den[nuovi]
        restano &= ~nuovi
    return b.astype(DTYPE)


class PixelDifettosi(Operazione):
    id = "pixels"
    famiglia = "science"
    parametri = [
        P("hot", "bool", True),
        P("k", "float", 8.0, 3.0, 30.0, 0.5),
        P("cosmic", "bool", False),
        P("sigclip", "float", 5.0, 3.0, 15.0, 0.5, visibile_se=("cosmic", [True])),
        P("holes", "bool", True),
        P("hole_size", "int", 25, 2, 400, 1, px=True, visibile_se=("holes", [True])),
    ]

    def descrivi(self, par):
        v = []
        if par["hot"]:
            v.append(f"hot/cold pixels > {par['k']:g} sigma")
        if par["cosmic"]:
            v.append(f"cosmic rays (sigclip {par['sigclip']:g})")
        if par["holes"]:
            v.append(f"holes up to {par['hole_size']} px filled")
        return "bad pixels: " + (", ".join(v) or "none")

    def applica(self, a, par, ctx):
        s = rumore_pixel(a)
        out = a.astype(DTYPE, copy=True)
        tot = 0
        if par["hot"]:
            c, f = pixel_caldi(out, float(par["k"]), s)
            tot += int(c.sum() + f.sum())
            out = sostituisci_con_vicini(out, c | f, 3)
        if par["cosmic"]:
            cr = raggi_cosmici(out, s, float(par["sigclip"]))
            tot += int(cr.sum())
            out = sostituisci_con_vicini(out, cr, 5)
        if par["holes"]:
            lato = max(1, int(round(ctx.geom.px(par["hole_size"]))))
            bu = buchi_interni(out, lato)
            tot += int(bu.sum())
            out = riempi_buchi(out, bu)
        ctx.meta["pixels_fixed"] = tot
        return out


# =============================================================================
# FONDO E GRADIENTE
# =============================================================================
def maschera_sorgenti(a, m, s, k=3.0, cresci=2):
    liscia = gauss_nan(a, 1.5)
    mask = np.isfinite(liscia) & (liscia > m + k * s * 0.6)
    if cresci > 0 and mask.any():
        mask = ndi.binary_dilation(mask, iterations=int(cresci))
    return mask


def campioni_griglia(a, mask, n):
    """Mediane del fondo su una griglia n x n (x, y, valore) ignorando le sorgenti."""
    ny, nx = a.shape
    xs, ys, vs = [], [], []
    for i in range(n):
        for j in range(n):
            y0, y1 = int(i * ny / n), int((i + 1) * ny / n)
            x0, x1 = int(j * nx / n), int((j + 1) * nx / n)
            blocco = a[y0:y1, x0:x1]
            buoni = np.isfinite(blocco) & ~mask[y0:y1, x0:x1]
            if buoni.sum() < max(20, 0.25 * blocco.size):
                continue
            v = blocco[buoni]
            med = np.median(v)
            sd = 1.4826 * np.median(np.abs(v - med))
            if sd > 0:
                v = v[np.abs(v - med) < 3 * sd]
                med = np.median(v) if v.size else med
            xs.append((x0 + x1 - 1) / 2)
            ys.append((y0 + y1 - 1) / 2)
            vs.append(med)
    return np.array(xs), np.array(ys), np.array(vs)


def campioni_punti(a, punti_arr, raggio):
    xs, ys, vs = [], [], []
    ny, nx = a.shape
    r = max(2, int(round(raggio)))
    for x, y in punti_arr:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < nx and 0 <= yi < ny):
            continue
        b = a[max(0, yi - r): yi + r + 1, max(0, xi - r): xi + r + 1]
        v = b[np.isfinite(b)]
        if v.size < 5:
            continue
        med = np.median(v)
        sd = 1.4826 * np.median(np.abs(v - med))
        if sd > 0:
            v = v[np.abs(v - med) < 2.5 * sd]
        xs.append(x)
        ys.append(y)
        vs.append(np.median(v) if v.size else med)
    return np.array(xs), np.array(ys), np.array(vs)


def _base_polinomio(x, y, grado):
    return np.stack([x ** i * y ** j for i in range(grado + 1) for j in range(grado + 1 - i)], axis=-1)


def modello_polinomio(shape, xs, ys, vs, grado):
    ny, nx = shape
    grado = int(min(grado, max(0, int(math.sqrt(max(1, len(vs)))) - 1)))
    sx, sy = max(nx - 1, 1), max(ny - 1, 1)
    u, v = 2 * xs / sx - 1, 2 * ys / sy - 1
    usa = np.ones(len(vs), bool)
    coef = None
    for _ in range(5):
        A = _base_polinomio(u[usa], v[usa], grado)
        coef, *_ = np.linalg.lstsq(A, vs[usa], rcond=None)
        res = vs - _base_polinomio(u, v, grado) @ coef
        mad = 1.4826 * np.median(np.abs(res[usa])) if usa.sum() > 3 else 0
        nuovo = np.abs(res) < 3 * mad if mad > 0 else usa
        if nuovo.sum() < max(3, A.shape[1]) or (nuovo == usa).all():
            break
        usa = nuovo
    out = np.empty(shape, dtype=DTYPE)
    gx = 2 * np.arange(nx) / sx - 1
    for y0 in range(0, ny, 256):
        y1 = min(ny, y0 + 256)
        gy = 2 * np.arange(y0, y1) / sy - 1
        U, V = np.meshgrid(gx, gy)
        out[y0:y1] = (_base_polinomio(U, V, grado) @ coef).astype(DTYPE)
    return out


def modello_griglia(a, mask, box, filtro=3):
    """Fondo da mediane su riquadri di lato 'box', filtrate e interpolate (bicubica)."""
    ny, nx = a.shape
    box = int(max(8, min(box, max(ny, nx))))
    gy, gx = max(1, int(round(ny / box))), max(1, int(round(nx / box)))
    griglia = np.full((gy, gx), np.nan)
    for i in range(gy):
        for j in range(gx):
            y0, y1 = int(i * ny / gy), int((i + 1) * ny / gy)
            x0, x1 = int(j * nx / gx), int((j + 1) * nx / gx)
            b = a[y0:y1, x0:x1]
            buoni = np.isfinite(b) & ~mask[y0:y1, x0:x1]
            if buoni.sum() < max(10, 0.1 * b.size):
                continue
            v = b[buoni]
            med = np.median(v)
            sd = 1.4826 * np.median(np.abs(v - med))
            if sd > 0:
                v = v[np.abs(v - med) < 3 * sd]
            griglia[i, j] = 2.5 * np.median(v) - 1.5 * np.mean(v) if v.size > 10 else med
    if not np.isfinite(griglia).any():
        return np.full(a.shape, np.nanmedian(a), dtype=DTYPE)
    # i riquadri vuoti prendono il valore del vicino più prossimo
    vuoti = ~np.isfinite(griglia)
    if vuoti.any():
        idx = ndi.distance_transform_edt(vuoti, return_distances=False, return_indices=True)
        griglia = griglia[tuple(idx)]
    if filtro > 1 and min(griglia.shape) >= 3:
        griglia = ndi.median_filter(griglia, size=int(filtro), mode="nearest")
    # interpolazione bicubica sulla griglia dei centri dei riquadri
    yy = (np.arange(ny) + 0.5) / ny * gy - 0.5
    xx = (np.arange(nx) + 0.5) / nx * gx - 0.5
    out = np.empty(a.shape, dtype=DTYPE)
    ordine = 3 if min(gy, gx) >= 4 else 1
    for y0 in range(0, ny, 512):
        y1 = min(ny, y0 + 512)
        Y, X = np.meshgrid(yy[y0:y1], xx, indexing="ij")
        out[y0:y1] = ndi.map_coordinates(griglia, [Y, X], order=ordine, mode="nearest").astype(DTYPE)
    return out


class Fondo(Operazione):
    id = "background"
    famiglia = "science"
    parametri = [
        P("method", "choice", "poly", scelte=["poly", "mesh"]),
        P("degree", "int", 2, 1, 6, 1, visibile_se=("method", ["poly"])),
        P("samples", "int", 12, 4, 40, 1, visibile_se=("method", ["poly"])),
        P("box", "int", 256, 16, 4096, 16, px=True, visibile_se=("method", ["mesh"])),
        P("smooth", "int", 3, 1, 7, 2, visibile_se=("method", ["mesh"])),
        P("mask_k", "float", 3.0, 1.0, 10.0, 0.5),
        P("points", "points", []),
        P("radius", "int", 12, 3, 200, 1, px=True),
        P("only_points", "bool", False),
        P("output", "choice", "subtract", scelte=["subtract", "divide"]),
        P("pedestal", "bool", False),
    ]

    def descrivi(self, par):
        if par["method"] == "poly":
            t = f"background: polynomial degree {par['degree']}, {par['samples']}x{par['samples']} samples"
        else:
            t = f"background: mesh {par['box']} px, filter {par['smooth']}"
        if par["points"]:
            t += f", {len(par['points'])} manual points" + (" only" if par["only_points"] else "")
        return t + f", {par['output']}" + (" + pedestal" if par["pedestal"] else "")

    def modello(self, a, par, ctx):
        m, s = statistiche(a)
        mask = maschera_sorgenti(a, m, s, float(par["mask_k"]))
        punti = par.get("points") or []
        pa = []
        for x, y in punti:
            ax, ay = ctx.geom.arr_da_orig(x, y)
            pa.append((float(ax), float(ay)))
        raggio = ctx.geom.px(par["radius"])
        if par["method"] == "poly" or (par["only_points"] and pa):
            if par["only_points"] and pa:
                xs, ys, vs = campioni_punti(a, pa, raggio)
            else:
                xs, ys, vs = campioni_griglia(a, mask, int(par["samples"]))
                if pa:
                    x2, y2, v2 = campioni_punti(a, pa, raggio)
                    xs, ys, vs = np.r_[xs, x2], np.r_[ys, y2], np.r_[vs, v2]
            if len(vs) < 3:
                return np.full(a.shape, m, dtype=DTYPE)
            return modello_polinomio(a.shape, xs, ys, vs, int(par["degree"]))
        return modello_griglia(a, mask, ctx.geom.px(par["box"]), int(par["smooth"]))

    def applica(self, a, par, ctx):
        mod = self.modello(a, par, ctx)
        ctx.modello = mod
        livello = float(np.nanmedian(mod)) if np.isfinite(mod).any() else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            if par["output"] == "divide":
                sicuro = np.where(np.abs(mod) > 1e-12, mod, np.nan)
                out = a / sicuro * (livello if livello != 0 else 1.0)
                ctx.meta["unit_changed"] = False
            else:
                out = a - mod
                if par["pedestal"]:
                    out = out + livello
        return out.astype(DTYPE)


# =============================================================================
# RIDUZIONE DEL RUMORE
# =============================================================================
def a_trous(a, livelli):
    """Trasformata wavelet «à trous» con spline B3 (come MultiscaleLinearTransform)."""
    k1 = np.array([1, 4, 6, 4, 1], float) / 16
    c = a.astype(np.float64)
    piani = []
    for j in range(livelli):
        k = np.zeros(4 * 2 ** j + 1)
        k[:: 2 ** j] = k1
        liscia = ndi.convolve1d(ndi.convolve1d(c, k, axis=0, mode="reflect"), k, axis=1, mode="reflect")
        piani.append(c - liscia)
        c = liscia
    return piani, c


# rumore relativo di ogni piano wavelet per rumore bianco di sigma 1 (Starck & Murtagh)
_SIGMA_PIANI = [0.889, 0.200, 0.086, 0.041, 0.020, 0.010, 0.005, 0.0025]


def denoise_multiscala(a, sigma, forza, livelli=4):
    """Soglia morbida sui piani wavelet (rumore di ogni piano da Starck & Murtagh):
    forza 1 = toglie circa 1,5 sigma per piano, più sui piani fini."""
    piani, resto = a_trous(a, livelli)
    out = resto
    for j, w in enumerate(piani):
        t = forza * 1.5 * sigma * _SIGMA_PIANI[min(j, 7)] * max(0.4, 1.0 - 0.2 * j)
        out = out + np.sign(w) * np.maximum(np.abs(w) - t, 0)
    return out


class RiduzioneRumore(Operazione):
    id = "denoise"
    famiglia = "science"
    parametri = [
        P("algorithm", "choice", "multiscale", scelte=["multiscale", "gaussian", "median", "wavelet", "tv", "nlmeans"]),
        P("strength", "float", 0.7, 0.1, 5.0, 0.05),
        P("layers", "int", 4, 2, 7, 1, visibile_se=("algorithm", ["multiscale"])),
        P("size", "float", 1.5, 0.3, 10.0, 0.1, px=True, visibile_se=("algorithm", ["gaussian", "median"])),
        P("protect", "float", 50.0, 0.0, 100.0, 5.0),
        P("amount", "float", 100.0, 0.0, 100.0, 5.0),
    ]

    def descrivi(self, par):
        return (f"noise reduction: {par['algorithm']} strength {par['strength']:g}, "
                f"signal protection {par['protect']:g}%, amount {par['amount']:g}%")

    def applica(self, a, par, ctx):
        m, _ = statistiche(a)
        s = rumore_pixel(a)
        nan = ~np.isfinite(a)
        pieno = _riempi_nan(a, m).astype(np.float64)
        algo = par["algorithm"]
        forza = float(par["strength"])
        if algo == "multiscale":
            den = denoise_multiscala(pieno, s, forza, int(par["layers"]))
        elif algo == "gaussian":
            den = ndi.gaussian_filter(pieno, max(0.1, ctx.geom.px(par["size"])) * forza ** 0.5)
        elif algo == "median":
            n = max(3, int(round(ctx.geom.px(par["size"]) * 2)) | 1)
            den = ndi.median_filter(pieno, size=n)
        else:
            x = (pieno - m) / s
            try:
                if algo == "wavelet":
                    from skimage.restoration import denoise_wavelet
                    d = denoise_wavelet(x, sigma=forza, wavelet="sym4", mode="soft", method="BayesShrink",
                                        rescale_sigma=True)
                elif algo == "tv":
                    from skimage.restoration import denoise_tv_chambolle
                    d = denoise_tv_chambolle(x, weight=0.25 * forza)
                else:
                    from skimage.restoration import denoise_nl_means
                    d = denoise_nl_means(x, h=0.8 * forza, sigma=1.0, patch_size=5, patch_distance=6,
                                         fast_mode=True, preserve_range=True)
            except ImportError:
                d = denoise_multiscala(x, 1.0, forza)
            den = d * s + m
        prot = float(par["protect"]) / 100.0
        quanto = float(par["amount"]) / 100.0
        if prot > 0:
            liscia = ndi.gaussian_filter(pieno, 2.0)
            segnale = np.clip((liscia - m) / (8 * s), 0, 1)
            peso = quanto * (1 - prot * segnale)
        else:
            peso = quanto
        out = pieno + peso * (den - pieno)
        out = out.astype(DTYPE)
        out[nan] = np.nan
        ctx.avvisi.add("denoised")
        return out


# =============================================================================
# DECONVOLUZIONE (Richardson-Lucy con PSF misurata sulle stelle)
# =============================================================================
def psf_gaussiana(fwhm, raggio):
    s = max(0.3, fwhm / 2.3548)
    y, x = np.mgrid[-raggio:raggio + 1, -raggio:raggio + 1]
    k = np.exp(-(x * x + y * y) / (2 * s * s))
    return k / k.sum()


def psf_moffat(fwhm, raggio, beta=3.0):
    alfa = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    y, x = np.mgrid[-raggio:raggio + 1, -raggio:raggio + 1]
    k = (1 + (x * x + y * y) / (alfa * alfa)) ** (-beta)
    return k / k.sum()


def psf_dalle_stelle(a, m, sigma, raggio, max_stelle=60):
    """PSF media delle stelle isolate e non sature: ritagli ricentrati al
    sottopixel, normalizzati e combinati con la mediana. None se < 5 stelle."""
    b = np.where(np.isfinite(a), a - m, 0).astype(np.float32)
    liscia = ndi.gaussian_filter(b, 1.0)
    picchi = (liscia == ndi.maximum_filter(liscia, size=2 * raggio + 1)) & (liscia > 20 * sigma)
    ys, xs = np.nonzero(picchi)
    ny, nx = a.shape
    r2 = raggio + 2
    tieni = (ys > r2) & (ys < ny - r2 - 1) & (xs > r2) & (xs < nx - r2 - 1)
    ys, xs = ys[tieni], xs[tieni]
    if ys.size < 5:
        return None, 0
    vmax = np.nanmax(a)
    ordine = np.argsort(-liscia[ys, xs])
    ys, xs = ys[ordine], xs[ordine]
    salta = min(len(ys) // 20, 10)                 # le più brillanti: spesso sature
    ritagli = []
    dy, dx = np.mgrid[-r2:r2 + 1, -r2:r2 + 1]
    for y, x in zip(ys[salta:], xs[salta:]):
        fin = a[y - r2:y + r2 + 1, x - r2:x + r2 + 1].astype(np.float64)
        if not np.isfinite(fin).all() or fin.max() >= 0.95 * vmax:
            continue
        bordo = np.concatenate([fin[0], fin[-1], fin[:, 0], fin[:, -1]])
        f = fin - np.median(bordo)
        if f[r2, r2] <= 0:
            continue
        # una sola sorgente: nessun altro picco forte nella finestra
        if (f > 0.3 * f[r2, r2]).sum() > 0.25 * f.size or (f[[0, -1], :] > 0.2 * f[r2, r2]).any() \
                or (f[:, [0, -1]] > 0.2 * f[r2, r2]).any():
            continue
        pos = np.clip(f, 0, None)
        t = pos.sum()
        cx, cy = (pos * dx).sum() / t, (pos * dy).sum() / t
        if abs(cx) > 1.5 or abs(cy) > 1.5:
            continue
        f = ndi.shift(f, (-cy, -cx), order=3, mode="nearest")[2:-2, 2:-2]
        tot = f.sum()
        if tot <= 0:
            continue
        ritagli.append(f / tot)
        if len(ritagli) >= max_stelle:
            break
    if len(ritagli) < 5:
        return None, len(ritagli)
    k = np.median(np.array(ritagli), axis=0)
    # toglie il fondo residuo e le ali di rumore, rende simmetrica
    k = k - np.median(np.concatenate([k[0], k[-1], k[:, 0], k[:, -1]]))
    k = np.clip(k, 0, None)
    k = (k + k[::-1, ::-1]) / 2
    yy, xx = np.mgrid[-raggio:raggio + 1, -raggio:raggio + 1]
    k[np.hypot(xx, yy) > raggio + 0.5] = 0
    if k.sum() <= 0:
        return None, len(ritagli)
    return k / k.sum(), len(ritagli)


def richardson_lucy(d, psf, iterazioni, sigma, smorzamento=1.0, progresso=None):
    """Richardson-Lucy smorzato: dove il modello è già entro smorzamento·sigma
    dai dati il rapporto è 1 (il rumore non viene «affilato»)."""
    from scipy.signal import fftconvolve
    psf_m = psf[::-1, ::-1]
    u = d.copy()
    eps = 1e-12
    for i in range(iterazioni):
        conv = fftconvolve(u, psf, mode="same")
        rapporto = d / np.maximum(conv, eps)
        if smorzamento > 0:
            rapporto = np.where(np.abs(d - conv) < smorzamento * sigma, 1.0, rapporto)
        u = u * fftconvolve(rapporto, psf_m, mode="same")
        u = np.maximum(u, eps)
        if progresso:
            progresso((i + 1) / iterazioni)
    return u


class Deconvoluzione(Operazione):
    id = "deconv"
    famiglia = "science"
    parametri = [
        P("psf", "choice", "stars", scelte=["stars", "gaussian", "moffat"]),
        P("fwhm", "float", 3.0, 0.8, 20.0, 0.1, px=True, visibile_se=("psf", ["gaussian", "moffat"])),
        P("iterations", "int", 15, 1, 100, 1),
        P("damping", "float", 2.0, 0.0, 10.0, 0.5),
        P("deringing", "bool", True),
        P("only_signal", "bool", True),
        P("amount", "float", 100.0, 0.0, 100.0, 5.0),
    ]

    def descrivi(self, par):
        psf = par["psf"] if par["psf"] == "stars" else f"{par['psf']} FWHM {par['fwhm']:g} px"
        return (f"deconvolution: Richardson-Lucy {par['iterations']} iterations, PSF {psf}, "
                f"damping {par['damping']:g} sigma, deringing {'on' if par['deringing'] else 'off'}, "
                f"amount {par['amount']:g}%")

    def applica(self, a, par, ctx):
        m, _ = statistiche(a)
        s = max(rumore_pixel(a), 1e-12)
        nan = ~np.isfinite(a)
        pieno = _riempi_nan(a, m).astype(np.float64)
        if par["psf"] == "stars":
            fwhm0 = max(1.0, ctx.geom.px(3.0))
            raggio = 12
            psf, n = psf_dalle_stelle(pieno, m, s, raggio)
            if psf is None:
                psf = psf_gaussiana(max(1.0, fwhm0), 6)
                ctx.note.append(f"deconvolution: only {n} usable stars, gaussian PSF FWHM {fwhm0:.1f} px used")
            else:
                ctx.note.append(f"deconvolution: PSF from {n} stars")
        else:
            fwhm = max(0.5, ctx.geom.px(par["fwhm"]))
            raggio = int(min(30, max(4, np.ceil(2.5 * fwhm))))
            psf = (psf_gaussiana if par["psf"] == "gaussian" else psf_moffat)(fwhm, raggio)
        # Richardson-Lucy vuole dati positivi: fondo spostato a +10 sigma
        c = max(10 * s, m - np.percentile(pieno, 0.05) + 3 * s)
        d = np.maximum(pieno - m + c, 1e-6)
        bordo = psf.shape[0]                      # bordi riflessi: niente schiarimento ai lati
        dp = np.pad(d, bordo, mode="reflect")
        u = richardson_lucy(dp, psf, int(par["iterations"]), s, float(par["damping"]))[bordo:-bordo, bordo:-bordo]
        out = u - c + m                           # stesso fondo e stessa scala di prima
        quanto = float(par["amount"]) / 100.0
        if par["only_signal"]:
            # maschera allargata quanto la PSF: ogni stella riceve lo stesso peso dal nucleo alle ali
            lato = 2 * (psf.shape[0] // 2) + 1
            liscia = ndi.gaussian_filter(pieno, 2.0)
            segnale = np.clip((liscia - m) / (5 * s), 0, 1)
            peso = quanto * ndi.gaussian_filter(ndi.maximum_filter(segnale, size=lato), lato / 4.0)
        else:
            peso = quanto
        out = pieno + peso * (out - pieno)
        # flusso conservato su scala locale: la deconvoluzione sposta luce solo entro pochi
        # raggi della PSF, quindi la differenza sfocata su scala più grande va tolta
        lato = 2 * (psf.shape[0] // 2) + 1
        out = out - ndi.gaussian_filter(out - pieno, 1.5 * lato)
        if par["deringing"]:
            # niente aloni scuri: non sotto il minimo locale dell'originale (meno 1 sigma)
            lato = 2 * (psf.shape[0] // 2) + 1
            basso = ndi.minimum_filter(ndi.gaussian_filter(pieno, 1.0), size=lato) - s
            alzato = np.maximum(out, basso)
            # il flusso aggiunto per togliere l'alone viene tolto nei dintorni (flusso conservato)
            out = alzato - ndi.gaussian_filter(alzato - out, lato / 2.0)
        out = out.astype(DTYPE)
        out[nan] = np.nan
        ctx.avvisi.add("deconvolved")
        return out


# =============================================================================
# STIRAMENTO (da qui in poi i dati sono per la visione)
# =============================================================================
def mtf(m, x):
    """Midtones transfer function di PixInsight: mtf(m, m) = 0.5."""
    x = np.clip(x, 0, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, (m - 1) * x / ((2 * m - 1) * x - m)))


def normalizza(a):
    """(lo, hi) per portare i dati in [0, 1]: dal minimo robusto al massimo robusto."""
    v = campione(a)
    if v.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(v, 0.01))
    hi = float(np.percentile(v, 99.995))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def parametri_stf(a01, ombre=-2.8, bersaglio=0.25):
    """Punto del nero e mezzitoni dello «Screen Transfer Function» automatico."""
    v = campione(a01)
    if v.size == 0:
        return 0.0, 0.5
    med = float(np.median(v))
    mad = float(1.4826 * np.median(np.abs(v - med)))
    nero = min(max(0.0, med + ombre * mad), max(0.0, med - 1e-6)) if mad > 0 else 0.0
    x = (med - nero) / max(1e-9, 1 - nero)
    # mezzitono m tale che mtf(m, x) = bersaglio
    if x <= 0 or x >= 1:
        return nero, 0.5
    mezzo = (bersaglio - 1) * x / ((2 * bersaglio - 1) * x - bersaglio)
    return nero, float(np.clip(mezzo, 1e-4, 0.9999))


def stira_auto(a):
    """Visione rapida dei dati lineari (STF automatico), per le anteprime."""
    lo, hi = normalizza(a)
    x = np.clip((a - lo) / (hi - lo), 0, 1)
    nero, mezzo = parametri_stf(x)
    return mtf(mezzo, np.clip((x - nero) / max(1e-9, 1 - nero), 0, 1)).astype(DTYPE)


class Stiramento(Operazione):
    id = "stretch"
    famiglia = "display"
    parametri = [
        P("method", "choice", "auto", scelte=["auto", "ht", "asinh", "log", "sqrt", "histeq"]),
        P("target", "float", 0.25, 0.05, 0.6, 0.01, visibile_se=("method", ["auto"])),
        P("shadows", "float", -2.8, -6.0, 0.0, 0.1, visibile_se=("method", ["auto"])),
        P("black", "float", 0.0, 0.0, 0.5, 0.0005, visibile_se=("method", ["ht", "asinh", "log", "sqrt"])),
        P("mid", "float", 0.5, 0.0005, 0.9995, 0.0005, visibile_se=("method", ["ht"])),
        P("white", "float", 1.0, 0.01, 1.0, 0.0005, visibile_se=("method", ["ht", "asinh", "log", "sqrt"])),
        P("strength", "float", 50.0, 1.0, 5000.0, 1.0, visibile_se=("method", ["asinh", "log"])),
    ]

    def descrivi(self, par):
        mt = par["method"]
        if mt == "auto":
            return f"stretch: automatic STF (target {par['target']:g}, shadows {par['shadows']:g} sigma)"
        if mt == "ht":
            return f"stretch: histogram black {par['black']:.4f} mid {par['mid']:.4f} white {par['white']:.4f}"
        if mt in ("asinh", "log"):
            return f"stretch: {mt} strength {par['strength']:g}, black {par['black']:.4f}, white {par['white']:.4f}"
        return f"stretch: {mt}"

    def applica(self, a, par, ctx):
        if ctx.lineare:
            lo, hi = normalizza(a)
            ctx.norm = (lo, hi)
            x = np.clip((a - lo) / (hi - lo), 0, 1)
        else:
            x = np.clip(a, 0, 1)
        nan = ~np.isfinite(x)
        mt = par["method"]
        if mt == "auto":
            nero, mezzo = parametri_stf(x, float(par["shadows"]), float(par["target"]))
            y = mtf(mezzo, np.clip((x - nero) / max(1e-9, 1 - nero), 0, 1))
        else:
            nero, bianco = float(par["black"]), float(par["white"])
            if bianco <= nero:
                bianco = nero + 1e-4
            x = np.clip((x - nero) / (bianco - nero), 0, 1)
            if mt == "ht":
                y = mtf(float(par["mid"]), x)
            elif mt == "asinh":
                k = float(par["strength"])
                y = np.arcsinh(k * x) / np.arcsinh(k)
            elif mt == "log":
                k = float(par["strength"])
                y = np.log1p(k * x) / np.log1p(k)
            elif mt == "sqrt":
                y = np.sqrt(x)
            else:
                v = x[np.isfinite(x)]
                if v.size:
                    vs = np.sort(v[:: max(1, v.size // 200_000)])
                    y = np.interp(x, vs, np.linspace(0, 1, vs.size))
                else:
                    y = x
        y = np.asarray(y, dtype=DTYPE)
        y[nan] = np.nan
        ctx.lineare = False
        return y

    @staticmethod
    def auto_valori(a, lineare=True):
        """Valori di nero/mezzitoni/bianco equivalenti allo STF (per il metodo manuale)."""
        if lineare:
            lo, hi = normalizza(a)
            x = np.clip((a - lo) / (hi - lo), 0, 1)
        else:
            x = np.clip(a, 0, 1)
        nero, mezzo = parametri_stf(x)
        return dict(black=round(nero, 5), mid=round(mezzo, 5), white=1.0)


def istogramma(a, lineare=True, bins=256):
    """Istogramma dei dati portati in [0, 1] (per il pannello dello stiramento)."""
    if lineare:
        lo, hi = normalizza(a)
        x = (campione(a) - lo) / (hi - lo)
    else:
        x = campione(a)
    h, _ = np.histogram(np.clip(x, 0, 1), bins=bins, range=(0, 1))
    return h


# =============================================================================
# CURVE
# =============================================================================
PRESET_CURVE = {
    "linear": [[0, 0], [1, 1]],
    "s_soft": [[0, 0], [0.25, 0.20], [0.75, 0.80], [1, 1]],
    "s_strong": [[0, 0], [0.25, 0.15], [0.75, 0.87], [1, 1]],
    "lift_shadows": [[0, 0], [0.2, 0.28], [1, 1]],
    "darken_bg": [[0, 0], [0.15, 0.08], [0.5, 0.5], [1, 1]],
}


def funzione_curva(punti):
    pts = sorted((float(x), float(y)) for x, y in punti)
    if pts[0][0] > 0:
        pts.insert(0, (0.0, 0.0))
    if pts[-1][0] < 1:
        pts.append((1.0, 1.0))
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    xs, idx = np.unique(xs, return_index=True)
    ys = ys[idx]
    if xs.size < 3:
        return lambda x: np.interp(x, xs, ys)
    from scipy.interpolate import PchipInterpolator
    f = PchipInterpolator(xs, ys)
    return lambda x: np.clip(f(np.clip(x, 0, 1)), 0, 1)


class Curve(Operazione):
    id = "curves"
    famiglia = "display"
    parametri = [
        P("preset", "choice", "custom", scelte=["custom", "linear", "s_soft", "s_strong", "lift_shadows", "darken_bg"]),
        P("curve", "curve", [[0, 0], [1, 1]]),
        P("contrast", "float", 0.0, -100.0, 100.0, 1.0),
        P("brightness", "float", 0.0, -100.0, 100.0, 1.0),
    ]

    def descrivi(self, par):
        return f"curves: {len(par['curve'])} points, contrast {par['contrast']:g}, brightness {par['brightness']:g}"

    def applica(self, a, par, ctx):
        x = a if not ctx.lineare else stira_auto(a)
        y = funzione_curva(par["curve"])(x)
        c = float(par["contrast"]) / 100.0
        if c:
            y = 0.5 + (y - 0.5) * (1 + 2 * c if c > 0 else 1 + c)
        b = float(par["brightness"]) / 100.0
        if b:
            y = y + b * 0.5
        y = np.clip(y, 0, 1).astype(DTYPE)
        y[~np.isfinite(a)] = np.nan
        ctx.lineare = False
        return y


# =============================================================================
# TAVOLOZZA (colore di un singolo file)
# =============================================================================
TAVOLOZZE = ["gray", "gold", "teal", "red", "green", "blue", "hue", "inferno", "magma", "plasma", "viridis",
             "cividis", "hot", "afmhot", "cubehelix", "twilight"]


def colora(y, nome, tinta=30.0, sat=80.0):
    """y in [0,1] -> RGB HxWx3 in [0,1]."""
    y = np.nan_to_num(np.clip(y, 0, 1), nan=0.0).astype(DTYPE)
    if nome == "gray":
        return np.repeat(y[..., None], 3, axis=2)
    if nome in ("gold", "teal", "red", "green", "blue", "hue"):
        import colorsys
        h = {"gold": 38, "teal": 185, "red": 0, "green": 120, "blue": 220}.get(nome, tinta) / 360.0
        s = 0.85 if nome != "hue" else sat / 100.0
        r, g, b = colorsys.hsv_to_rgb(h, s, 1.0)
        col = np.array([r, g, b], dtype=DTYPE)
        # i toni alti vanno verso il bianco (come nelle immagini a banda stretta)
        bianco = np.clip((y - 0.6) / 0.4, 0, 1)[..., None] ** 1.5
        return np.clip(y[..., None] * (col * (1 - bianco) + bianco), 0, 1)
    import matplotlib
    cmap = matplotlib.colormaps[nome]
    return cmap(y)[..., :3].astype(DTYPE)


class Tavolozza(Operazione):
    id = "palette"
    famiglia = "display"
    parametri = [
        P("palette", "choice", "gold", scelte=TAVOLOZZE),
        P("hue", "float", 30.0, 0.0, 360.0, 1.0, visibile_se=("palette", ["hue"])),
        P("saturation", "float", 80.0, 0.0, 100.0, 1.0, visibile_se=("palette", ["hue"])),
    ]

    def descrivi(self, par):
        return f"palette: {par['palette']}" + (f" hue {par['hue']:g}" if par["palette"] == "hue" else "")

    def applica(self, a, par, ctx):
        x = a if not ctx.lineare else stira_auto(a)
        ctx.lineare = False
        if a.ndim == 3:
            x = x.mean(axis=2)
        rgb = colora(x, par["palette"], float(par["hue"]), float(par["saturation"]))
        rgb[~np.isfinite(x)] = 0
        return rgb


# =============================================================================
# REGISTRO
# =============================================================================
OPERAZIONI = {c.id: c() for c in (Ritaglio, Binning, PixelDifettosi, Fondo, RiduzioneRumore, Deconvoluzione,
                                  Stiramento, Curve, Tavolozza)}
ORDINE_MENU = [("science", ["crop", "bin", "pixels", "background", "denoise", "deconv"]),
               ("display", ["stretch", "curves", "palette"])]


def applica_passo(a, passo, ctx):
    """Applica un passo {'op','on','params'} (i passi spenti non fanno nulla).
    Le operazioni sui pixel lavorano canale per canale se l'immagine è già a colori."""
    if not passo.get("on", True):
        return a
    op = OPERAZIONI[passo["op"]]
    par = op.predefiniti()
    par.update(passo.get("params", {}))
    if a.ndim == 3 and op.id not in ("palette",):
        if op.id in ("crop", "bin"):
            canali = []
            g0 = ctx.geom.copia()
            for c in range(a.shape[2]):
                ctx.geom = g0.copia()
                canali.append(op.applica(a[..., c], par, ctx))
            return np.stack(canali, axis=2)
        lin = ctx.lineare
        canali = []
        for c in range(a.shape[2]):
            ctx.lineare = lin
            canali.append(op.applica(a[..., c], par, ctx))
        return np.stack(canali, axis=2)
    return op.applica(a, par, ctx)
