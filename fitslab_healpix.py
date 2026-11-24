# -*- coding: utf-8 -*-
"""
FITS LAB - MAPPE HEALPix (Planck e altre mappe di tutto il cielo)
================================================================
Le mappe di Planck non sono immagini: sono tabelle con un valore per ogni
pixel HEALPix della sfera celeste (Górski et al. 2005). Questo modulo:
  - legge la tabella (NSIDE, ordinamento RING/NESTED, sistema di coordinate);
  - calcola l'indice HEALPix di qualsiasi direzione (ang2pix ring e nested,
    stesse formule della libreria C di HEALPix) senza pacchetti esterni;
  - proietta la mappa su una griglia con un WCS: tutto il cielo (Mollweide) o
    un ritaglio gnomonico (TAN) attorno a una posizione.
Il risultato è una normale immagine 2D con coordinate, che FITS Lab lavora
come le altre.

© 2026 Edoardo Battistel - FITS Lab
"""

import math

import numpy as np


# =============================================================================
# FORMULE HEALPix
# =============================================================================
def _spargi_bit(v):
    """Interlaccia i bit di v nelle posizioni pari (0, 2, 4, ...)."""
    v = v.astype(np.int64) & 0xFFFFFFFF
    v = (v | (v << 16)) & 0x0000FFFF0000FFFF
    v = (v | (v << 8)) & 0x00FF00FF00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F0F0F0F0F
    v = (v | (v << 2)) & 0x3333333333333333
    v = (v | (v << 1)) & 0x5555555555555555
    return v


def ang2pix(nside, theta, phi, nested=False):
    """Indice HEALPix per colatitudine theta e longitudine phi (radianti, array)."""
    nside = int(nside)
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.mod(np.asarray(phi, dtype=np.float64), 2 * np.pi)
    z = np.cos(theta)
    za = np.abs(z)
    tt = np.mod(phi * (2 / np.pi), 4.0)
    out = np.zeros(z.shape, dtype=np.int64)
    eq = za <= 2.0 / 3.0
    po = ~eq
    if not nested:
        npix = 12 * nside * nside
        ncap = 2 * nside * (nside - 1)
        # fascia equatoriale
        t1 = nside * (0.5 + tt[eq])
        t2 = nside * z[eq] * 0.75
        jp = np.floor(t1 - t2).astype(np.int64)
        jm = np.floor(t1 + t2).astype(np.int64)
        ir = nside + 1 + jp - jm
        kshift = 1 - (ir & 1)
        ip = (jp + jm - nside + kshift + 1) // 2
        ip = np.mod(ip, 4 * nside)
        out[eq] = ncap + (ir - 1) * 4 * nside + ip
        # calotte polari
        ttp = tt[po]
        tp = ttp - np.floor(ttp)
        tmp = nside * np.sqrt(3 * (1 - za[po]))
        jp = np.floor(tp * tmp).astype(np.int64)
        jm = np.floor((1 - tp) * tmp).astype(np.int64)
        ir = jp + jm + 1
        ip = np.floor(ttp * ir).astype(np.int64)
        ip = np.mod(ip, 4 * ir)
        nord = z[po] > 0
        out[po] = np.where(nord, 2 * ir * (ir - 1) + ip, npix - 2 * ir * (ir + 1) + ip)
        return out
    # ordinamento NESTED
    face = np.zeros(z.shape, dtype=np.int64)
    ix = np.zeros(z.shape, dtype=np.int64)
    iy = np.zeros(z.shape, dtype=np.int64)
    t1 = nside * (0.5 + tt[eq])
    t2 = nside * (z[eq] * 0.75)
    jp = np.floor(t1 - t2).astype(np.int64)
    jm = np.floor(t1 + t2).astype(np.int64)
    ifp = jp // nside
    ifm = jm // nside
    face[eq] = np.where(ifp == ifm, ifp | 4, np.where(ifp < ifm, ifp, ifm + 8))
    ix[eq] = jm & (nside - 1)
    iy[eq] = nside - (jp & (nside - 1)) - 1
    ttp = tt[po]
    ntt = np.minimum(np.floor(ttp).astype(np.int64), 3)
    tp = ttp - ntt
    tmp = nside * np.sqrt(3 * (1 - za[po]))
    jp = np.minimum(np.floor(tp * tmp).astype(np.int64), nside - 1)
    jm = np.minimum(np.floor((1 - tp) * tmp).astype(np.int64), nside - 1)
    nord = z[po] >= 0
    face[po] = np.where(nord, ntt, ntt + 8)
    ix[po] = np.where(nord, nside - jm - 1, jp)
    iy[po] = np.where(nord, nside - jp - 1, jm)
    return face * nside * nside + (_spargi_bit(ix) | (_spargi_bit(iy) << 1))


def pix2ang_ring(nside, pix):
    """Centro (theta, phi) dei pixel RING (usato dall'autotest)."""
    nside = int(nside)
    pix = np.asarray(pix, dtype=np.int64)
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    theta = np.zeros(pix.shape)
    phi = np.zeros(pix.shape)
    nord = pix < ncap
    sud = pix >= npix - ncap
    eq = ~nord & ~sud
    p = pix[nord]
    iring = (1 + np.floor(np.sqrt(1 + 2 * p))).astype(np.int64) // 2
    iphi = p + 1 - 2 * iring * (iring - 1)
    theta[nord] = np.arccos(1 - iring ** 2 / (3.0 * nside ** 2))
    phi[nord] = (iphi - 0.5) * np.pi / (2 * iring)
    p = pix[eq] - ncap
    iring = p // (4 * nside) + nside
    iphi = p % (4 * nside) + 1
    fodd = np.where(((iring + nside) & 1) == 1, 1.0, 0.5)
    theta[eq] = np.arccos((2 * nside - iring) * 2.0 / (3 * nside))
    phi[eq] = (iphi - fodd) * np.pi / (2 * nside)
    p = npix - pix[sud]
    iring = (1 + np.floor(np.sqrt(2 * p - 1))).astype(np.int64) // 2
    iphi = 4 * iring + 1 - (p - 2 * iring * (iring - 1))
    theta[sud] = np.arccos(-1 + iring ** 2 / (3.0 * nside ** 2))
    phi[sud] = (iphi - 0.5) * np.pi / (2 * iring)
    return theta, phi


# =============================================================================
# LETTURA DELLE MAPPE
# =============================================================================
def e_mappa_healpix(hdul):
    """Indice dell'estensione con una mappa HEALPix, oppure None."""
    for i, hdu in enumerate(hdul):
        h = hdu.header
        if str(h.get("XTENSION", "")).strip().upper() != "BINTABLE":
            continue
        if str(h.get("PIXTYPE", "")).strip().upper() == "HEALPIX" or "NSIDE" in h:
            return i
    return None


def colonne_mappa(hdu):
    """Colonne con valori (esclusi indici di pixel)."""
    nomi = []
    for c in hdu.columns:
        if c.name.upper() in ("PIXEL", "PIX", "INDEX"):
            continue
        nomi.append(c.name)
    return nomi


class MappaHEALPix:
    """Una colonna di una mappa HEALPix letta in memoria (float32)."""

    def __init__(self, percorso, colonna=None):
        from astropy.io import fits
        with fits.open(percorso, memmap=True) as hdul:
            i = e_mappa_healpix(hdul)
            if i is None:
                raise ValueError("no HEALPix map in this file")
            hdu = hdul[i]
            h = hdu.header
            self.header = h.copy()
            self.header_primario = hdul[0].header.copy()
            self.colonne = colonne_mappa(hdu)
            self.colonna = colonna if colonna in self.colonne else self.colonne[0]
            self.nside = int(h["NSIDE"])
            self.nested = str(h.get("ORDERING", "RING")).strip().upper().startswith("NEST")
            sistema = str(h.get("COORDSYS", "G")).strip().upper()[:1]
            self.sistema = {"C": "C", "Q": "C", "E": "E", "G": "G"}.get(sistema, "G")
            j = [c.name for c in hdu.columns].index(self.colonna)
            self.unita = str(h.get(f"TUNIT{j + 1}", "")).strip()
            parziale = str(h.get("INDXSCHM", "IMPLICIT")).strip().upper() == "EXPLICIT"
            dati = np.asarray(hdu.data[self.colonna], dtype=np.float32).ravel()
            npix = 12 * self.nside ** 2
            if parziale or dati.size != npix:
                idx_col = next((c.name for c in hdu.columns if c.name.upper() in ("PIXEL", "PIX", "INDEX")), None)
                piena = np.full(npix, np.nan, dtype=np.float32)
                if idx_col is not None:
                    piena[np.asarray(hdu.data[idx_col], dtype=np.int64).ravel()] = dati
                else:
                    piena[:min(npix, dati.size)] = dati[:npix]
                dati = piena
            dati[(dati < -1.6e30) | ~np.isfinite(dati)] = np.nan     # UNSEEN = -1.6375e30
            self.dati = dati
        self.telescopio = str(self.header_primario.get("TELESCOP", self.header.get("TELESCOP", ""))).strip() or "Planck"
        self.frequenza = str(self.header.get("FREQ", self.header_primario.get("FREQ", ""))).strip()

    def risoluzione_deg(self):
        return math.degrees(math.sqrt(4 * math.pi / (12 * self.nside ** 2)))

    def campiona(self, lon_deg, lat_deg):
        """Valori della mappa nelle direzioni date (gradi, nel sistema della mappa)."""
        ok = np.isfinite(lon_deg) & np.isfinite(lat_deg)
        out = np.full(np.shape(lon_deg), np.nan, dtype=np.float32)
        theta = np.radians(90.0 - np.asarray(lat_deg)[ok])
        phi = np.radians(np.asarray(lon_deg)[ok])
        out[ok] = self.dati[ang2pix(self.nside, theta, phi, self.nested)]
        return out


def _ctype(sistema, proiezione):
    if sistema == "G":
        return [f"GLON-{proiezione}", f"GLAT-{proiezione}"]
    if sistema == "E":
        return [f"ELON-{proiezione}", f"ELAT-{proiezione}"]
    return [f"RA---{proiezione}", f"DEC--{proiezione}"]


def wcs_cielo_intero(sistema, larghezza=2400):
    """WCS Mollweide di tutto il cielo (centro 0,0, longitudine verso sinistra)."""
    from astropy.wcs import WCS
    nx = int(larghezza)
    ny = nx // 2
    w = WCS(naxis=2)
    w.wcs.ctype = _ctype(sistema, "MOL")
    w.wcs.crval = [0.0, 0.0]
    # in Mollweide l'equatore è lungo 4*sqrt(2) radianti -> gradi "di proiezione"
    larg_deg = 4 * math.sqrt(2) * 180 / math.pi
    passo = larg_deg / (nx - 2)
    w.wcs.cdelt = [-passo, passo]
    w.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    return w, (ny, nx)


def wcs_ritaglio(sistema, lon0, lat0, lato_deg, pixel_arcmin):
    """WCS gnomonico (TAN) quadrato centrato su (lon0, lat0) nel sistema dato."""
    from astropy.wcs import WCS
    passo = pixel_arcmin / 60.0
    n = int(max(16, min(8000, round(lato_deg / passo))))
    w = WCS(naxis=2)
    w.wcs.ctype = _ctype(sistema, "TAN")
    w.wcs.crval = [float(lon0), float(lat0)]
    w.wcs.cdelt = [-passo, passo]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    return w, (n, n)


def mollweide_inversa(w, x, y):
    """Pixel (0-based) -> (lon, lat) in gradi per il WCS Mollweide di wcs_cielo_intero,
    calcolato a mano: fuori dall'ellisse NaN (wcslib lì darebbe un errore).
    Centro in (0, 0), formule di Calabretta & Greisen 2002."""
    R = 180.0 / math.pi
    X = (np.asarray(x, float) + 1 - w.wcs.crpix[0]) * w.wcs.cdelt[0]
    Y = (np.asarray(y, float) + 1 - w.wcs.crpix[1]) * w.wcs.cdelt[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        s = Y / (math.sqrt(2) * R)
        dentro = np.abs(s) <= 1
        t = np.arcsin(np.clip(s, -1, 1))
        z = (2 * t + np.sin(2 * t)) / math.pi
        lat = np.degrees(np.arcsin(np.clip(z, -1, 1)))
        lon = np.degrees(math.pi * X / (2 * math.sqrt(2) * R * np.cos(t)))
    dentro &= np.abs(lon) <= 180
    lon = np.where(dentro, np.mod(lon + float(w.wcs.crval[0]), 360.0), np.nan)
    lat = np.where(dentro, lat, np.nan)
    return lon, lat


def proietta(mappa, w, shape, sistema_wcs=None, righe=512, progresso=None):
    """Immagine 2D: per ogni pixel del WCS il valore della mappa HEALPix.
    Se il WCS è in un altro sistema (es. RA/Dec e mappa galattica) si converte."""
    ny, nx = shape
    out = np.full((ny, nx), np.nan, dtype=np.float32)
    sistema_wcs = sistema_wcs or mappa.sistema
    converti = sistema_wcs != mappa.sistema
    if converti:
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        nomi = {"C": "icrs", "G": "galactic", "E": "barycentrictrueecliptic"}
    xs = np.arange(nx)
    mollweide = str(w.wcs.ctype[0]).endswith("MOL")
    for y0 in range(0, ny, righe):
        y1 = min(ny, y0 + righe)
        yy, xx = np.meshgrid(np.arange(y0, y1), xs, indexing="ij")
        if mollweide:
            lon, lat = mollweide_inversa(w, xx.ravel(), yy.ravel())
        else:
            with np.errstate(invalid="ignore"):
                lon, lat = w.wcs_pix2world(xx.ravel(), yy.ravel(), 0)
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        if converti:
            ok = np.isfinite(lon) & np.isfinite(lat)
            if ok.any():
                c = SkyCoord(lon[ok] * u.deg, lat[ok] * u.deg, frame=nomi[sistema_wcs]).transform_to(nomi[mappa.sistema])
                sph = c.spherical
                lon[ok] = sph.lon.deg
                lat[ok] = sph.lat.deg
        out[y0:y1] = mappa.campiona(lon, lat).reshape(y1 - y0, nx)
        if progresso:
            progresso(y1 / ny)
    return out
