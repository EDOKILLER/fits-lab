# -*- coding: utf-8 -*-
"""
AUTOASTROPIPE (AAP) - MOTORE DI CALCOLO
=======================================
Tutte le operazioni di calcolo, senza interfaccia grafica:
  - ricerca, rinomina e copia dei FITS (nome dal filtro: f090.fits, f090a.fits ...)
  - lettura dell'estensione scientifica, filtro e lunghezza d'onda
  - ricostruzione delle zone di pixel rovinati
  - riduzione del rumore (6 algoritmi), sottrazione del fondo (4 stimatori),
    stretching (9 tipi) sempre con ZScaleInterval
  - elaborazione a piena risoluzione a tasselli (modalità RAM / disco / ibrida)
  - stack bianco e nero e RGB riproiettando con il WCS
  - analisi scientifiche e oggetti con nome nel campo (SIMBAD)
  - stato della pipeline (JSON) e file di log
  - file .asdf del Roman Space Telescope, convertiti in FITS (modulo roman_asdf)

© 2026 Edoardo Battistel - AutoAstroPipe versione 2.3
"""

import colorsys
import csv
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# --- autore e versione (compaiono nella finestra, nei FITS e nelle immagini) ---
NOME_PROGRAMMA = "AutoAstroPipe"
SIGLA = "AAP"
AUTORE = "Edoardo Battistel"
VERSIONE = "2.5"
ANNO = "2026"
COPYRIGHT = f"© {ANNO} {AUTORE}"
FIRMA = f"© {AUTORE}"
COPYRIGHT_ASCII = f"(C) {ANNO} {AUTORE}"     # le intestazioni FITS ammettono solo ASCII


def _ascii(testo):
    """Converte un testo in ASCII stampabile per le intestazioni FITS
    (è → e, × → x, µ → u, © → (C); gli altri caratteri vengono tolti)."""
    import unicodedata
    t = str(testo).replace("©", "(C)").replace("×", "x").replace("µ", "u").replace("μ", "u").replace("λ", "lambda")
    t = t.replace("—", "-").replace("–", "-").replace("→", "->").replace("°", " deg")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return "".join(c for c in t if 32 <= ord(c) < 127)


import roman_asdf

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.convolution import (Gaussian2DKernel, Box2DKernel, convolve_fft,
                                 interpolate_replace_nans)
from astropy.visualization import (ZScaleInterval, LinearStretch, AsinhStretch, LogStretch,
                                   SqrtStretch, PowerStretch, SquaredStretch, SinhStretch,
                                   HistEqStretch, PowerDistStretch)

try:
    import psutil
except ImportError:
    psutil = None

warnings.simplefilter("ignore", FITSFixedWarning)

DTYPE = np.float32
ESTENSIONI_FITS = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz", ".fits.fz", ".fit.fz", ".fz",
                   ".fits.bz2", ".fit.bz2")
CARTELLA_CACHE = "_aap_cache"
FILE_STATO = "aap_stato.json"
FILE_LOG = "aap_log.txt"
FILE_SCELTE_CSV = "aap_scelte.csv"
# nomi usati fino alla versione 1.4 (Pipeline FITS): il lavoro vecchio si riprende
FILE_STATO_VECCHIO = "pipeline_stato.json"
CARTELLE_CACHE = (CARTELLA_CACHE, "_pipeline_cache")
PREFISSI_OUTPUT = ("stack_", "analisi_")
TILE = 2048
MAX_LATO_RIDOTTA = 3000          # immagine ridotta usata per fondo e panoramica
# SIMBAD: server principale (CDS, Strasburgo) e mirror ufficiale (Harvard CfA)
URL_SIMBAD_TAP = ["https://simbad.cds.unistra.fr/simbad/sim-tap",
                  "https://simbad.cfa.harvard.edu/simbad/sim-tap"]


# =============================================================================
# ALGORITMI DISPONIBILI (etichette, parametri e valori predefiniti)
# =============================================================================
ALGORITMI_NR = {
    "gaussiano": dict(etichetta="Gaussiano", parametro="sigma in pixel", valori=[0.5, 1.0, 1.5, 2.0, 3.0],
                      minimo=0.1, intero=False,
                      descrizione="Sfocatura gaussiana (astropy): semplice e veloce. "
                                  "Il valore è il raggio in pixel: più alto = più liscio."),
    "mediano": dict(etichetta="Mediano", parametro="finestra in pixel", valori=[3, 5, 7],
                    minimo=3, intero=True,
                    descrizione="Filtro mediano (scipy): toglie pixel caldi e rumore a puntini, "
                                "conserva i bordi. Valore = lato della finestra."),
    "box": dict(etichetta="Media mobile (box)", parametro="lato in pixel", valori=[3, 5, 9],
                minimo=2, intero=True,
                descrizione="Media su un quadrato (astropy): veloce, ma sfuma anche i dettagli."),
    "wavelet": dict(etichetta="Wavelet (BayesShrink)", parametro="intensità × rumore", valori=[0.5, 1.0, 1.5, 2.0],
                    minimo=0.05, intero=False,
                    descrizione="Wavelet (scikit-image): riduce il rumore conservando i dettagli. "
                                "Ottimo per nebulose."),
    "tv": dict(etichetta="Variazione totale", parametro="peso × rumore", valori=[0.25, 0.5, 1.0, 2.0],
               minimo=0.01, intero=False,
               descrizione="Total variation (scikit-image): appiattisce il fondo, tiene i contorni netti."),
    "nlmeans": dict(etichetta="Non-local means", parametro="h × rumore", valori=[0.6, 0.8, 1.0, 1.2],
                    minimo=0.05, intero=False,
                    descrizione="Non-local means (scikit-image): la qualità migliore, ma LENTO "
                                "sulle immagini grandi."),
}

ALGORITMI_FONDO = {
    "mediana": dict(etichetta="Mediana", descrizione="Mediana sigma-clipped in ogni riquadro: robusta (consigliata)."),
    "sextractor": dict(etichetta="SExtractor", descrizione="Stimatore di SExtractor: 2.5·mediana − 1.5·media."),
    "mmm": dict(etichetta="MMM (DAOPHOT)", descrizione="Moda stimata come 3·mediana − 2·media, come DAOPHOT."),
    "media": dict(etichetta="Media", descrizione="Media sigma-clipped: semplice, più sensibile alle stelle."),
}
VALORI_FONDO = [64, 128, 256, 512, 800]

ALGORITMI_STRETCH = {
    "lineare": dict(etichetta="Lineare", parametro=None, valori=[None]),
    "asinh": dict(etichetta="Asinh", parametro="a", valori=[0.01, 0.1, 1.0, 100.0]),
    "log": dict(etichetta="Logaritmico", parametro="a", valori=[100.0, 1000.0, 10000.0]),
    "sqrt": dict(etichetta="Radice quadrata", parametro=None, valori=[None]),
    "power": dict(etichetta="Potenza", parametro="esponente", valori=[0.5, 2.0, 3.0]),
    "square": dict(etichetta="Quadrato", parametro=None, valori=[None]),
    "sinh": dict(etichetta="Sinh", parametro="a", valori=[0.3, 0.5, 1.0]),
    "powerdist": dict(etichetta="Power dist", parametro="a", valori=[100.0, 1000.0]),
    "histeq": dict(etichetta="Equalizzazione istogramma", parametro=None, valori=[None]),
}

METODI_SNR = {"segnale": "SNR medio dei pixel con segnale (rumore misurato da pixel a pixel)",
              "globale": "media / deviazione standard di tutta l'immagine"}

ANALISI = {
    "griglia": "Coordinate celesti con griglia",
    "snr": "Mappa del rapporto segnale/rumore",
    "gradiente": "Gradiente morfologico (fronti d'urto, Sobel)",
    "strutture": "Strutture fini (differenza di gaussiane)",
    "isofote": "Isofote (curve di livello)",
    "sorgenti": "Sorgenti puntiformi rilevate + catalogo CSV",
    "istogramma": "Istogramma delle intensità",
    "profilo": "Profilo radiale dal centro",
    "oggetti": "Oggetti con nome (NGC, IC, M...) + CSV",
    "cielo": "Posizione nel cielo: mappa della costellazione + CSV",
    "raggi_x": "Raggi X: conteggi, flusso e profilo (Chandra, XMM...) + CSV",
}


# =============================================================================
# MEMORIA (RAM / disco / ibrida)
# =============================================================================
def ram_libera():
    if psutil is not None:
        return psutil.virtual_memory().available
    try:
        with open("/proc/meminfo") as f:
            for r in f:
                if r.startswith("MemAvailable:"):
                    return int(r.split()[1]) * 1024
    except OSError:
        pass
    return None


def ram_processo_gb():
    return psutil.Process().memory_info().rss / 1024**3 if psutil else float("nan")


class Memoria:
    """Alloca gli array grandi in RAM o su disco (memmap .npy nella cache):
       'ram'    sempre in RAM (più veloce)
       'disco'  sempre su disco (RAM minima)
       'ibrido' in RAM finché il totale resta sotto il limite e c'è RAM libera"""

    def __init__(self, modo="ibrido", limite_gb=4.0, cartella=None):
        self.modo, self.limite = modo, limite_gb * 1024**3
        self.cartella = Path(cartella) if cartella else None
        self.in_ram = 0
        self._n = 0

    def alloca(self, nome, shape, dtype=DTYPE, riempi=None):
        nbyte = int(np.prod(shape)) * np.dtype(dtype).itemsize
        libera = ram_libera()
        usa_ram = (self.modo == "ram" or
                   (self.modo == "ibrido" and self.in_ram + nbyte <= self.limite
                    and (libera is None or nbyte < 0.6 * libera)))
        if usa_ram or self.cartella is None:
            a = np.empty(shape, dtype=dtype)
            self.in_ram += nbyte
        else:
            self.cartella.mkdir(parents=True, exist_ok=True)
            self._n += 1
            p = self.cartella / f"tmp_{os.getpid()}_{self._n}_{nome}.npy"
            a = np.lib.format.open_memmap(p, mode="w+", dtype=dtype, shape=shape)
        if riempi is not None:
            for y in range(0, shape[0], TILE):
                a[y:y + TILE] = riempi
        return a

    def libera(self, a):
        if a is None:
            return
        if not isinstance(a, np.memmap):
            self.in_ram = max(0, self.in_ram - a.nbytes)

    def pulisci_temporanei(self):
        gc.collect()
        if self.cartella and self.cartella.is_dir():
            for p in self.cartella.glob(f"tmp_{os.getpid()}_*.npy"):
                try:
                    p.unlink()
                except OSError:
                    pass


def restituisci_ram():
    gc.collect()
    if os.name == "posix":
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


# =============================================================================
# FILTRI E LUNGHEZZE D'ONDA
# =============================================================================
LUNGHEZZE_NOTE_UM = {
    "FUV": 0.153, "NUV": 0.231, "U": 0.365, "B": 0.445, "V": 0.551, "R": 0.658, "I": 0.806,
    "SDSS_U": 0.355, "SDSS_G": 0.477, "SDSS_R": 0.623, "SDSS_I": 0.763, "SDSS_Z": 0.913,
    "G": 0.477, "Z": 0.913, "Y": 1.02, "J": 1.25, "H": 1.65, "K": 2.2, "KS": 2.15,
    "VIS": 0.715, "NIR_Y": 1.08, "NIR_J": 1.37, "NIR_H": 1.77,
    "W1": 3.4, "W2": 4.6, "W3": 12.0, "W4": 22.0,
    "IRAC1": 3.6, "IRAC2": 4.5, "IRAC3": 5.8, "IRAC4": 8.0, "I1": 3.6, "I2": 4.5, "I3": 5.8, "I4": 8.0,
    "MIPS24": 24.0, "M1": 24.0, "PACS70": 70.0, "PACS100": 100.0, "PACS160": 160.0,
    "SPIRE250": 250.0, "SPIRE350": 350.0, "SPIRE500": 500.0,
    "MIPS70": 70.0, "MIPS160": 160.0,
    # Pan-STARRS1 (Tonry et al. 2012)
    "PS1_G": 0.481, "PS1_R": 0.617, "PS1_I": 0.752, "PS1_Z": 0.866, "PS1_Y": 0.962,
    # SPHEREx: centro della banda di ciascuno dei 6 rivelatori (spettro 0,75-5 µm)
    "SPHEREX_D1": 0.93, "SPHEREX_D2": 1.37, "SPHEREX_D3": 2.03, "SPHEREX_D4": 3.12,
    "SPHEREX_D5": 4.12, "SPHEREX_D6": 4.71,
    "HALPHA": 0.6563, "HA": 0.6563, "H_ALPHA": 0.6563, "OIII": 0.5007, "O_III": 0.5007,
    "SII": 0.6724, "S_II": 0.6724, "HBETA": 0.4861, "HB": 0.4861, "NII": 0.6584,
    # raggi X (Chandra): lunghezza d'onda equivalente all'energia media della banda
    "SOFT": 0.0014, "MEDIUM": 0.00083, "HARD": 0.00028, "BROAD": 0.00052, "ULTRASOFT": 0.0035,
}


TELESCOPI_X = ("CHANDRA", "XMM", "NEWTON", "ROSAT", "EROSITA", "SRG", "NUSTAR", "SUZAKU", "EINSTEIN")
STRUMENTI_X = ("ACIS", "HRC", "EPN", "EMOS1", "EMOS2", "EPIC", "XRT", "PSPC", "HRI", "FPMA", "FPMB", "XIS")


def e_conteggi_x(*headers):
    """Vero per le immagini a raggi X fatte di conteggi di fotoni (Chandra, XMM...):
    lì lo 0 è un dato valido (nessun fotone), non un pixel senza dati."""
    h = _norm_hdr(*headers)
    tel = str(h.get("TELESCOP", "")).upper()
    strum = str(h.get("INSTRUME", "")).upper()
    if any(t in tel for t in TELESCOPI_X) or (strum in STRUMENTI_X and "UVOT" not in strum):
        return True
    unita = str(h.get("BUNIT", "")).lower()
    try:
        intero = int(h.get("BITPIX", -32)) > 0
    except (TypeError, ValueError):
        intero = False
    return intero and ("count" in unita or "cts" in unita)


def impronta_conteggi(dati, max_lato=256):
    """Zona del rivelatore in un'immagine a conteggi: si riducono i conteggi a
    blocchi (somma), si uniscono i pixel con almeno un fotone e si riempiono i
    buchi. Fuori dall'impronta non ci sono dati. Restituisce (maschera, fattore)."""
    ny, nx = dati.shape
    f = max(1, math.ceil(max(ny, nx) / max_lato))
    nyb, nxb = math.ceil(ny / f), math.ceil(nx / f)
    somma = np.zeros((nyb, nxb), dtype=np.float64)
    for jb in range(nyb):
        s = np.nan_to_num(np.array(dati[jb * f:(jb + 1) * f, :], dtype=np.float64))
        if s.shape[1] < nxb * f:
            s = np.pad(s, ((0, 0), (0, nxb * f - s.shape[1])))
        somma[jb] = s.reshape(s.shape[0], nxb, f).sum(axis=(0, 2))
    m = somma > 0
    m = ndi.binary_closing(m, structure=np.ones((3, 3)), iterations=4, border_value=0)
    m = ndi.binary_fill_holes(m)
    lab, n = ndi.label(m)
    if n > 1:
        aree = ndi.sum(m, lab, range(1, n + 1))
        m = np.isin(lab, 1 + np.where(aree >= 0.02 * aree.max())[0])
    if not m.any():
        m[:] = True
    return m, f


def maschera_su_griglia(maschera, f_maschera, y0, y1, x0, x1, passo=1):
    """La maschera (ridotta di f_maschera) letta sui pixel [y0:y1:passo, x0:x1:passo]."""
    yy = np.clip((np.arange(y0, y1, passo) + passo / 2) // f_maschera, 0, maschera.shape[0] - 1).astype(int)
    xx = np.clip((np.arange(x0, x1, passo) + passo / 2) // f_maschera, 0, maschera.shape[1] - 1).astype(int)
    return maschera[np.ix_(yy, xx)]


def _norm_hdr(*headers):
    """Unisce più header in un dizionario (il primo ha la precedenza)."""
    d = {}
    for h in reversed([h for h in headers if h is not None]):
        for k in h.keys():
            if k and k not in ("COMMENT", "HISTORY", ""):
                try:
                    d[k.upper()] = h[k]
                except Exception:
                    pass
    return d


def rileva_filtro(hdr_p, hdr_s, nome_file=""):
    """Restituisce (filtro come scritto nell'header, nome base del file).
    Esempi: F090W -> 'f090', F1130W -> 'f1130', NIRCam F444W+F470N -> 'f470',
    VIS -> 'vis', Ks -> 'ks'."""
    h = _norm_hdr(hdr_p, hdr_s)
    tel = str(h.get("TELESCOP", "")).upper()
    cand = None
    pupil = str(h.get("PUPIL", "")).strip().upper()
    if re.match(r"^F\d{3,4}[A-Z]{1,2}\d?$", pupil):         # NIRCam: filtro stretto nella ruota pupilla
        cand = pupil
    if cand is None:
        cand = filtro_missione(h, nome_file)
    if cand is None:
        for k in ("FILTER", "FILTER1", "FILTER2", "FILTER3", "FILTNAM1", "FILTNAM2", "FILTNAME",
                  "FILT_ID", "FILTER_ID", "WAVEBAND", "BANDNAME", "BAND"):
            v = str(h.get(k, "")).strip()
            if not v or v.upper().startswith(("CLEAR", "OPEN", "N/A", "NONE", "BLANK", "UNKNOWN")):
                continue
            if k == "BAND" and v.isdigit():
                if "GALEX" in tel:
                    cand = {"1": "NUV", "2": "FUV"}.get(v)
                continue
            cand = v
            break
    nf = nome_file.lower()
    if cand is None:
        m = re.search(r"(f\d{3,4}[wmnxlp]{1,2}\d?)", nf)
        if m:
            cand = m.group(1).upper()
        elif "-nd-" in nf or "_nd_" in nf:
            cand = "NUV"
        elif "-fd-" in nf or "_fd_" in nf:
            cand = "FUV"
        elif "CHANDRA" in tel or "acis" in nf or "hrc" in nf:
            for banda in ("ultrasoft", "soft", "medium", "hard", "broad"):
                if banda in nf:
                    cand = banda.upper()
                    break
            else:
                cand = "BROAD"
        else:
            cand = "sconosciuto"
    m = re.match(r"^F(\d{3,4})", cand.upper())
    if m:
        base = "f" + m.group(1)
    else:
        base = re.sub(r"[^a-z0-9]+", "_", cand.lower()).strip("_") or "filtro"
    if "CHANDRA" in tel and str(h.get("CONTENT", "")).upper().startswith("LORES"):
        base += "_intero"               # tutto il campo a bassa risoluzione (doppione ridotto)
    det = str(h.get("DETECTOR", "")).strip().lower()
    if "ROMAN" in tel and det and str(h.get("ROMANLIV", "L2")).upper() != "L3":
        base += "_" + re.sub(r"[^a-z0-9]+", "", det)   # Roman: 18 rivelatori per esposizione (f158_wfi07)
    return cand, base


def _intero(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def filtro_missione(h, nome_file=""):
    """Filtro per le missioni che non usano la parola chiave FILTER:
    Pan-STARRS (FPA.FILTER), WISE/NEOWISE/unWISE (BAND 1-4), Spitzer IRAC e MIPS
    (CHNLNUM), Herschel PACS e SPIRE, SPHEREx (rivelatori D1-D6). None se non riconosciuto."""
    tel = " ".join(str(h.get(k, "")) for k in ("TELESCOP", "FPA.TELESCOPE", "MISSION", "OBSERVAT")).upper()
    strum = " ".join(str(h.get(k, "")) for k in ("INSTRUME", "FPA.INSTRUMENT", "DETNAM")).upper()
    nf = Path(nome_file).name.lower()
    # Pan-STARRS1: «HIERARCH FPA.FILTER = 'r.00000'»; nomi rings.v3.skycell.xxxx.stk.r.unconv.fits
    f_ps1 = str(h.get("FPA.FILTER", h.get("FPA.FILTERID", ""))).strip()
    m = re.search(r"\.(?:stk|wrp|warp|exp)\.([grizy])\.", nf)
    if f_ps1[:1].lower() in "grizyw" and f_ps1 and ("PS1" in tel or "GPC" in strum or "FPA.FILTER" in h):
        return f"PS1 {f_ps1[0].lower()}"
    if m and ("rings" in nf or "skycell" in nf or "ps1" in nf or "PS1" in tel):
        return f"PS1 {m.group(1)}"
    # SPHEREx: rivelatori 1-6 (ognuno copre una parte dello spettro)
    if "SPHEREX" in tel or "spherex" in nf or "_spx_" in nf:
        det = _intero(h.get("DETECTOR"))
        if det is None:
            m = re.search(r"d([1-6])(?:_spx|[_.-])", nf)
            det = int(m.group(1)) if m else None
        if det in range(1, 7):
            return f"SPHEREx D{det}"
    # WISE / NEOWISE / unWISE: BAND = 1-4, nomi ...-w1-int-3.fits, unwise-...-w2-img-m.fits
    m = re.search(r"(?:^|[-_.])w([1-4])(?:[-_.]|$)", nf)
    if "WISE" in tel or "wise" in nf or (h.get("MAGZP") is not None and _intero(h.get("BAND")) in (1, 2, 3, 4)):
        banda = _intero(h.get("BAND"))
        if banda in (1, 2, 3, 4):
            return f"W{banda}"
        if m:
            return f"W{m.group(1)}"
    # Spitzer IRAC (canali 1-4: 3,6 4,5 5,8 8 µm) e MIPS (24, 70, 160 µm)
    if ("SPITZER" in tel or "IRAC" in strum or "MIPS" in strum or "spitzer" in nf or "glimpse" in nf
            or "smog" in nf or "seip" in nf):
        canale = _intero(h.get("CHNLNUM"))
        if "MIPS" in strum or "mips" in nf:
            if canale in (1, 2, 3):
                return f"MIPS{(24, 70, 160)[canale - 1]}"
            m2 = re.search(r"mips[_-]?(24|70|160)|_m([123])_", nf)
            if m2:
                return f"MIPS{m2.group(1) or (24, 70, 160)[int(m2.group(2)) - 1]}"
            return "MIPS24"
        if canale in (1, 2, 3, 4):
            return f"IRAC{canale}"
        m2 = re.search(r"irac[_-]?([1-4])|_i([1-4])[_.]|ch([1-4])|mosaic_i([1-4])", nf)
        if m2:
            return f"IRAC{next(g for g in m2.groups() if g)}"
    # Herschel PACS (70/100/160 µm, blu/verde/rosso) e SPIRE (250/350/500 µm)
    herschel_java = "HCSS____" in h or str(h.get("CLASS___", "")).lower().startswith("herschel")
    if ("HERSCHEL" in tel or "PACS" in strum or "SPIRE" in strum or herschel_java
            or nf.startswith(("hpacs", "hspire", "hppjs", "hspir"))):
        lam = None
        for k in ("WAVELNTH", "WAVELEN", "WAVELENG", "LAMBDA"):
            try:
                lam = float(h.get(k))
                break
            except (TypeError, ValueError):
                pass
        if "SPIRE" in strum or "spire" in nf:
            for sigla, um in (("psw", 250), ("pmw", 350), ("plw", 500)):
                if sigla in nf or str(h.get("DETECTOR", "")).lower().startswith(sigla):
                    return f"SPIRE{um}"
            if lam:
                return f"SPIRE{min((250, 350, 500), key=lambda x: abs(x - lam))}"
            return None
        filtro = str(h.get("FILTER", h.get("BAND", ""))).strip().lower()
        colori = {"blue": 70, "green": 100, "red": 160}
        m_col = re.search(r"(?:^|[-_.])(blue|green|red)(?:[-_.]|$)", nf)
        if filtro not in colori and m_col:
            filtro = m_col.group(1)            # es. Hi-GAL mw2-blue.fits / mw2-red.fits
        if filtro in colori:
            return f"PACS{colori[filtro]}"
        if lam:
            return f"PACS{min((70, 100, 160), key=lambda x: abs(x - lam))}"
        m2 = re.search(r"(?:map|_)(b|g|r)(?:_|\.|$)|hppjsmap([bgr])|pacs[_-]?(70|100|160)", nf)
        if m2:
            g = next(x for x in m2.groups() if x)
            return f"PACS{ {'b': 70, 'g': 100, 'r': 160}.get(g, g) }".replace(" ", "")
    return None


def _lunghezza_da_intestazione(hdr_p, hdr_s):
    """Lunghezza d'onda dalle parole chiave generiche (WAVELEN, WAVELNTH, ...), in µm.
    L'unità si capisce dal commento (Angstrom, nm, micron); senza commento:
    >= 1000 = Ångström (es. SDO 171), sotto = micron (WISE 3,4, Herschel 250)."""
    for hdr in (hdr_s, hdr_p):
        if hdr is None:
            continue
        for k in ("WAVELEN", "WAVELNTH", "WAVELENG", "LAMBDA", "CWAVE", "WAVE"):
            if k not in hdr:
                continue
            try:
                v = float(hdr[k])
            except (TypeError, ValueError):
                continue
            if v <= 0:
                continue
            try:
                com = str(hdr.comments[k]).lower()
            except Exception:
                com = ""
            if "angstrom" in com or "ångström" in com or "[a]" in com or "(a)" in com or "å" in com:
                return v / 1e4
            if "nm" in com.split() or "[nm]" in com or "(nm)" in com or "nanom" in com:
                return v / 1e3
            if "micro" in com or "um" in com.split() or "µm" in com or "[um]" in com:
                return v
            return v / 1e4 if v >= 1000 else v
    return None


def lunghezza_onda_um(filtro, hdr_p=None, hdr_s=None):
    """Lunghezza d'onda centrale in micron (None se non determinabile)."""
    h = _norm_hdr(hdr_p, hdr_s)
    for k in ("PHOTPLAM", "PIVOTWAV"):
        try:
            v = float(h.get(k))
            if v > 0:
                return v / 1e4 if v > 50 else v           # Å -> µm (o già in µm)
        except (TypeError, ValueError):
            pass
    lo, hi = h.get("ENERG_LO"), h.get("ENERG_HI")
    try:
        e = (float(lo) + float(hi)) / 2
        if e > 0:
            return 12.398 / e / 1e4                         # keV -> Å -> µm
    except (TypeError, ValueError):
        pass
    tel = str(h.get("TELESCOP", "")).upper()
    strum = str(h.get("INSTRUME", "")).upper()
    f = str(filtro or "").upper()
    m = re.match(r"^F(\d{3,4})", f)
    if m:
        d = int(m.group(1))
        if "JWST" in tel or strum in ("NIRCAM", "MIRI", "NIRISS", "NIRSPEC"):
            return d / 100.0
        if "HST" in tel or "HUBBLE" in tel or strum in ("ACS", "WFC3", "WFPC2", "NICMOS", "STIS"):
            return d / 1000.0
        if "ROMAN" in tel or strum == "WFI":                # Roman: F158 = 1,58 µm (non 0,158)
            return roman_asdf.unita_lunghezza_onda_filtro(f)
        return d / 1000.0 if d >= 150 else d / 100.0
    chiave = re.sub(r"[^A-Z0-9_]", "", f.replace("-", "_").replace(" ", "_"))
    if chiave in LUNGHEZZE_NOTE_UM:
        return LUNGHEZZE_NOTE_UM[chiave]
    for k, v in LUNGHEZZE_NOTE_UM.items():
        if chiave.endswith(k) and len(k) >= 2:
            return v
    return _lunghezza_da_intestazione(hdr_p, hdr_s)


def descrivi_lunghezza(um):
    if um is None:
        return "?"
    if um < 0.01:
        return f"{12.398 / (um * 1e4):.1f} keV"
    if um < 1:
        return f"{um * 1000:.0f} nm"
    return f"{um:.2f} µm"


# =============================================================================
# LETTURA DEI FITS
# =============================================================================
def _hdu_scientifica(hdul):
    if "SCI" in hdul and hdul["SCI"].header.get("NAXIS", 0) >= 2:
        return hdul.index_of("SCI")
    for i, hdu in enumerate(hdul):
        h = hdu.header
        if str(h.get("XTENSION", "IMAGE")).strip().upper() in ("BINTABLE", "TABLE", "A3DTABLE"):
            if not str(h.get("ZIMAGE", "")).strip().upper() in ("T", "TRUE") and h.get("ZIMAGE") is not True:
                continue                     # tabelle (es. SPECTRAL_CHANNELS di SPHEREx), non immagini
        n = h.get("NAXIS", 0)
        if n == 2 or (n == 3 and h.get("NAXIS3", 1) == 1):
            if h.get("NAXIS1", 0) > 1 and h.get("NAXIS2", 0) > 1:
                return i
    raise ValueError("nessuna immagine 2D nel file")


def motivo_file_illeggibile(percorso, errore=None):
    """Spiegazione in parole semplici di un FITS che non si apre: spesso è un
    messaggio d'errore del server salvato al posto del file."""
    p = Path(percorso)
    try:
        dim = p.stat().st_size
        with open(p, "rb") as f:
            testa = f.read(512)
    except OSError as e:
        return f"non leggibile ({e})"
    if p.name.lower().endswith(".gz") and testa.startswith(b"\x1f\x8b"):
        return f"FITS compresso danneggiato o incompleto ({errore})"
    if not testa.startswith(b"SIMPLE"):
        testo = " ".join(r.strip() for r in testa.decode("utf-8", "replace").splitlines()
                         if r.strip() and ":" not in r[:30])[:160]
        return (f"NON è un FITS ({dim} byte): probabilmente lo scaricamento non è riuscito e il server "
                f"ha mandato un messaggio al posto del file" + (f" («{testo}»)" if testo else "")
                + ". Riscaricalo (Ricerca archivi 1.6 lo fa da sola).")
    if dim % 2880:
        return f"FITS incompleto ({dim} byte, non multiplo di 2880): riscaricalo"
    return f"FITS danneggiato ({type(errore).__name__}: {errore})"


def leggi_intestazioni(percorso):
    """(header primario, header scientifico, indice, forma (ny, nx))"""
    with fits.open(percorso, memmap=True, lazy_load_hdus=True) as hdul:
        i = _hdu_scientifica(hdul)
        hs = hdul[i].header.copy()
        hp = hdul[0].header.copy()
    return hp, hs, i, (int(hs["NAXIS2"]), int(hs["NAXIS1"]))


_RIQUADRI = {}


def riquadro_dati(img, passo=16, soglia=0.7):
    """Riquadro [y0, y1, x0, x1] che contiene tutti i pixel con dati (non NaN, non 0),
    se è più piccolo di 'soglia' dell'immagine intera; altrimenti None. Si leggono
    solo una riga ogni 'passo' (con un margine di 'passo' pixel), quindi è veloce
    anche sui mosaici enormi. Risultato ricordato per percorso, dimensione e data."""
    p = Path(img.percorso)
    st = p.stat()
    chiave = (str(p.resolve()), st.st_size, st.st_mtime)
    if chiave in _RIQUADRI:
        return _RIQUADRI[chiave]
    d = img.dati
    ny, nx = d.shape
    righe = np.asarray(d[::passo, :], dtype=np.float32)
    ok = np.isfinite(righe) & (righe != 0)
    r = None
    if ok.any():
        iy = np.where(ok.any(axis=1))[0]
        ix = np.where(ok.any(axis=0))[0]
        y0, y1 = max(0, iy[0] * passo - passo), min(ny, (iy[-1] + 1) * passo + passo)
        x0, x1 = max(0, ix[0] - passo), min(nx, ix[-1] + 1 + passo)
        if (y1 - y0) * (x1 - x0) < soglia * ny * nx:
            r = (int(y0), int(y1), int(x0), int(x1))
    _RIQUADRI[chiave] = r
    return r


class ImmagineFITS:
    """Accesso all'immagine scientifica (memmap quando possibile) e ai metadati."""

    def __init__(self, percorso):
        self.percorso = Path(percorso)
        self.hdr_p, self.hdr_s, self.indice, self.shape = leggi_intestazioni(percorso)
        self.filtro, self.base = rileva_filtro(self.hdr_p, self.hdr_s, self.percorso.name)
        self.lambda_um = lunghezza_onda_um(self.filtro, self.hdr_p, self.hdr_s)
        h = _norm_hdr(self.hdr_p, self.hdr_s)
        self.telescopio = str(h.get("TELESCOP", "")).strip()
        self.strumento = str(h.get("INSTRUME", "")).strip()
        self.conteggi = e_conteggi_x(self.hdr_p, self.hdr_s)       # raggi X: lo 0 è un dato
        self.campo_ridotto = "CHANDRA" in self.telescopio.upper() and str(h.get("CONTENT", "")).upper().startswith("LORES")
        try:
            self.esposizione = float(h.get("EXPOSURE") or h.get("LIVETIME") or h.get("ONTIME") or h.get("EXPTIME") or 0)
        except (TypeError, ValueError):
            self.esposizione = 0.0
        self._impronta = None
        self._hdul = None
        self._dati = None
        # Pan-STARRS: pixel in «luptitudini» (scala asinh) -> flusso lineare
        try:
            self.asinh_ps1 = (float(h["BSOFTEN"]), float(h["BOFFSET"])) if "BSOFTEN" in h and "BOFFSET" in h else None
        except (TypeError, ValueError):
            self.asinh_ps1 = None
        try:
            w = WCS(self.hdr_s, naxis=2)
            self.wcs = w.celestial if w.has_celestial else None
        except Exception:
            self.wcs = None
        # mosaici con grandi bordi vuoti (es. Spitzer SMOG: 94% di NaN): si lavora solo
        # sul riquadro che contiene dati, come se il file fosse già ritagliato
        self.shape_intera = self.shape
        self.riquadro = None
        if not self.conteggi:
            try:
                r = riquadro_dati(self)
            except Exception:
                r = None
            if r is not None:
                y0, y1, x0, x1 = r
                self.riquadro = r
                self.shape = (y1 - y0, x1 - x0)
                if self.wcs is not None:
                    self.wcs = self.wcs.slice((slice(y0, y1), slice(x0, x1)))
            self.chiudi()

    @property
    def dati(self):
        """Array 2D (memmap se il file non è compresso). Valori grezzi del file."""
        if self._dati is None:
            self._hdul = fits.open(self.percorso, memmap=True)
            try:
                d = self._hdul[self.indice].data
            except ValueError as e:
                # immagini compresse o intere con BZERO/BSCALE/BLANK (es. Pan-STARRS):
                # astropy non le può leggere in memmap, si caricano in memoria
                if "memmap" not in str(e).lower():
                    raise
                self._hdul.close()
                self._hdul = fits.open(self.percorso, memmap=False)
                d = self._hdul[self.indice].data
            if d.ndim == 3:
                d = d[0]
            if getattr(self, "riquadro", None) is not None:
                y0, y1, x0, x1 = self.riquadro
                d = d[y0:y1, x0:x1]
            if self.asinh_ps1 is not None:
                d = flusso_ps1(d, *self.asinh_ps1)
            self._dati = d
        return self._dati

    def chiudi(self):
        self._dati = None
        if self._hdul is not None:
            try:
                self._hdul.close()
            except Exception:
                pass
            self._hdul = None

    def impronta(self):
        """(maschera, fattore) della zona con dati, per le immagini a conteggi."""
        if self._impronta is None:
            self._impronta = impronta_conteggi(self.dati)
        return self._impronta

    def blocco(self, y0, y1, x0, x1):
        """Copia float32 di un blocco. Nei FITS ottici/infrarossi gli zeri esatti
        diventano NaN (lo 0 indica 'nessun dato'); nelle immagini a raggi X lo 0
        resta (nessun fotone) e diventano NaN solo i pixel fuori dal rivelatore."""
        b = np.array(self.dati[y0:y1, x0:x1], dtype=DTYPE)
        if self.conteggi:
            m, f = self.impronta()
            b[~maschera_su_griglia(m, f, y0, y1, x0, x1)] = np.nan
        else:
            b[b == 0] = np.nan
        return b

    def ridotta_valida(self, fattore):
        """Media a blocchi con i pixel senza dati a NaN (rispettando i raggi X)."""
        if not self.conteggi:
            return immagine_ridotta(self.dati, fattore)
        r = immagine_ridotta(self.dati, fattore, zeri_nan=False)
        m, f = self.impronta()
        ny, nx = self.shape
        r[~maschera_su_griglia(m, f, 0, r.shape[0] * fattore, 0, r.shape[1] * fattore, fattore)] = np.nan
        return r

    def centro_pixel(self, ra=None, dec=None):
        ny, nx = self.shape
        if ra is not None and dec is not None and self.wcs is not None:
            try:
                x, y = (float(np.ravel(v)[0]) for v in radec_a_pix(self.wcs, ra, dec, completo=True))
                if 0 <= x < nx and 0 <= y < ny:
                    return int(round(x)), int(round(y))
            except Exception:
                pass
        return nx // 2, ny // 2

    def ritaglio(self, ra, dec, lato):
        """Ritaglio a piena risoluzione centrato su (ra, dec) o sul centro."""
        ny, nx = self.shape
        xc, yc = self.centro_pixel(ra, dec)
        lato = int(min(lato, ny, nx))
        x0 = int(min(max(xc - lato // 2, 0), nx - lato))
        y0 = int(min(max(yc - lato // 2, 0), ny - lato))
        return self.blocco(y0, y0 + lato, x0, x0 + lato), (y0, x0)


def flusso_ps1(v, bsoften, boffset):
    """Pan-STARRS1: i pixel delle immagini «stack» sono in scala asinh
    (v = a·asinh((f − boffset) / 2·bsoften), a = 2,5/ln 10): torna al flusso lineare.
    I pixel senza dati sono già NaN nel file."""
    a = 2.5 / math.log(10)
    v = np.asarray(v, dtype=DTYPE)
    return (boffset + bsoften * 2 * np.sinh(v / a)).astype(DTYPE)


def campione(arr, n=1_500_000, zeri=False):
    """Campione regolare dei pixel finiti di un array grande; gli zeri esatti
    sono esclusi se zeri=False (dati grezzi, dove lo 0 vuol dire 'senza dati')."""
    ny, nx = arr.shape
    p = max(1, int(math.sqrt(ny * nx / n)))
    v = np.asarray(arr[::p, ::p], dtype=np.float64).ravel()
    return v[np.isfinite(v) & ((v != 0) | zeri)]


def stima_rumore_img(img):
    """Livello del fondo e rumore di un file; per i raggi X solo dentro il rivelatore."""
    if img.conteggi:
        ny, nx = img.shape
        p = max(1, int(math.sqrt(ny * nx / 4_000_000)))
        return stima_rumore(img.blocco(0, ny, 0, nx)[::p, ::p], zeri=True)
    return stima_rumore(img.dati)


def livello_cielo(arr, max_lato=600):
    """Livello del fondo cielo misurato solo nelle zone più scure (30% più
    scuro dell'immagine ridotta e sfocata): galassie e nebulose estese, che
    alzano la mediana di tutta l'immagine, non lo falsano."""
    f = fattore_ridotta(arr.shape, max_lato)
    r = immagine_ridotta(arr, f, zeri_nan=False) if f > 1 else np.array(arr, dtype=DTYPE)
    ok = np.isfinite(r)
    if ok.sum() < 50:
        return float(np.nanmedian(r)) if ok.any() else 0.0
    pieno = np.where(ok, r, np.median(r[ok]))
    liscio = ndi.gaussian_filter(pieno, 3)
    scuri = ok & (liscio <= np.percentile(liscio[ok], 30))
    return float(np.median(r[scuri]))


def stima_rumore(arr, zeri=False):
    """(livello del fondo, rumore). Il livello è la mediana sigma-clipped; il
    rumore è misurato da pixel a pixel, così nebulose e gradienti non lo gonfiano."""
    v = campione(arr, zeri=zeri)
    if v.size < 10:
        return 0.0, 1.0
    _, med, std = sigma_clipped_stats(v, sigma=3.0, maxiters=5)
    sig = rumore_pixel(arr)
    if not math.isfinite(sig) or sig <= 0:
        sig = std if std > 0 else 1.0
    return float(med), float(sig)


# =============================================================================
# RICERCA, RINOMINA E COPIA
# =============================================================================
def e_fits(p):
    n = p.name.lower()
    return any(n.endswith(e) for e in ESTENSIONI_FITS)


def e_output(p):
    return p.name.lower().startswith(PREFISSI_OUTPUT)


def elenca_file_pipeline(madre):
    """FITS nella cartella madre (non nelle sottocartelle), esclusi i prodotti della pipeline."""
    madre = Path(madre)
    if not madre.is_dir():
        return []
    return sorted(p for p in madre.iterdir() if p.is_file() and e_fits(p) and not e_output(p)
                  and not e_cubo_spettrale(p))


def e_cubo_spettrale(p):
    """True per i cubi spettrali (es. SPHEREx DR1): non sono immagini, vanno divisi in bande."""
    try:
        with fits.open(p, memmap=True, lazy_load_hdus=True) as hdul:
            try:
                _hdu_scientifica(hdul)
                return False
            except ValueError:
                return indice_cubo_spettrale(hdul) is not None
    except Exception:
        return False


def cubi_da_dividere(madre):
    """Cubi spettrali nella cartella madre (e sottocartelle) senza bande già ricavate e aggiornate."""
    madre = Path(madre)
    out = []
    for p in sorted(madre.rglob("*")):
        if (p.is_file() and e_fits(p) and p.parent.name != CARTELLA_BANDE_SPHEREX
                and CARTELLA_CACHE not in p.parts and e_cubo_spettrale(p)):
            bande = bande_spherex_esistenti(p)
            if not bande or any(b.stat().st_mtime < p.stat().st_mtime for b in bande):
                out.append(p)
    return out


def e_asdf(p):
    return roman_asdf.e_asdf(p)


def asdf_da_convertire(madre):
    """File .asdf di Roman nella cartella madre e nelle sottocartelle."""
    madre = Path(madre)
    return sorted(p for p in madre.rglob("*") if p.is_file() and e_asdf(p)
                  and not any(c in p.parts for c in CARTELLE_CACHE))


def _voce_roman(p, madre, log):
    """Voce della tabella di rinomina per un file Roman .asdf (None se non adatto)."""
    try:
        info = roman_asdf.leggi_info(p)
    except Exception as e:
        log(f"Saltato {p.name}: file ASDF non leggibile ({type(e).__name__}: {e})")
        return None
    problema = roman_asdf.problema_immagine(info)
    if problema:
        log(f"Saltato {p.name}: {problema}")
        return None
    forma = tuple(info["forma"][-2:])
    tel = f"{info['telescopio']} {info['strumento']}"
    return dict(percorso=p, relativo=str(p.relative_to(madre)), filtro=info["filtro"],
                base=roman_asdf.nome_base(info), lambda_um=info["lambda_um"], forma=forma,
                telescopio=info["telescopio"], strumento=f"{info['strumento']} {info['rivelatore']}".strip(),
                dimensione=forma[0] * forma[1] * 4 + 20000, nuovo_nome="", asdf=True,
                livello=info["livello"], descrizione=f"{tel} {info['livello']}")


# SPHEREx DR1: cubi spettrali (102 canali, 17 per ciascuno dei 6 rivelatori) in proiezione HEALPix
BANDE_SPHEREX = [(1, 0.75, 1.11), (2, 1.11, 1.64), (3, 1.64, 2.42), (4, 2.42, 3.82), (5, 3.82, 4.42), (6, 4.42, 5.01)]
CARTELLA_BANDE_SPHEREX = "spherex_bande"
_CHIAVI_WCS_2D = ("WCSAXES", "CRPIX1", "CRPIX2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "CD1_1", "CD1_2", "CD2_1",
                  "CD2_2", "CDELT1", "CDELT2", "CUNIT1", "CUNIT2", "CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2",
                  "PV2_1", "PV2_2", "LONPOLE", "LATPOLE", "RADESYS", "EQUINOX", "MJDREF")


def indice_cubo_spettrale(hdul):
    """Indice dell'estensione con un cubo spettrale (3 assi, il terzo in lunghezza d'onda), o None."""
    for i, hdu in enumerate(hdul):
        h = hdu.header
        if h.get("NAXIS", 0) == 3 and h.get("NAXIS3", 1) > 1 and (
                str(h.get("CTYPE3", "")).upper().startswith(("WAVE", "FREQ", "AWAV")) or
                str(h.get("CNAME3", "")).upper().startswith("WAVE")):
            return i
    return None


def bande_spherex_esistenti(p):
    d = Path(p).parent / CARTELLA_BANDE_SPHEREX
    return sorted(d.glob(f"{Path(p).stem}_D[1-6]_spx.fits")) if d.is_dir() else []


def converti_cubo_spherex(p, log=print):
    """Cubo SPHEREx (o altro cubo spettrale) -> 6 immagini 2D, una per rivelatore:
    media dei canali pesata con NHITS (copertura), NaN dove non ci sono dati.
    Scrive spherex_bande/<nome>_D1_spx.fits ... accanto al cubo; restituisce i percorsi."""
    p = Path(p)
    uscita = p.parent / CARTELLA_BANDE_SPHEREX
    uscita.mkdir(exist_ok=True)
    scritti = []
    with fits.open(p, memmap=True) as hdul:
        i = indice_cubo_spettrale(hdul)
        if i is None:
            raise ValueError("nessun cubo spettrale nel file")
        h = hdul[i].header
        hp = hdul[0].header
        n3 = int(h["NAXIS3"])
        lam = None
        try:
            tab = hdul[str(h.get("PS3_0", "WCS-WAVE")).strip()].data
            lam = np.ravel(np.asarray(tab[str(h.get("PS3_1", "WAVELENGTHS")).strip()], dtype=float))
            if lam.size != n3:
                lam = None
        except Exception:
            lam = None
        if lam is None:
            lam = np.linspace(float(h.get("WLMIN", 0.75)), float(h.get("WLMAX", 5.0)), n3)
        cubo = hdul[i].data
        copertura = None
        if "NHITS" in hdul and hdul["NHITS"].header.get("NAXIS3") == n3:
            copertura = hdul["NHITS"].data
        tel = str(hp.get("TELESCOP", h.get("TELESCOP", "SPHEREx"))).strip() or "SPHEREx"
        for det, a, b in BANDE_SPHEREX:
            if n3 == 102:                    # SPHEREx: 17 canali per rivelatore, in ordine
                canali = list(range((det - 1) * 17, det * 17))
            else:
                canali = [k for k in range(n3) if a <= lam[k] < b]
            if not canali:
                continue
            somma = np.zeros(cubo.shape[1:], np.float64)
            peso = np.zeros(cubo.shape[1:], np.float64)
            for k in canali:
                piano = np.asarray(cubo[k], dtype=np.float64)
                pk = np.asarray(copertura[k], dtype=np.float64) if copertura is not None else np.ones_like(piano)
                ok = np.isfinite(piano) & (pk > 0)
                somma[ok] += piano[ok] * pk[ok]
                peso[ok] += pk[ok]
            with np.errstate(invalid="ignore", divide="ignore"):
                img = np.where(peso > 0, somma / peso, np.nan).astype(np.float32)
            if not np.isfinite(img).any():
                continue
            hh = fits.Header()
            for k in _CHIAVI_WCS_2D:
                if k in h:
                    hh[k] = (h[k], h.comments[k])
            hh["WCSAXES"] = 2
            l_media = float(np.mean(lam[canali]))
            hh["TELESCOP"] = tel
            hh["INSTRUME"] = str(hp.get("INSTRUME", "SPHEREx")).strip() or "SPHEREx"
            hh["DETECTOR"] = (det, "rivelatore SPHEREx (banda)")
            hh["WAVELEN"] = (round(l_media, 4), "[micron] media dei canali usati")
            hh["WAVEMIN"] = (round(float(np.min(lam[canali])), 4), "[micron]")
            hh["WAVEMAX"] = (round(float(np.max(lam[canali])), 4), "[micron]")
            hh["NCANALI"] = (len(canali), "canali del cubo mediati")
            hh["BUNIT"] = h.get("BUNIT", "MJy/sr")
            hh["ORIGFILE"] = (p.name[:60], "cubo spettrale di origine")
            hh["COMMENT"] = f"AutoAstroPipe {VERSIONE}: banda D{det} dal cubo SPHEREx, media pesata con NHITS"
            q = uscita / f"{p.stem}_D{det}_spx.fits"
            fits.PrimaryHDU(img, hh).writeto(q, overwrite=True)
            scritti.append(q)
            log(f"  {q.name}: canali {canali[0] + 1}-{canali[-1] + 1}, {l_media:.2f} µm")
    return scritti


def trova_fits_da_rinominare(madre, log=print, salta=()):
    """FITS nelle sottocartelle (a qualsiasi profondità) con il nome proposto, e
    file .asdf di Roman (anche nella cartella madre), che verranno convertiti in FITS.
    'salta' = percorsi relativi dei file .asdf già convertiti in passato."""
    madre = Path(madre)
    trovati = []
    salta = set(salta)
    for p in sorted(madre.rglob("*")):
        if not p.is_file() or any(c in p.parts for c in CARTELLE_CACHE):
            continue
        if e_asdf(p):
            if str(p.relative_to(madre)) in salta:
                log(f"{p.name}: già convertito in FITS in precedenza, lo salto")
                continue
            voce = _voce_roman(p, madre, log)
            if voce:
                trovati.append(voce)
            continue
        if not e_fits(p):
            continue
        if p.parent == madre and not e_cubo_spettrale(p):
            continue                         # già nella madre (i cubi invece vanno divisi)
        if p.parent.name == CARTELLA_BANDE_SPHEREX:
            continue                         # bande già ricavate da un cubo: aggiunte sotto
        try:
            hp, hs, _, shape = leggi_intestazioni(p)
        except ValueError as e:
            # cubo spettrale (es. SPHEREx DR1): una immagine per banda
            try:
                with fits.open(p, memmap=True, lazy_load_hdus=True) as hdul:
                    cubo = indice_cubo_spettrale(hdul) is not None
            except Exception:
                cubo = False
            if not cubo:
                log(f"Saltato {p.name}: {e}")
                continue
            bande = bande_spherex_esistenti(p)
            if bande and all(b.stat().st_mtime >= p.stat().st_mtime for b in bande):
                log(f"{p.name}: cubo spettrale, uso le {len(bande)} bande già ricavate in {CARTELLA_BANDE_SPHEREX}/")
            else:
                log(f"{p.name}: cubo spettrale, ricavo un'immagine per banda (rivelatori D1-D6)")
                try:
                    bande = converti_cubo_spherex(p, log)
                except Exception as e2:
                    log(f"Saltato {p.name}: cubo non convertibile ({e2})")
                    continue
            for b in bande:
                _aggiungi_trovato(trovati, b, madre, log)
            continue
        except Exception as e:
            log(f"Saltato {p.name}: {e}")
            continue
        _aggiungi_trovato(trovati, p, madre, log, (hp, hs, shape))
    assegna_nomi(trovati, madre)
    return trovati


def _aggiungi_trovato(trovati, p, madre, log, intestazioni=None):
    try:
        if intestazioni:
            hp, hs, shape = intestazioni
        else:
            hp, hs, _, shape = leggi_intestazioni(p)
    except Exception as e:
        log(f"Saltato {p.name}: {e}")
        return
    filtro, base = rileva_filtro(hp, hs, p.name)
    h = _norm_hdr(hp, hs)
    trovati.append(dict(percorso=p, relativo=str(p.relative_to(madre)), filtro=filtro, base=base,
                        lambda_um=lunghezza_onda_um(filtro, hp, hs), forma=shape,
                        telescopio=str(h.get("TELESCOP", "")).strip(),
                        strumento=str(h.get("INSTRUME", "")).strip(),
                        dimensione=p.stat().st_size, nuovo_nome=""))


def assegna_nomi(trovati, madre):
    """Nome nuovo dal filtro; se più file hanno lo stesso filtro: f090a, f090b, ..."""
    madre = Path(madre)
    esistenti = {p.name.lower() for p in madre.iterdir() if p.is_file()}
    gruppi = {}
    for t in trovati:
        gruppi.setdefault(t["base"], []).append(t)
    for base, lista in gruppi.items():
        estensione = ".fits"
        if len(lista) == 1 and f"{base}{estensione}" not in esistenti:
            lista[0]["nuovo_nome"] = f"{base}{estensione}"
            continue
        lettere = (chr(c) for c in range(ord("a"), ord("z") + 1))
        for t in lista:
            for l in lettere:
                nome = f"{base}{l}{estensione}"
                if nome not in esistenti:
                    t["nuovo_nome"] = nome
                    esistenti.add(nome)
                    break
            else:
                t["nuovo_nome"] = f"{base}_{len(esistenti)}{estensione}"
        esistenti.update(t["nuovo_nome"] for t in lista)


def _md5(percorso, blocco=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(percorso, "rb") as f:
        while True:
            b = f.read(blocco)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copia_e_verifica(trovati, madre, progresso=None, log=print):
    """Copia ogni file nella cartella madre con il nuovo nome e verifica che la
    copia sia identica (dimensione + MD5). I file .gz vengono decompressi.
    I file Roman .asdf vengono convertiti in FITS e si verifica che i pixel
    del FITS siano identici a quelli letti dall'ASDF."""
    madre = Path(madre)
    copiati = []
    for n, t in enumerate(trovati, 1):
        dest = madre / t["nuovo_nome"]
        if progresso:
            progresso(n, len(trovati), t["relativo"])
        src = t["percorso"]
        if t.get("asdf"):
            try:
                ok, descr, _info = roman_asdf.converti_in_fits(
                    src, dest, log, creatore=f"{NOME_PROGRAMMA} {VERSIONE}", autore=AUTORE,
                    nota_ascii=f"{COPYRIGHT_ASCII} - convertito con {NOME_PROGRAMMA} ({SIGLA}) {VERSIONE}")
                t["wcs"] = descr
            except Exception as e:
                log(f"ERRORE nella conversione di {t['relativo']}: {type(e).__name__}: {e}")
                ok = False
            if ok:
                copiati.append(t)
                log(f"{t['relativo']}  ->  {t['nuovo_nome']}  (Roman {t.get('livello', '')}; coordinate: {t.get('wcs')})")
            else:
                log(f"ERRORE: conversione non riuscita di {t['relativo']} (l'originale NON verrà cancellato)")
                try:
                    dest.unlink()
                except OSError:
                    pass
            continue
        if src.name.lower().endswith(".gz"):
            with fits.open(src) as hdul:
                hdul.writeto(dest, overwrite=False)
            ok = dest.exists()
        else:
            shutil.copy2(src, dest)
            ok = dest.stat().st_size == src.stat().st_size and _md5(src) == _md5(dest)
        if ok:
            copiati.append(t)
            log(f"{t['relativo']}  ->  {t['nuovo_nome']}")
        else:
            log(f"ERRORE: copia non identica di {t['relativo']} (l'originale NON verrà cancellato)")
            try:
                dest.unlink()
            except OSError:
                pass
    return copiati


def cancella_originali(copiati, madre, log=print):
    """Cancella gli originali copiati e poi le sottocartelle rimaste vuote."""
    madre = Path(madre).resolve()
    cartelle = set()
    for t in copiati:
        try:
            t["percorso"].unlink()
            cartelle.add(t["percorso"].parent)
        except OSError as e:
            log(f"Non cancellato {t['relativo']}: {e}")
    for c in sorted(cartelle, key=lambda p: len(p.parts), reverse=True):
        p = c.resolve()
        while p != madre and madre in p.parents:
            try:
                p.rmdir()                       # solo se vuota
            except OSError:
                break
            p = p.parent


# =============================================================================
# RICOSTRUZIONE DELLE ZONE ROVINATE
# =============================================================================
def immagine_ridotta(sorgente, fattore, zeri_nan=True):
    """Media a blocchi fattore×fattore ignorando i NaN (lettura a strisce)."""
    ny, nx = sorgente.shape
    f = max(1, int(fattore))
    nyb, nxb = math.ceil(ny / f), math.ceil(nx / f)
    out = np.empty((nyb, nxb), dtype=DTYPE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for jb in range(nyb):
            s = np.array(sorgente[jb * f:(jb + 1) * f, :], dtype=DTYPE)
            if zeri_nan:
                s[s == 0] = np.nan
            if s.shape[1] < nxb * f:
                s = np.pad(s, ((0, 0), (0, nxb * f - s.shape[1])), constant_values=np.nan)
            out[jb] = np.nanmean(s.reshape(s.shape[0], nxb, f), axis=(0, 2))
    return out


def fattore_ridotta(shape, max_lato=MAX_LATO_RIDOTTA):
    return max(1, math.ceil(max(shape) / max_lato))


def trova_zone_rovinate(img, max_zone=2000):
    """Zone senza dati (NaN o 0) all'INTERNO dell'impronta dell'immagine
    (i bordi esterni non contano). Restituisce riquadri [y0, y1, x0, x1]."""
    f = fattore_ridotta(img.shape)
    rid = img.ridotta_valida(f)
    validi = np.isfinite(rid)
    impronta = ndi.binary_fill_holes(validi)
    buchi = impronta & ~validi
    etichette, n = ndi.label(buchi)
    if n == 0:
        return []
    zone = []
    oggetti = ndi.find_objects(etichette)
    aree = ndi.sum(buchi, etichette, index=np.arange(1, n + 1))
    ordine = np.argsort(aree)[::-1][:max_zone]
    ny, nx = img.shape
    m = 2 * f + 8
    for k in ordine:
        sl = oggetti[k]
        y0, y1 = max(sl[0].start * f - m, 0), min(sl[0].stop * f + m, ny)
        x0, x1 = max(sl[1].start * f - m, 0), min(sl[1].stop * f + m, nx)
        zone.append([int(y0), int(y1), int(x0), int(x1)])
    return zone


def _riempi(sub, da_riempire):
    """Riempie i pixel indicati interpolando dai vicini con gaussiane crescenti."""
    tmp = sub.astype(np.float64)
    tmp[da_riempire] = np.nan
    for s in (1.5, 3, 6, 12, 24, 48):
        if not np.isnan(tmp[da_riempire]).any():
            break
        k = Gaussian2DKernel(s)
        pieno = interpolate_replace_nans(tmp, k, convolve=convolve_fft)
        sel = da_riempire & np.isnan(tmp)
        tmp[sel] = pieno[sel]
    rest = da_riempire & np.isnan(tmp)
    if rest.any():
        v = tmp[np.isfinite(tmp)]
        tmp[rest] = np.median(v) if v.size else 0.0
    sub[da_riempire] = tmp[da_riempire].astype(sub.dtype)


def ripara(arr, y0, x0, zone_auto=(), rettangoli=()):
    """Ricostruisce in place, sull'array 'arr' che inizia al pixel (y0, x0)
    dell'immagine intera: i buchi interni nelle zone automatiche e tutti i
    pixel dentro i rettangoli disegnati dall'utente."""
    H, W = arr.shape
    for tipo, zone in (("auto", zone_auto), ("manuale", rettangoli)):
        for (zy0, zy1, zx0, zx1) in zone:
            a0, a1 = max(zy0 - y0, 0), min(zy1 - y0, H)
            b0, b1 = max(zx0 - x0, 0), min(zx1 - x0, W)
            if a0 >= a1 or b0 >= b1:
                continue
            m = 24                                  # anello di pixel validi attorno
            c0, c1, d0, d1 = max(a0 - m, 0), min(a1 + m, H), max(b0 - m, 0), min(b1 + m, W)
            sub = arr[c0:c1, d0:d1]
            if tipo == "manuale":
                bersaglio = np.zeros(sub.shape, bool)
                bersaglio[a0 - c0:a1 - c0, b0 - d0:b1 - d0] = True
            else:
                invalidi = ~np.isfinite(sub)
                lab, n = ndi.label(invalidi)
                if n == 0:
                    continue
                bordo = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
                bersaglio = invalidi & ~np.isin(lab, bordo[bordo > 0])
                if not bersaglio.any():
                    continue
            _riempi(sub, bersaglio)
    return arr


# =============================================================================
# RIDUZIONE DEL RUMORE
# =============================================================================
def margine_nr(algo, valore):
    v = float(valore)
    return {"gaussiano": int(4 * v) + 2, "box": int(v) + 2, "mediano": int(v) + 2,
            "wavelet": 64, "tv": 32, "nlmeans": 24}.get(algo, 32)


def riduci_rumore(arr, algo, valore, mediana, sigma):
    """Riduzione del rumore su un array (i NaN restano NaN)."""
    if valore is None or float(valore) <= 0:
        return np.array(arr, dtype=DTYPE, copy=True)
    arr = np.asarray(arr, dtype=DTYPE)
    nan = ~np.isfinite(arr)
    v = float(valore)
    if algo == "gaussiano":
        out = convolve_fft(arr.astype(np.float64), Gaussian2DKernel(v), nan_treatment="interpolate",
                           preserve_nan=True, allow_huge=True)
        return out.astype(DTYPE)
    if algo == "box":
        out = convolve_fft(arr.astype(np.float64), Box2DKernel(max(2, int(round(v)))),
                           nan_treatment="interpolate", preserve_nan=True, allow_huge=True)
        return out.astype(DTYPE)
    pieno = np.where(nan, mediana, arr).astype(np.float64)
    if algo == "mediano":
        n = max(3, int(round(v)) | 1)
        out = ndi.median_filter(pieno, size=n)
    else:
        x = (pieno - mediana) / sigma                   # rumore del fondo = 1
        if algo == "wavelet":
            from skimage.restoration import denoise_wavelet
            out = denoise_wavelet(x, sigma=v, wavelet="sym4", mode="soft", method="BayesShrink",
                                  rescale_sigma=True)
        elif algo == "tv":
            from skimage.restoration import denoise_tv_chambolle
            out = denoise_tv_chambolle(x, weight=v)
        elif algo == "nlmeans":
            from skimage.restoration import denoise_nl_means
            out = denoise_nl_means(x, h=v, sigma=1.0, patch_size=5, patch_distance=6, fast_mode=True,
                                   preserve_range=True)
        else:
            raise ValueError(f"algoritmo di riduzione del rumore sconosciuto: {algo}")
        out = out * sigma + mediana
    out = out.astype(DTYPE)
    out[nan] = np.nan
    return out


def riduci_rumore_tasselli(src, dst, algo, valore, mediana, sigma, progresso=None):
    """Come riduci_rumore ma a tasselli sovrapposti (per immagini enormi)."""
    ny, nx = src.shape
    m = margine_nr(algo, valore)
    tot = math.ceil(ny / TILE) * math.ceil(nx / TILE)
    k = 0
    for y0 in range(0, ny, TILE):
        y1 = min(y0 + TILE, ny)
        ya, yb = max(y0 - m, 0), min(y1 + m, ny)
        for x0 in range(0, nx, TILE):
            x1 = min(x0 + TILE, nx)
            xa, xb = max(x0 - m, 0), min(x1 + m, nx)
            r = riduci_rumore(np.asarray(src[ya:yb, xa:xb], dtype=DTYPE), algo, valore, mediana, sigma)
            dst[y0:y1, x0:x1] = r[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
            k += 1
            if progresso:
                progresso(k, tot)
    if hasattr(dst, "flush"):
        dst.flush()
    return dst


# =============================================================================
# SOTTRAZIONE DEL FONDO (calcolata sull'immagine ridotta, poi interpolata)
# =============================================================================
def _stimatore(algo):
    from photutils.background import (MedianBackground, SExtractorBackground, MMMBackground,
                                      MeanBackground)
    return {"mediana": MedianBackground, "sextractor": SExtractorBackground,
            "mmm": MMMBackground, "media": MeanBackground}[algo]()


def maschera_sorgenti(rid):
    """Maschera delle sorgenti sull'immagine ridotta (esclude stelle e nebulosa)."""
    from photutils.segmentation import detect_sources
    import inspect
    validi = np.isfinite(rid)
    if validi.sum() < 100:
        return np.zeros(rid.shape, bool)
    _, med, std = sigma_clipped_stats(rid[validi], sigma=3.0)
    arg = "n_pixels" if "n_pixels" in inspect.signature(detect_sources).parameters else "npixels"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                # "No sources were found" non è un errore
        segm = detect_sources(np.where(validi, rid, med), med + 3 * std, mask=~validi, **{arg: 5})
    m = np.zeros(rid.shape, bool) if segm is None else segm.data > 0
    return ndi.binary_dilation(m, iterations=2)


def mappa_fondo_ridotta(rid, fattore, algo, box, maschera=None):
    """Mappa del fondo alla risoluzione dell'immagine ridotta."""
    from photutils.background import Background2D
    copertura = ~np.isfinite(rid)
    box_r = max(4, int(round(box / fattore)))
    box_r = min(box_r, max(4, min(rid.shape) // 2))
    m = maschera & ~copertura if maschera is not None else None
    for esclusione in (50, 80, 95):
        try:
            bkg = Background2D(rid, box_size=box_r, filter_size=3, sigma_clip=SigmaClip(sigma=3.0),
                               bkg_estimator=_stimatore(algo), mask=m, coverage_mask=copertura,
                               exclude_percentile=esclusione)
            mappa = np.array(bkg.background, dtype=DTYPE)
            break
        except ValueError:
            continue
    else:
        v = rid[np.isfinite(rid)]
        mappa = np.full(rid.shape, np.median(v) if v.size else 0, dtype=DTYPE)
    # dove non ci sono dati: valore del pixel valido più vicino (serve all'interpolazione)
    if copertura.any() and (~copertura).any():
        idx = ndi.distance_transform_edt(copertura, return_distances=False, return_indices=True)
        mappa = mappa[tuple(idx)]
    return mappa


def fondo_regione(mappa_rid, fattore, y0, y1, x0, x1):
    """Fondo a piena risoluzione per la regione [y0:y1, x0:x1] (interpolazione lineare)."""
    yy = (np.arange(y0, y1, dtype=np.float64) + 0.5) / fattore - 0.5
    xx = (np.arange(x0, x1, dtype=np.float64) + 0.5) / fattore - 0.5
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    return ndi.map_coordinates(mappa_rid, [Y, X], order=1, mode="nearest").astype(DTYPE)


# =============================================================================
# STRETCHING (sempre con ZScaleInterval)
# =============================================================================
def limiti_zscale(arr):
    v = campione(arr, zeri=True)
    if v.size < 10:
        return 0.0, 1.0
    try:
        vmin, vmax = ZScaleInterval().get_limits(v)
    except Exception:
        vmin, vmax = np.percentile(v, [1, 99.5])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(v)), float(np.max(v))
        if vmax <= vmin:
            vmax = vmin + 1e-5
    return float(vmin), float(vmax)


def crea_stretch(algo, valore, campione_norm=None):
    if algo == "lineare":
        return LinearStretch()
    if algo == "asinh":
        return AsinhStretch(a=float(valore))
    if algo == "log":
        return LogStretch(a=float(valore))
    if algo == "sqrt":
        return SqrtStretch()
    if algo == "power":
        return PowerStretch(a=float(valore))
    if algo == "square":
        return SquaredStretch()
    if algo == "sinh":
        return SinhStretch(a=float(valore))
    if algo == "powerdist":
        return PowerDistStretch(a=float(valore))
    if algo == "histeq":
        c = campione_norm if campione_norm is not None and campione_norm.size else np.linspace(0, 1, 100)
        return HistEqStretch(np.asarray(c, dtype=np.float64))
    raise ValueError(f"stretching sconosciuto: {algo}")


def descrivi_stretch(algo, valore):
    info = ALGORITMI_STRETCH.get(algo, {})
    nome = info.get("etichetta", algo)
    if info.get("parametro") and valore is not None:
        return f"{nome} {info['parametro']}={float(valore):g}"
    return nome


def applica_stretch(arr, algo, valore, limiti=None):
    """Normalizza con ZScale (o i limiti dati) e applica lo stretching -> [0, 1]."""
    vmin, vmax = limiti if limiti is not None else limiti_zscale(arr)
    x = (np.asarray(arr, dtype=DTYPE) - vmin) / (vmax - vmin)
    nan = ~np.isfinite(x)
    x = np.clip(np.where(nan, 0, x), 0, 1)
    camp = None
    if algo == "histeq":
        camp = x[::max(1, x.shape[0] // 400), ::max(1, x.shape[1] // 400)].ravel()
    s = crea_stretch(algo, valore, camp)
    out = np.clip(s(x, clip=True), 0, 1).astype(DTYPE)
    out[nan] = np.nan
    return out


# =============================================================================
# SNR
# =============================================================================
def maschera_segnale(arr):
    """Pixel con segnale (> 3σ sopra il fondo), calcolati UNA volta sull'immagine
    di riferimento: così l'SNR delle anteprime si confronta sugli stessi pixel."""
    v = campione(arr, 1_000_000, zeri=True)
    if v.size < 10:
        return None
    _, med, std = sigma_clipped_stats(v, sigma=3.0, maxiters=5)
    a = np.asarray(arr, dtype=DTYPE)
    with np.errstate(invalid="ignore"):
        return np.isfinite(a) & (a > med + 3 * std)


def rumore_pixel(arr, escludi=None, max_righe=600):
    """Rumore da pixel a pixel: MAD delle differenze tra pixel vicini / √2.
    Non è influenzato dalle strutture estese (nebulose, gradienti) e scende
    davvero quando si riduce il rumore. 'escludi' = pixel con segnale da ignorare."""
    a = np.asarray(arr, dtype=DTYPE)
    p = max(1, a.shape[0] // max_righe)
    a = a[::p]
    d = np.diff(a, axis=1)
    ok = np.isfinite(d)
    if escludi is not None:
        e = np.asarray(escludi)[::p]
        ok &= ~(e[:, 1:] | e[:, :-1])
    d = d[ok]
    if d.size < 10:
        return float("nan")
    mad = 1.4826 * np.median(np.abs(d - np.median(d)))
    if mad <= 0:                                   # immagine quasi piatta (molto levigata)
        mad = float(np.std(d))
    return float(mad / math.sqrt(2))


def snr_medio(arr, metodo="segnale", maschera=None):
    """'globale': media / deviazione standard di tutti i pixel.
    'segnale': (luminosità media dei pixel con segnale − fondo) / rumore da pixel a pixel."""
    v = campione(arr, 1_000_000, zeri=True)
    if v.size < 10:
        return float("nan")
    if metodo == "globale":
        s = v.std()
        return float(v.mean() / s) if s > 0 else float("nan")
    a = np.asarray(arr, dtype=DTYPE)
    if maschera is None:
        maschera = maschera_segnale(a)
        if maschera is None:
            return float("nan")
    fondo = a[~maschera & np.isfinite(a)]
    seg = a[maschera & np.isfinite(a)]
    sigma = rumore_pixel(a, maschera)
    if not seg.size or not fondo.size or not sigma or not math.isfinite(sigma):
        return 0.0
    return float((np.mean(seg) - np.median(fondo)) / sigma)


# =============================================================================
# ELABORAZIONE COMPLETA DI UN FILE (piena risoluzione)
# =============================================================================
def chiave_scelte(img, s):
    st = img.percorso.stat()
    testo = json.dumps({"f": str(img.percorso), "t": st.st_mtime, "n": st.st_size, "x": bool(img.conteggi),
                        "nr": s.get("nr"), "bg": s.get("bg") if s.get("usa_bg", True) else None,
                        "st": s.get("stretch"), "rp": s.get("riparazione")}, sort_keys=True, default=str)
    return hashlib.md5(testo.encode()).hexdigest()[:12]


class Pipeline:
    """Calcoli che usano la cache su disco nella cartella madre."""

    def __init__(self, madre, memoria):
        self.madre = Path(madre)
        self.cache = self.madre / CARTELLA_CACHE
        self.cache.mkdir(parents=True, exist_ok=True)
        self.mem = memoria
        self.mem.cartella = self.cache
        self._ridotte = {}         # nome file -> (ridotta, fattore)
        self._fondi = {}           # (nome, algo, box) -> mappa ridotta
        self._maschere = {}

    # ---------- immagini ridotte (fondo, panoramica)
    def ridotta(self, img):
        k = img.percorso.name
        if k not in self._ridotte:
            f = fattore_ridotta(img.shape)
            p = self.cache / f"rid_{img.percorso.stem}_{f}_{int(img.percorso.stat().st_mtime)}.npy"
            if p.exists():
                r = np.load(p)
            else:
                r = img.ridotta_valida(f)
                np.save(p, r)
            self._ridotte[k] = (r, f)
        return self._ridotte[k]

    def mappa_fondo(self, img, algo, box, scelte=None):
        k = (img.percorso.name, algo, int(box))
        if k not in self._fondi:
            rid, f = self.ridotta(img)
            if scelte and scelte.get("riparazione"):
                rid = rid.copy()
                rp = scelte["riparazione"]
                zone = [[z[0] // f, math.ceil(z[1] / f), z[2] // f, math.ceil(z[3] / f)]
                        for z in rp.get("zone_auto", [])] if rp.get("auto") else []
                rett = [[z[0] // f, math.ceil(z[1] / f), z[2] // f, math.ceil(z[3] / f)]
                        for z in rp.get("rettangoli", [])]
                ripara(rid, 0, 0, zone, rett)
            if img.percorso.name not in self._maschere:
                self._maschere[img.percorso.name] = maschera_sorgenti(rid)
            self._fondi[k] = mappa_fondo_ridotta(rid, f, algo, box, self._maschere[img.percorso.name])
        rid, f = self._ridotte[img.percorso.name]
        return self._fondi[k], f

    def dimentica_file(self, nome):
        self._ridotte.pop(nome, None)
        self._maschere.pop(nome, None)
        for k in [k for k in self._fondi if k[0] == nome]:
            self._fondi.pop(k)

    # ---------- anteprime (ritaglio)
    def ritaglio_preparato(self, img, ra, dec, lato, scelte):
        crop, (y0, x0) = img.ritaglio(ra, dec, lato)
        rp = scelte.get("riparazione") or {}
        zone = rp.get("zone_auto", []) if rp.get("auto") else []
        if zone or rp.get("rettangoli"):
            ripara(crop, y0, x0, zone, rp.get("rettangoli", []))
        return crop, (y0, x0)

    def fondo_ritaglio(self, img, algo, box, y0, x0, shape, scelte=None):
        mappa, f = self.mappa_fondo(img, algo, box, scelte)
        return fondo_regione(mappa, f, y0, y0 + shape[0], x0, x0 + shape[1])

    # ---------- elaborazione completa
    def elabora(self, img, scelte, rumore, progresso=None, log=print):
        """Riparazione -> riduzione rumore -> fondo -> stretching, a piena
        risoluzione. Restituisce (array 'stretch' in [0,1], limiti ZScale).
        Il risultato viene messo in cache su disco (si riusa se le scelte non cambiano)."""
        chiave = chiave_scelte(img, scelte)
        p_st = self.cache / f"elab_{img.percorso.stem}_{chiave}.npy"
        p_info = p_st.with_suffix(".json")
        if p_st.exists() and p_info.exists():
            log(f"{img.percorso.name}: uso il risultato già calcolato (cache)")
            return np.load(p_st, mmap_mode="r"), json.loads(p_info.read_text())["limiti"]
        ny, nx = img.shape
        med, sig = rumore
        log(f"{img.percorso.name}: copia di lavoro {nx}×{ny}")
        lav = self.mem.alloca("lavoro", (ny, nx))
        for y in range(0, ny, TILE):
            lav[y:y + TILE] = img.blocco(y, min(y + TILE, ny), 0, nx)
        rp = scelte.get("riparazione") or {}
        zone = rp.get("zone_auto", []) if rp.get("auto") else []
        if zone or rp.get("rettangoli"):
            log(f"{img.percorso.name}: ricostruzione di {len(zone) + len(rp.get('rettangoli', []))} zone")
            m = 32                                      # anello di pixel validi per interpolare
            for tipo, lista in (("auto", zone), ("manuale", rp.get("rettangoli", []))):
                for z in lista:
                    y0, y1 = max(z[0] - m, 0), min(z[1] + m, ny)
                    x0, x1 = max(z[2] - m, 0), min(z[3] + m, nx)
                    sub = np.array(lav[y0:y1, x0:x1])
                    ripara(sub, y0, x0, [z] if tipo == "auto" else [], [z] if tipo == "manuale" else [])
                    lav[y0:y1, x0:x1] = sub
        nr = scelte.get("nr") or {}
        if nr.get("valore"):
            log(f"{img.percorso.name}: riduzione del rumore {nr['algo']} {nr['valore']}")
            out = self.mem.alloca("nr", (ny, nx))
            riduci_rumore_tasselli(lav, out, nr["algo"], nr["valore"], med, sig,
                                   lambda k, t: progresso and progresso("rumore", k, t))
            self.mem.libera(lav)
            del lav
            lav = out
        bg = scelte.get("bg") or {}
        if scelte.get("usa_bg", True) and bg.get("box"):
            log(f"{img.percorso.name}: sottrazione del fondo {bg['algo']} box {bg['box']}")
            mappa, f = self.mappa_fondo(img, bg["algo"], bg["box"], scelte)
            for y in range(0, ny, 1024):
                y1 = min(y + 1024, ny)
                lav[y:y1] = np.asarray(lav[y:y1]) - fondo_regione(mappa, f, y, y1, 0, nx)
                if progresso:
                    progresso("fondo", y1, ny)
        st = scelte.get("stretch") or {"algo": "asinh", "valore": 0.1}
        if st.get("algo") == "nessuno":
            # dati LINEARI per la rifinitura con i colori fedeli (Lupton): fondo a 0 e
            # unità di rumore (σ), così ogni filtro ha lo stesso rumore di fondo
            _, sig2 = stima_rumore(lav)
            sig2 = sig2 if sig2 > 0 else 1.0
            med2 = livello_cielo(lav)
            log(f"{img.percorso.name}: dati lineari in unità di rumore (fondo {med2:.4g}, σ {sig2:.4g})")
            uscita = np.lib.format.open_memmap(p_st, mode="w+", dtype=DTYPE, shape=(ny, nx))
            for y in range(0, ny, 1024):
                y1 = min(y + 1024, ny)
                uscita[y:y1] = ((np.asarray(lav[y:y1], dtype=DTYPE) - med2) / sig2).astype(DTYPE)
                if progresso:
                    progresso("dati lineari", y1, ny)
            uscita.flush()
            del uscita
            p_info.write_text(json.dumps({"limiti": [med2, sig2], "scelte": scelte}, default=str))
            self.mem.libera(lav)
            del lav
            self.mem.pulisci_temporanei()
            restituisci_ram()
            return np.load(p_st, mmap_mode="r"), [med2, sig2]
        limiti = limiti_zscale(lav)
        log(f"{img.percorso.name}: stretching {descrivi_stretch(st['algo'], st['valore'])}")
        uscita = np.lib.format.open_memmap(p_st, mode="w+", dtype=DTYPE, shape=(ny, nx))
        camp = None
        if st["algo"] == "histeq":
            c = (campione(lav, 400_000, zeri=True) - limiti[0]) / (limiti[1] - limiti[0])
            camp = np.clip(c, 0, 1)
        stretch = crea_stretch(st["algo"], st.get("valore"), camp)
        for y in range(0, ny, 1024):
            y1 = min(y + 1024, ny)
            x = (np.asarray(lav[y:y1], dtype=DTYPE) - limiti[0]) / (limiti[1] - limiti[0])
            nan = ~np.isfinite(x)
            r = np.clip(stretch(np.clip(np.where(nan, 0, x), 0, 1), clip=True), 0, 1).astype(DTYPE)
            r[nan] = np.nan
            uscita[y:y1] = r
            if progresso:
                progresso("stretching", y1, ny)
        uscita.flush()
        del uscita
        p_info.write_text(json.dumps({"limiti": limiti, "scelte": scelte}, default=str))
        self.mem.libera(lav)
        del lav
        self.mem.pulisci_temporanei()
        restituisci_ram()
        return np.load(p_st, mmap_mode="r"), limiti


# =============================================================================
# GRIGLIE E RIPROIEZIONE
# =============================================================================
def e_equatoriale(w):
    """True se il WCS è in AR/Dec (la maggior parte dei FITS); False per le mappe in
    coordinate galattiche o eclittiche (es. Spitzer GLIMPSE/SMOG, Herschel Hi-GAL)."""
    try:
        return not str(w.wcs.ctype[0]).upper().startswith(("GLON", "ELON", "SLON", "HLON"))
    except Exception:
        return True


def pix_a_radec(w, x, y):
    """Pixel (origine 0) -> AR, Dec ICRS in gradi, qualunque sia il sistema del WCS."""
    if e_equatoriale(w):
        return w.wcs_pix2world(x, y, 0)
    from astropy.wcs.utils import pixel_to_skycoord
    c = pixel_to_skycoord(np.asarray(x, dtype=float), np.asarray(y, dtype=float), w, origin=0, mode="wcs").icrs
    return c.ra.deg, c.dec.deg


def radec_a_pix(w, ra, dec, completo=False):
    """AR, Dec ICRS (gradi) -> pixel (origine 0); completo=True usa anche la distorsione SIP."""
    if e_equatoriale(w):
        if completo:
            out = w.all_world2pix(np.column_stack([np.atleast_1d(ra), np.atleast_1d(dec)]), 0)
            return out[:, 0], out[:, 1]
        return w.wcs_world2pix(ra, dec, 0)
    from astropy.coordinates import SkyCoord
    from astropy.wcs.utils import skycoord_to_pixel
    import astropy.units as u
    c = SkyCoord(np.asarray(ra, dtype=float) * u.deg, np.asarray(dec, dtype=float) * u.deg, frame="icrs")
    return skycoord_to_pixel(c, w, origin=0, mode="all" if completo else "wcs")


def impronta_radec(w, nx, ny):
    """I 4 angoli del campo in AR/Dec ICRS (come calc_footprint, ma per ogni sistema)."""
    if e_equatoriale(w):
        return w.calc_footprint(axes=(nx, ny))
    ra, de = pix_a_radec(w, np.array([0, 0, nx - 1, nx - 1]), np.array([0, ny - 1, ny - 1, 0]))
    return np.column_stack([ra, de])


def scala_pixel_deg(w):
    return float(np.mean(np.abs(proj_plane_pixel_scales(w))))


def scala_wcs(w, f):
    """WCS di un'immagine ridotta di un fattore f con la media a blocchi."""
    w2 = w.deepcopy()
    w2.wcs.crpix = (np.array(w.wcs.crpix) - 0.5) / f + 0.5
    if w2.wcs.has_cd():
        w2.wcs.cd = np.array(w.wcs.cd) * f
    else:
        w2.wcs.cdelt = np.array(w.wcs.cdelt) * f
    w2.sip = None
    w2.wcs.set()
    return w2


def stessa_griglia(immagini, tolleranza_pixel=0.05):
    """True se tutte le immagini hanno la stessa forma e lo stesso WCS (entro
    'tolleranza_pixel' pixel su tutti e quattro gli angoli e il centro)."""
    if len(immagini) < 2 or any(i.wcs is None for i in immagini):
        return False
    ny, nx = immagini[0].shape
    if any(tuple(i.shape) != (ny, nx) for i in immagini):
        return False
    xs = np.array([0, nx - 1, nx - 1, 0, (nx - 1) / 2])
    ys = np.array([0, 0, ny - 1, ny - 1, (ny - 1) / 2])
    ra0, de0 = pix_a_radec(immagini[0].wcs, xs, ys)
    for img in immagini[1:]:
        x, y = radec_a_pix(img.wcs, ra0, de0)
        if not (np.all(np.isfinite(x)) and np.max(np.hypot(np.asarray(x) - xs, np.asarray(y) - ys)) <= tolleranza_pixel):
            return False
    return True


def griglia_riferimento(img, riduzione=1):
    w = img.wcs.deepcopy()
    shape = img.shape
    if riduzione > 1:
        w = scala_wcs(w, riduzione)
        shape = (math.ceil(shape[0] / riduzione), math.ceil(shape[1] / riduzione))
    return w, shape


def angolo_ottimale(x, y, passo=0.25):
    """Rotazione (gradi, tra -45 e +45) dei pixel che rende minimo il rettangolo
    che contiene i punti (x, y). Con la rotazione p = R·q, R = [[cos, sin], [-sin, cos]].
    Se il Nord in alto è quasi altrettanto buono (entro lo 0,5%) restituisce 0."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()

    def area(t):
        c, sn = math.cos(math.radians(t)), math.sin(math.radians(t))
        u, v = x * c + y * sn, -x * sn + y * c
        return (u.max() - u.min()) * (v.max() - v.min())

    angoli = np.arange(-45, 45, passo)
    aree = np.array([area(t) for t in angoli])
    migliore = float(angoli[int(np.argmin(aree))])
    if area(0.0) <= aree.min() * 1.005:
        return 0.0
    return migliore


def contorno_dati_radec(img, max_lato=400, max_punti=1500):
    """Punti (AR, Dec) del bordo della zona che contiene DAVVERO dati (non i 4 angoli
    del file): così la griglia del mosaico si orienta sui dati anche quando il campo
    è un parallelogramma o un triangolo dentro un file rettangolare (SPHEREx HEALPix,
    mosaici a scalini). Se i dati riempiono il file restituisce i 4 angoli."""
    if getattr(img, "_contorno_dati", None) is not None:
        return img._contorno_dati
    ny, nx = img.shape
    f = max(1, math.ceil(max(ny, nx) / max_lato))
    m = np.isfinite(img.ridotta_valida(f))
    if not m.any() or m.mean() > 0.97:
        img._contorno_dati = impronta_radec(img.wcs, nx, ny)
        return img._contorno_dati
    bordo = m & ~ndi.binary_erosion(m)
    yy, xx = np.nonzero(bordo)
    if yy.size > max_punti:
        k = np.linspace(0, yy.size - 1, max_punti).astype(int)
        yy, xx = yy[k], xx[k]
    # angoli esterni dei blocchi ridotti, riportati a piena risoluzione
    px = np.concatenate([xx * f - 0.5, (xx + 1) * f - 0.5])
    py = np.concatenate([yy * f - 0.5, (yy + 1) * f - 0.5])
    ra, de = pix_a_radec(img.wcs, np.clip(px, -0.5, nx - 0.5), np.clip(py, -0.5, ny - 0.5))
    img._contorno_dati = np.column_stack([np.ravel(ra), np.ravel(de)])
    return img._contorno_dati


def griglia_mosaico(immagini, riduzione=1, margine=4, nord_in_alto=False):
    """Griglia TAN che contiene tutti i campi, alla risoluzione del file più
    fine (× riduzione). Di base è orientata come le immagini (rettangolo di area
    minima: niente angoli vuoti dovuti alla rotazione); con nord_in_alto=True
    il Nord è esattamente in alto e l'Est a sinistra."""
    angoli = []
    scale = []
    for img in immagini:
        ny, nx = img.shape
        try:
            angoli.append(contorno_dati_radec(img))
        except Exception:
            angoli.append(impronta_radec(img.wcs, nx, ny))
        scale.append(scala_pixel_deg(img.wcs))
    tutti = np.vstack(angoli)
    ra, dec = np.radians(tutti[:, 0]), np.radians(tutti[:, 1])
    v = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]).mean(axis=1)
    ra0 = math.degrees(math.atan2(v[1], v[0])) % 360
    dec0 = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
    s = min(scale) * riduzione
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.cdelt = [-s, s]
    w.wcs.crpix = [1, 1]
    w.wcs.set()
    theta = 0.0
    if not nord_in_alto:
        qx, qy = w.wcs_world2pix(tutti[:, 0], tutti[:, 1], 0)
        theta = angolo_ottimale(qx, qy)
    if theta:
        c, sn = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        R = np.array([[c, sn], [-sn, c]])
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = [ra0, dec0]
        w.wcs.cd = np.diag([-s, s]) @ R.T
        w.wcs.crpix = [1, 1]
        w.wcs.set()
    x, y = w.wcs_world2pix(tutti[:, 0], tutti[:, 1], 0)
    xmin, ymin = math.floor(x.min()) - margine, math.floor(y.min()) - margine
    xmax, ymax = math.ceil(x.max()) + margine, math.ceil(y.max()) + margine
    w.wcs.crpix = [1 - xmin, 1 - ymin]
    w.wcs.set()
    return w, (int(ymax - ymin + 1), int(xmax - xmin + 1))


def ritaglia_wcs(w, y0, x0):
    """WCS di un ritaglio che inizia al pixel (y0, x0): scala e rotazione non
    cambiano, si sposta solo il pixel di riferimento CRPIX."""
    w2 = w.deepcopy()
    w2.wcs.crpix = [w.wcs.crpix[0] - x0, w.wcs.crpix[1] - y0]
    w2.wcs.set()
    return w2


def rettangolo_massimo(maschera):
    """Rettangolo (allineato agli assi) di area massima tutto True nella maschera.
    Restituisce (y0, y1, x0, x1) con y1, x1 esclusi, oppure None.
    Metodo dell'istogramma per righe: tempo proporzionale al numero di pixel."""
    m = np.asarray(maschera, bool)
    H, W = m.shape
    altezze = np.zeros(W, dtype=np.int64)
    migliore, area_max = None, 0
    for y in range(H):
        altezze = np.where(m[y], altezze + 1, 0)
        pila = []                                   # indici di colonne con altezze crescenti
        for x in range(W + 1):
            h = altezze[x] if x < W else 0
            inizio = x
            while pila and pila[-1][1] >= h:
                xi, hi = pila.pop()
                area = hi * (x - xi)
                if area > area_max:
                    area_max, migliore = area, (y - hi + 1, y + 1, xi, x)
                inizio = xi
            pila.append((inizio, h))
    return migliore


def ritaglio_automatico(arrays, max_lato=600):
    """Rettangolo più grande in cui TUTTI gli array hanno dati (niente bordi neri).
    Calcolato su versioni ridotte e poi riportato a piena risoluzione,
    con un margine di sicurezza di un blocco verso l'interno."""
    ny, nx = arrays[0].shape
    f = max(1, math.ceil(max(ny, nx) / max_lato))
    pieno = None
    for a in arrays:
        r = np.isfinite(immagine_ridotta(a, f, zeri_nan=False))
        pieno = r if pieno is None else (pieno & r)
    pieno = ndi.binary_erosion(pieno, iterations=1)
    r = rettangolo_massimo(pieno)
    if r is None:
        return None
    y0, y1, x0, x1 = r
    return [int(y0 * f), int(min(y1 * f, ny)), int(x0 * f), int(min(x1 * f, nx))]


def riquadro_tutti_i_dati(arrays, max_lato=1200):
    """Rettangolo più piccolo che contiene TUTTI i pixel con dati di almeno un filtro:
    toglie solo le righe e le colonne esterne completamente vuote."""
    ny, nx = arrays[0].shape
    f = max(1, math.ceil(max(ny, nx) / max_lato))
    dati = None
    for a in arrays:
        r = np.isfinite(immagine_ridotta(a, f, zeri_nan=False))
        dati = r if dati is None else (dati | r)
    if dati is None or not dati.any():
        return None
    iy = np.where(dati.any(axis=1))[0]
    ix = np.where(dati.any(axis=0))[0]
    return [int(iy[0] * f), int(min((iy[-1] + 1) * f, ny)), int(ix[0] * f), int(min((ix[-1] + 1) * f, nx))]


def riproietta(arr_in, wcs_in, wcs_out, shape_out, dst, righe=1024, progresso=None):
    """Riproietta arr_in sulla griglia di uscita a blocchi di righe. Per ogni
    blocco legge solo la parte di arr_in che serve (memoria contenuta).
    I pixel senza dati restano NaN."""
    from reproject import reproject_interp
    ny, nx = shape_out
    ni, mi = arr_in.shape
    for y0 in range(0, ny, righe):
        y1 = min(y0 + righe, ny)
        dst[y0:y1] = np.nan
        # zona dell'immagine di ingresso che cade in questo blocco
        xs = np.concatenate([np.linspace(0, nx - 1, 32), np.linspace(0, nx - 1, 32),
                             np.zeros(16), np.full(16, nx - 1)])
        ys = np.concatenate([np.full(32, y0), np.full(32, y1 - 1),
                             np.linspace(y0, y1 - 1, 16), np.linspace(y0, y1 - 1, 16)])
        try:
            ra, dec = pix_a_radec(wcs_out, xs, ys)
            xi, yi = radec_a_pix(wcs_in, ra, dec)
        except Exception:
            continue
        ok = np.isfinite(xi) & np.isfinite(yi)
        if not ok.any():
            continue
        mg = 64 if wcs_in.sip is not None else 6       # la distorsione SIP sposta i pixel
        a0 = int(max(math.floor(yi[ok].min()) - mg, 0))
        a1 = int(min(math.ceil(yi[ok].max()) + mg + 1, ni))
        b0 = int(max(math.floor(xi[ok].min()) - mg, 0))
        b1 = int(min(math.ceil(xi[ok].max()) + mg + 1, mi))
        if a1 - a0 < 2 or b1 - b0 < 2:
            if progresso:
                progresso(y1, ny)
            continue
        sub = np.asarray(arr_in[a0:a1, b0:b1], dtype=np.float64)
        if wcs_in.sip is not None:
            w_sub = wcs_in.slice((slice(a0, a1), slice(b0, b1)))
        else:
            w_sub = wcs_in.deepcopy()
            w_sub.wcs.crpix = [w_sub.wcs.crpix[0] - b0, w_sub.wcs.crpix[1] - a0]
            w_sub.wcs.set()
        w_blk = wcs_out.deepcopy()
        w_blk.wcs.crpix = [w_blk.wcs.crpix[0], w_blk.wcs.crpix[1] - y0]
        w_blk.wcs.set()
        out, fp = reproject_interp((sub, w_sub), w_blk, shape_out=(y1 - y0, nx), order="bilinear",
                                   return_footprint=True)
        out = out.astype(DTYPE)
        out[fp <= 0] = np.nan
        dst[y0:y1] = out
        if progresso:
            progresso(y1, ny)
    if hasattr(dst, "flush"):
        dst.flush()
    return dst


def riduci_per_griglia(arr, w, scala_uscita, mem):
    """Se l'uscita ha pixel almeno 2 volte più grandi, riduce prima l'ingresso
    con la media a blocchi (più veloce e senza aliasing)."""
    f = int(scala_uscita / scala_pixel_deg(w))
    if f < 2 or w.sip is not None:
        return arr, w
    rid = immagine_ridotta(arr, f, zeri_nan=False)
    return rid, scala_wcs(w, f)


def anteprima(arr, max_lato=1500):
    f = max(1, math.ceil(max(arr.shape[:2]) / max_lato))
    if arr.ndim == 3:
        return np.stack([immagine_ridotta(arr[..., c], f, zeri_nan=False) for c in range(arr.shape[2])], -1)
    return immagine_ridotta(arr, f, zeri_nan=False)


# =============================================================================
# STACK E RGB
# =============================================================================
def livelli_uniformi(arrays, passo=8):
    """Per «uniforma la luminosità»: (a_i, b_i) tali che a_i·x + b_i porta il fondo
    (mediana) e le parti brillanti (99,5° percentile) di ogni immagine ai valori medi
    di tutte. Misurati su una riga ogni 'passo' (veloce anche con immagini enormi)."""
    stat = []
    for a in arrays:
        v = np.asarray(a[::passo, ::passo], dtype=np.float64)
        v = v[np.isfinite(v)]
        if v.size < 100:
            stat.append(None)
            continue
        stat.append((float(np.median(v)), float(np.percentile(v, 99.5))))
    validi = [x for x in stat if x is not None and x[1] > x[0]]
    if not validi:
        return [(1.0, 0.0)] * len(arrays)
    m_t = float(np.mean([x[0] for x in validi]))
    p_t = float(np.mean([x[1] for x in validi]))
    out = []
    for x in stat:
        if x is None or x[1] <= x[0]:
            out.append((1.0, 0.0))
        else:
            a = (p_t - m_t) / (x[1] - x[0])
            out.append((a, m_t - a * x[0]))
    return out


def combina_bn(riproiettate, pesi, metodo, dst, progresso=None, uniforma=False):
    """riproiettate: lista di array (stessa griglia); metodo: media/mediana/massimo.
    uniforma=True: prima porta tutte le immagini allo stesso fondo e alla stessa
    luminosità delle parti brillanti (livelli_uniformi)."""
    ny, nx = dst.shape
    livelli = livelli_uniformi(riproiettate) if uniforma else [(1.0, 0.0)] * len(riproiettate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for y0 in range(0, ny, 512):
            y1 = min(y0 + 512, ny)
            blocco = np.stack([np.asarray(a[y0:y1], dtype=DTYPE) * DTYPE(k) + DTYPE(q)
                               for a, (k, q) in zip(riproiettate, livelli)])
            if metodo == "mediana":
                r = np.nanmedian(blocco, axis=0)
            elif metodo == "massimo":
                r = np.nanmax(blocco, axis=0)
            else:
                p = np.asarray(pesi, dtype=DTYPE)[:, None, None] * np.isfinite(blocco)
                s = np.nansum(np.nan_to_num(blocco) * p, axis=0)
                ps = p.sum(axis=0)
                r = np.where(ps > 0, s / np.where(ps > 0, ps, 1), np.nan)
            dst[y0:y1] = r.astype(DTYPE)
            if progresso:
                progresso(y1, ny)
    if hasattr(dst, "flush"):
        dst.flush()
    return dst


def pesi_cromatici(lunghezze):
    """Ordine cromatico: i filtri più corti verso il blu, i più lunghi verso il rosso."""
    n_tot = len(lunghezze)
    if n_tot == 0:
        return []
    pesi = [(0.5, 0.5, 0.5)] * n_tot                  # lunghezza d'onda ignota: peso neutro
    ordine = sorted((i for i in range(n_tot) if lunghezze[i] is not None), key=lambda i: lunghezze[i])
    n = len(ordine)
    if n == 1:
        pesi[ordine[0]] = (1.0, 1.0, 1.0)
        return pesi
    for pos, i in enumerate(ordine):
        if n == 2:
            pesi[i] = (0.0, 0.5, 1.0) if pos == 0 else (1.0, 0.5, 0.0)
            continue
        t = pos / (n - 1)                       # 0 = più corto, 1 = più lungo
        r, g, b = colorsys.hsv_to_rgb((1 - t) * 240 / 360, 1.0, 1.0)
        pesi[i] = (round(r, 2), round(g, 2), round(b, 2))
    return pesi


def combina_rgb(arrays, pesi, scala_canali=(1.0, 1.0, 1.0)):
    """Canale c = Σ peso_ic·img_i / Σ peso_ic (solo dove ci sono dati), × scala."""
    H, W = arrays[0].shape
    out = np.zeros((H, W, 3), dtype=DTYPE)
    for c in range(3):
        num = np.zeros((H, W), dtype=DTYPE)
        den = np.zeros((H, W), dtype=DTYPE)
        for a, p in zip(arrays, pesi):
            w = float(p[c])
            if w <= 0:
                continue
            a = np.asarray(a, dtype=DTYPE)
            ok = np.isfinite(a)
            num += np.where(ok, a, 0) * w
            den += ok * w
        with np.errstate(invalid="ignore", divide="ignore"):
            out[..., c] = np.where(den > 0, num / np.where(den > 0, den, 1), 0) * scala_canali[c]
    return np.clip(out, 0, 1)


def bilanciamento(rgb_anteprima):
    """Fattori per portare il 99.7° percentile di ogni canale a 1."""
    f = []
    for c in range(3):
        v = rgb_anteprima[..., c]
        v = v[v > 0]
        p = np.percentile(v, 99.7) if v.size else 1.0
        f.append(1.0 / p if p > 0 else 1.0)
    return tuple(f)


# =============================================================================
# SALVATAGGIO
# =============================================================================
FORMATI_IMMAGINE = {"TIFF 16 bit": ".tif", "PNG 8 bit": ".png", "JPEG": ".jpg", "JPEG per forum": "_forum.jpg"}
# JPEG per forum: lato più lungo e peso massimi (modificabili da Opzioni → JPEG per forum)
FORUM = {"lato": 2048, "kb": 1000}


def _jpeg_forum(img, percorso, lato=None, kb=None):
    """Salva un'immagine PIL come JPEG entro il lato e il peso massimi: rimpicciolisce
    (Lanczos) e cerca la qualità più alta che sta nel limite; se nemmeno con qualità 50
    ci sta, rimpicciolisce ancora."""
    import io
    from PIL import Image
    lato = int(lato or FORUM["lato"])
    limite = int(kb or FORUM["kb"]) * 1024
    img = img.convert("RGB") if img.mode not in ("L", "RGB") else img
    if max(img.size) > lato:
        f = lato / max(img.size)
        img = img.resize((max(1, round(img.size[0] * f)), max(1, round(img.size[1] * f))), Image.LANCZOS)
    while True:
        migliore = None
        basso, alto = 50, 95
        while basso <= alto:                     # ricerca della qualità più alta che sta nel limite
            q = (basso + alto) // 2
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True, progressive=True)
            if buf.tell() <= limite:
                migliore, basso = buf, q + 1
            else:
                alto = q - 1
        if migliore is not None or min(img.size) < 64:
            if migliore is None:
                migliore = buf
            break
        img = img.resize((max(1, round(img.size[0] * 0.8)), max(1, round(img.size[1] * 0.8))), Image.LANCZOS)
    Path(percorso).write_bytes(migliore.getvalue())
    return Path(percorso)


def percorso_formato(base, formato):
    """Nome del file per un formato: stack_bn + .tif / .png / .jpg / _forum.jpg"""
    return Path(str(base) + FORMATI_IMMAGINE.get(formato, ".png"))


def _font_firma(px):
    """Carattere per la firma, alto circa 'px' pixel."""
    from PIL import ImageFont
    for nome in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(nome, px)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


def applica_firma(a, testo=FIRMA, opacita=0.6):
    """Scrive la firma (filigrana) nell'angolo in basso a destra di un'immagine
    float [0,1] già orientata per la visualizzazione (2D o HxWx3). Testo bianco
    semitrasparente con un'ombra scura, così si legge su fondo chiaro e scuro.
    Modifica solo l'angolo: anche con immagini enormi non usa memoria extra."""
    from PIL import Image, ImageDraw
    ny, nx = a.shape[:2]
    px = int(max(12, min(200, round(min(ny, nx) * 0.022))))
    font = _font_firma(px)
    prova = ImageDraw.Draw(Image.new("L", (1, 1)))
    x0, y0, x1, y1 = prova.textbbox((0, 0), testo, font=font)
    ombra = max(1, px // 12)
    tw, th = x1 - x0 + 2 * ombra + 2, y1 - y0 + 2 * ombra + 2
    margine = max(4, px // 2)
    if tw + margine >= nx or th + margine >= ny:
        return a
    m_testo = Image.new("L", (tw, th), 0)
    m_ombra = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(m_ombra).text((1 + ombra - x0 + ombra, 1 + ombra - y0 + ombra), testo, font=font, fill=255)
    ImageDraw.Draw(m_testo).text((1 + ombra - x0, 1 + ombra - y0), testo, font=font, fill=255)
    mt = np.asarray(m_testo, dtype=np.float32) / 255 * opacita
    mo = np.asarray(m_ombra, dtype=np.float32) / 255 * opacita * 0.7
    ys, xs = ny - margine - th, nx - margine - tw
    zona = a[ys:ys + th, xs:xs + tw]
    if zona.ndim == 3:
        mt, mo = mt[..., None], mo[..., None]
    zona[...] = zona * (1 - mo)               # ombra scura
    zona[...] = zona * (1 - mt) + mt          # testo bianco
    return a


def salva_immagine(arr, percorso, formato, firma=False):
    """arr in [0,1], 2D (grigi) o HxWx3 (RGB). L'immagine viene capovolta in
    verticale perché i FITS hanno l'origine in basso (così il Nord resta in alto).
    Con firma=True aggiunge «© Edoardo Battistel» in basso a destra."""
    from PIL import Image
    a = np.flipud(np.nan_to_num(np.asarray(arr, dtype=DTYPE), nan=0.0))
    a = np.clip(a, 0, 1)
    grande = formato == "JPEG per forum" and max(a.shape[:2]) > FORUM["lato"]
    if firma and not grande:
        a = np.ascontiguousarray(a)
        applica_firma(a)
    percorso = Path(percorso)
    if formato == "JPEG per forum":
        from PIL import Image as _I
        a8 = (a * 255 + 0.5).astype(np.uint8)
        img = _I.fromarray(a8, mode="L" if a.ndim == 2 else "RGB")
        lato = FORUM["lato"]
        if max(img.size) > lato:             # firma rifatta sull'immagine già rimpicciolita
            f = lato / max(img.size)
            img = img.resize((max(1, round(img.size[0] * f)), max(1, round(img.size[1] * f))), _I.LANCZOS)
            if firma:
                piccola = np.asarray(img, dtype=DTYPE) / 255
                piccola = np.ascontiguousarray(piccola)
                img = _I.fromarray((np.clip(applica_firma(piccola), 0, 1) * 255 + 0.5).astype(np.uint8),
                                   mode="L" if piccola.ndim == 2 else "RGB")
        return _jpeg_forum(img, percorso)
    if formato == "TIFF 16 bit":
        a16 = (a * 65535 + 0.5).astype(np.uint16)
        if a.ndim == 2:
            Image.fromarray(a16).save(percorso, format="TIFF")
        else:
            import tifffile
            tifffile.imwrite(percorso, a16, photometric="rgb")
    else:
        a8 = (a * 255 + 0.5).astype(np.uint8)
        img = Image.fromarray(a8, mode="L" if a.ndim == 2 else "RGB")
        if formato == "JPEG":
            img.save(percorso, format="JPEG", quality=95)
        else:
            img.save(percorso, format="PNG")
    return percorso


def salva_fits(piani, wcs, percorso, storia=(), extra=None):
    """Scrive un FITS con il WCS. 'piani' = un array 2D oppure una lista di 3
    array (cubo RGB); i dati vengono scritti a blocchi (StreamingHDU)."""
    piani = [piani] if not isinstance(piani, (list, tuple)) else list(piani)
    ny, nx = piani[0].shape
    h = fits.Header()
    h["SIMPLE"] = True
    h["BITPIX"] = -32
    if len(piani) == 1:
        h["NAXIS"] = 2
        h["NAXIS1"], h["NAXIS2"] = nx, ny
    else:
        h["NAXIS"] = 3
        h["NAXIS1"], h["NAXIS2"], h["NAXIS3"] = nx, ny, len(piani)
    h.extend(wcs.to_header(), update=True)
    h["CREATOR"] = (f"{NOME_PROGRAMMA} {VERSIONE}", "programma che ha creato il file")
    h["AUTHOR"] = (AUTORE, "autore del programma")
    h["DATE"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    h.add_comment(f"{COPYRIGHT_ASCII} - creato con {NOME_PROGRAMMA} ({SIGLA}) {VERSIONE}")
    for k, v in (extra or {}).items():
        h[k] = _ascii(v) if isinstance(v, str) else v
    for s in storia:
        h.add_history(_ascii(s)[:70])
    percorso = Path(percorso)
    if percorso.exists():
        percorso.unlink()
    shdu = fits.StreamingHDU(percorso, h)
    for p in piani:
        for y in range(0, ny, 1024):
            shdu.write(np.asarray(p[y:min(y + 1024, ny)], dtype=">f4"))
    shdu.close()
    return percorso


# =============================================================================
# ANALISI SCIENTIFICHE
# =============================================================================
def carica_stack(percorso, max_lato=4000):
    """Legge lo stack B/N (riducendolo se enorme) e il suo WCS."""
    with fits.open(percorso, memmap=True) as hdul:
        i = _hdu_scientifica(hdul)
        h = hdul[i].header.copy()
        d = hdul[i].data
        if d.ndim == 3:
            d = d[0]
        w = WCS(h, naxis=2).celestial
        f = max(1, math.ceil(max(d.shape) / max_lato))
        dati = immagine_ridotta(d, f, zeri_nan=False) if f > 1 else np.array(d, dtype=DTYPE)
    if f > 1:
        w = scala_wcs(w, f)
    return dati, w, f


def _fig(dimensioni=(10, 10)):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=dimensioni, dpi=110, layout="constrained")
    FigureCanvasAgg(fig)
    return fig


def _norm_visuale(d):
    from astropy.visualization import ImageNormalize
    vmin, vmax = limiti_zscale(d)
    return ImageNormalize(vmin=vmin, vmax=vmax)


def _adatta_figura(fig, shape, lato=10.0):
    """Proporzioni della figura come quelle dell'immagine (più spazio per titolo,
    etichette e barra dei colori): niente grandi bordi bianchi con le immagini
    molto larghe (Herschel) o molto alte."""
    ny, nx = shape[:2]
    r = ny / max(nx, 1)
    if r < 0.8:
        fig.set_size_inches(lato * 1.15, min(max(lato * r + 2.6, 4.0), lato))
    elif r > 1.25:
        fig.set_size_inches(min(max(lato / r + 2.6, 5.0), lato), lato)


def _assi_wcs(fig, w, d, titolo, cmap="gray", norm=None, griglia=False):
    _adatta_figura(fig, d.shape)
    ax = fig.add_subplot(1, 1, 1, projection=w)
    im = ax.imshow(np.nan_to_num(d, nan=0), origin="lower", cmap=cmap,
                   norm=norm if norm is not None else _norm_visuale(d), interpolation="nearest")
    ra, de = ax.coords[0], ax.coords[1]
    ra.set_axislabel("Ascensione retta (J2000)")
    de.set_axislabel("Declinazione (J2000)")
    ra.set_major_formatter("hh:mm:ss")
    de.set_major_formatter("dd:mm:ss")
    ra.set_ticklabel(exclude_overlapping=True)
    de.set_ticklabel(exclude_overlapping=True)
    ra.set_ticks(number=5)
    de.set_ticks(number=5)
    if griglia:
        ax.coords.grid(color="white", alpha=0.25, linestyle=":", linewidth=0.8)
    # titolo sopra la figura (non sull'asse): con le immagini ruotate o molto larghe
    # le etichette delle coordinate possono stare anche in alto e si sovrapponevano
    fig.suptitle(titolo, fontsize=13, fontweight="bold")
    return ax, im


def analisi_griglia(d, w):
    fig = _fig()
    _assi_wcs(fig, w, d, "Immagine con coordinate celesti", griglia=True)
    return fig, None


def analisi_snr(d, w):
    _, med, std = sigma_clipped_stats(d[np.isfinite(d)], sigma=3.0)
    snr = np.clip((d - med) / (std if std > 0 else 1), 0, None)
    from astropy.visualization import ImageNormalize
    fig = _fig()
    ax, im = _assi_wcs(fig, w, snr, "Rapporto segnale/rumore (σ sopra il fondo)", cmap="viridis",
                       norm=ImageNormalize(vmin=0, vmax=float(np.nanpercentile(snr, 99.5)) or 50))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Numero di σ sopra il rumore")
    return fig, None


def analisi_gradiente(d, w):
    from skimage.filters import sobel
    vmin, vmax = limiti_zscale(d)
    n = np.clip((np.nan_to_num(d, nan=vmin) - vmin) / (vmax - vmin), 0, 1)
    e = sobel(n)
    from astropy.visualization import ImageNormalize
    fig = _fig()
    ax, im = _assi_wcs(fig, w, e, "Gradiente morfologico (fronti d'urto, Sobel)", cmap="inferno",
                       norm=ImageNormalize(vmin=float(np.percentile(e, 1)), vmax=float(np.percentile(e, 99.5))))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Intensità del gradiente")
    return fig, None


def analisi_strutture(d, w):
    x = np.nan_to_num(d, nan=float(np.nanmedian(d)))
    dog = ndi.gaussian_filter(x, 1.5) - ndi.gaussian_filter(x, 8)
    lim = float(np.percentile(np.abs(dog), 99.5)) or 1
    from astropy.visualization import ImageNormalize
    fig = _fig()
    ax, im = _assi_wcs(fig, w, dog, "Strutture fini: filamenti e bordi (differenza di gaussiane)",
                       cmap="RdGy_r", norm=ImageNormalize(vmin=-lim, vmax=lim))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Eccesso locale di luminosità")
    return fig, None


def analisi_isofote(d, w):
    fig = _fig()
    ax, _ = _assi_wcs(fig, w, d, "Isofote (curve di uguale luminosità)")
    liscia = ndi.gaussian_filter(np.nan_to_num(d, nan=float(np.nanmedian(d))), 3)
    livelli = np.unique(np.percentile(liscia, [50, 70, 85, 93, 97, 99]))
    cs = ax.contour(liscia, levels=livelli, cmap="plasma", linewidths=0.8, alpha=0.9)
    ax.clabel(cs, fmt="%.2f", fontsize=7)
    return fig, None


def analisi_sorgenti(d, w):
    from photutils.segmentation import detect_sources, SourceCatalog
    import inspect
    validi = np.isfinite(d)
    _, med, std = sigma_clipped_stats(d[validi], sigma=3.0)
    x = np.where(validi, d, med) - med
    liscia = ndi.gaussian_filter(x, 1.0)
    arg = "n_pixels" if "n_pixels" in inspect.signature(detect_sources).parameters else "npixels"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        segm = detect_sources(liscia, 3 * std, mask=~validi, **{arg: 8})
    fig = _fig()
    ax, _ = _assi_wcs(fig, w, d, "Sorgenti rilevate (> 3σ)")
    righe = []
    if segm is not None:
        cat = SourceCatalog(x, segm)
        tab = cat.to_table(columns=["label", "xcentroid", "ycentroid", "area", "segment_flux",
                                    "fwhm", "ellipticity"])
        ordine = np.argsort(np.nan_to_num(np.array(tab["segment_flux"], dtype=float)))[::-1]
        mostra = ordine[:2500]
        from matplotlib.patches import Circle
        for i in mostra:
            r = max(2.0, math.sqrt(float(tab["area"][i].value if hasattr(tab["area"][i], "value")
                                         else tab["area"][i]) / math.pi))
            ax.add_patch(Circle((float(tab["xcentroid"][i]), float(tab["ycentroid"][i])), r,
                                fill=False, edgecolor="#39ff14", linewidth=0.5, alpha=0.8))
        cielo = w.pixel_to_world(np.array(tab["xcentroid"], dtype=float), np.array(tab["ycentroid"], dtype=float)).icrs
        for k in ordine:
            def num(v):
                v = v.value if hasattr(v, "value") else v
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float("nan")
            righe.append({"id": int(tab["label"][k]), "x_pixel": f"{num(tab['xcentroid'][k]):.2f}",
                          "y_pixel": f"{num(tab['ycentroid'][k]):.2f}",
                          "ra_deg": f"{cielo[k].ra.deg:.6f}", "dec_deg": f"{cielo[k].dec.deg:.6f}",
                          "area_pixel": f"{num(tab['area'][k]):.0f}",
                          "flusso_relativo": f"{num(tab['segment_flux'][k]):.4g}",
                          "fwhm_pixel": f"{num(tab['fwhm'][k]):.2f}",
                          "ellitticita": f"{num(tab['ellipticity'][k]):.3f}"})
        fig.suptitle(f"Sorgenti rilevate (> 3σ): {len(tab)}")
    return fig, righe


def analisi_istogramma(d, w):
    v = d[np.isfinite(d)].ravel()
    fig = _fig((10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(v, bins=256, color="#4a90d9", log=True)
    _, med, std = sigma_clipped_stats(v, sigma=3.0)
    ax.axvline(med, color="k", linestyle="--", label=f"fondo (mediana) = {med:.3f}")
    ax.axvline(med + 3 * std, color="r", linestyle=":", label=f"fondo + 3σ = {med + 3 * std:.3f}")
    ax.set_xlabel("Intensità (dopo lo stretching)")
    ax.set_ylabel("Numero di pixel (scala log)")
    ax.set_title("Istogramma delle intensità")
    ax.legend()
    return fig, None


def analisi_profilo(d, w):
    ny, nx = d.shape
    yc, xc = (ny - 1) / 2, (nx - 1) / 2
    Y, X = np.indices(d.shape)
    r = np.hypot(X - xc, Y - yc)
    ok = np.isfinite(d)
    rmax = int(min(xc, yc))
    passo = max(1, rmax // 80)
    bins = np.arange(0, rmax + passo, passo)
    idx = np.digitize(r[ok], bins)
    vals = d[ok]
    medie, dev, raggi = [], [], []
    scala = scala_pixel_deg(w) * 3600
    for i in range(1, len(bins)):
        s = vals[idx == i]
        if s.size:
            medie.append(s.mean())
            dev.append(s.std())
            raggi.append((bins[i - 1] + passo / 2) * scala)
    medie, dev, raggi = map(np.array, (medie, dev, raggi))
    fig = _fig((10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(raggi, medie, color="#d9534f")
    ax.fill_between(raggi, medie - dev, medie + dev, color="#d9534f", alpha=0.2, label="± 1 deviazione standard")
    ax.set_xlabel("Distanza dal centro dell'immagine (secondi d'arco)")
    ax.set_ylabel("Intensità media")
    ax.set_title("Profilo radiale dal centro del campo")
    ax.legend()
    righe = [{"raggio_arcsec": f"{a:.2f}", "intensita_media": f"{b:.5f}", "deviazione": f"{c:.5f}"}
             for a, b, c in zip(raggi, medie, dev)]
    return fig, righe


def _con_timeout(funzione, timeout):
    """Aggiunge un tempo massimo di attesa a ogni richiesta HTTP."""
    def avvolta(*a, **k):
        k.setdefault("timeout", (15, timeout))      # connessione, risposta
        return funzione(*a, **k)
    return avvolta


URL_NED_TAP = "https://ned.ipac.caltech.edu/tap"
SIMBAD_USATO = ""
NED_COLONNE = []
TIPI_NED = {"G": "galassia", "GPair": "coppia di galassie", "GTrpl": "tripletta di galassie", "GGroup": "gruppo di galassie",
            "GClstr": "ammasso di galassie", "QSO": "quasar", "QGroup": "gruppo di quasar", "G_Lens": "lente gravitazionale",
            "AbLS": "sistema di assorbimento", "EmLS": "sorgente a righe di emissione", "*": "stella", "**": "stella doppia",
            "*Cl": "ammasso stellare", "HII": "regione HII", "PN": "nebulosa planetaria", "SNR": "resto di supernova",
            "SN": "supernova", "Neb": "nebulosa", "RfN": "nebulosa a riflessione", "EmObj": "oggetto a emissione",
            "RadioS": "sorgente radio", "IrS": "sorgente infrarossa", "UvS": "sorgente ultravioletta",
            "VisS": "sorgente visibile", "XrayS": "sorgente X", "GammaS": "sorgente gamma", "MCld": "nube molecolare",
            "DNe": "nebulosa oscura", "Other": "altro", "Nova": "nova", "WD*": "nana bianca", "C*": "stella al carbonio"}


def _campo(w, shape):
    """Centro (ra, dec) e raggio (gradi) del cerchio che contiene tutto il campo."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ny, nx = shape
    ang = impronta_radec(w, nx, ny)
    ra0, de0 = (float(np.ravel(v)[0]) for v in pix_a_radec(w, (nx - 1) / 2, (ny - 1) / 2))
    centro = SkyCoord(ra0 * u.deg, de0 * u.deg)
    raggio = max(SkyCoord(ang[:, 0] * u.deg, ang[:, 1] * u.deg).separation(centro).deg) * 1.02
    return ra0, de0, raggio


def _tap_righe(url, query, sessione, maxrec=3000):
    """Esegue una domanda ADQL su un servizio TAP: prima con pyvo, poi con una
    semplice richiesta GET in formato CSV (alcuni server accettano solo quella).
    Restituisce una lista di dizionari; in caso di errore il messaggio del server."""
    errori = []
    try:
        import pyvo
        tab = pyvo.dal.TAPService(url, session=sessione).run_sync(query, maxrec=maxrec).to_table()
        return [{k: r[k] for k in tab.colnames} for r in tab]
    except Exception as e:
        errori.append(f"pyvo {type(e).__name__}: {' '.join(str(e).split())[:160]}")
    try:
        risp = sessione.get(url.rstrip("/") + "/sync", params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                                                                "MAXREC": str(maxrec), "QUERY": query})
        testo = risp.text
        if risp.status_code != 200 or "QUERY_STATUS" in testo[:2000] or "<VOTABLE" in testo[:500].upper():
            raise RuntimeError(f"HTTP {risp.status_code}: {' '.join(testo.split())[:200]}")
        righe = list(csv.DictReader(testo.splitlines()))
        return [{k.strip().strip('"'): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k} for r in righe]
    except Exception as e:
        errori.append(f"GET {type(e).__name__}: {' '.join(str(e).split())[:200]}")
    raise ConnectionError(" | ".join(errori))


def _vuoto(v):
    if v is None:
        return True
    try:
        if np.ma.is_masked(v):
            return True
    except Exception:
        pass
    s = str(v).strip()
    return s in ("", "--", "nan", "None", "null")


def oggetti_ned(w, shape, timeout=60):
    """Oggetti NED nel campo (galassie, quasar, ammassi... con redshift), ordinati
    per numero di riferimenti incrociati (= quanto sono studiati). NED è l'archivio
    NASA degli oggetti extragalattici: ideale per i campi profondi."""
    import requests
    ny, nx = shape
    ra0, de0, raggio = _campo(w, shape)
    # tutte le colonne, senza ordinamento: i nomi delle colonne di NED sono cambiati
    # nel tempo (pretype, type, ...), quindi si scelgono dopo, tra quelle presenti
    q = ("SELECT TOP 3000 * FROM objdir "
         f"WHERE CONTAINS(POINT('J2000', ra, dec), CIRCLE('J2000', {ra0:.6f}, {de0:.6f}, {raggio:.5f})) = 1")
    sessione = requests.Session()
    sessione.headers["User-Agent"] = f"{NOME_PROGRAMMA}/{VERSIONE} (Python)"
    sessione.request = _con_timeout(sessione.request, timeout)
    try:
        righe = _tap_righe(URL_NED_TAP, q, sessione)
    except ConnectionError as e:
        raise ConnectionError(f"NED non raggiungibile ({e})")
    def colonna(riga, *nomi):
        chiavi = {k.lower(): k for k in riga}
        for n in nomi:
            if n in chiavi:
                return chiavi[n]
        return None

    global NED_COLONNE
    NED_COLONNE = list(righe[0].keys()) if righe else []
    oggetti = []
    for r in righe:
        c_ra, c_de = colonna(r, "ra", "ra_deg", "raj2000"), colonna(r, "dec", "dec_deg", "dej2000")
        c_nome = colonna(r, "prefname", "objname", "name", "main_id")
        c_tipo = colonna(r, "prefphytype", "pretype", "type", "objtype", "obj_type", "otype")
        c_z = colonna(r, "z", "redshift")
        c_ref = colonna(r, "n_crosref", "n_refs", "nrefs", "n_ref", "refs")
        try:
            ra, de = float(r[c_ra]), float(r[c_de])
        except (TypeError, ValueError, KeyError):
            continue
        x, y = (float(np.ravel(v)[0]) for v in radec_a_pix(w, ra, de))
        if not (0 <= x < nx and 0 <= y < ny):
            continue
        nome = re.sub(r"\s+", " ", str(r.get(c_nome, "") if c_nome else "").strip()) or f"NED {ra:.5f} {de:+.5f}"
        tipo_c = str(r.get(c_tipo, "") if c_tipo else "").strip()
        try:
            z = None if not c_z or _vuoto(r.get(c_z)) else float(r[c_z])
        except (TypeError, ValueError):
            z = None
        try:
            nref = int(float(r.get(c_ref) or 0)) if c_ref and not _vuoto(r.get(c_ref)) else 0
        except (TypeError, ValueError):
            nref = 0
        oggetti.append(dict(nome=nome, nome_simbad=nome, altri_nomi="", ra=ra, dec=de, x=float(x), y=float(y),
                            tipo=TIPI_NED.get(tipo_c, tipo_c), riferimenti=nref,
                            catalogo=bool(re.match(r"^(M|NGC|IC|UGC|ESO|PGC|MCG|Arp|Abell|ACO|HCG)\b", nome)),
                            magnitudine=None, fonte="NED", z=z))
    oggetti.sort(key=lambda o: (not o["catalogo"], -o["riferimenti"]))
    return oggetti


def unisci_oggetti(principali, aggiunti, distanza_arcsec=20, prendi_z=False):
    """Aggiunge gli oggetti di 'aggiunti' che non sono già (entro la distanza data)
    tra i 'principali': es. SIMBAD + NED, oppure NED + i Messier/NGC/IC del
    catalogo incluso. Se un oggetto c'è in entrambe le liste resta quello dei
    'principali'; con prendi_z=True riceve il redshift dell'altro se non lo ha."""
    out = [dict(o) for o in principali]
    if out and aggiunti:
        ra_p = np.radians([o["ra"] for o in out]); de_p = np.radians([o["dec"] for o in out])
        ra_a = np.radians([o["ra"] for o in aggiunti]); de_a = np.radians([o["dec"] for o in aggiunti])
        soglia = math.radians(distanza_arcsec / 3600.0)
        ordine = np.argsort(de_p)
        de_ord = de_p[ordine]
        for j, o in enumerate(aggiunti):
            # solo i principali in una fascia di declinazione vicina (veloce anche con migliaia di oggetti)
            i0, i1 = np.searchsorted(de_ord, [de_a[j] - soglia, de_a[j] + soglia])
            cand = ordine[i0:i1]
            if cand.size:
                h = (np.sin((de_p[cand] - de_a[j]) / 2) ** 2
                     + np.cos(de_a[j]) * np.cos(de_p[cand]) * np.sin((ra_p[cand] - ra_a[j]) / 2) ** 2)
                sep = 2 * np.arcsin(np.sqrt(np.minimum(1.0, h)))
                k = int(np.argmin(sep))
                if sep[k] <= soglia:
                    p = out[int(cand[k])]
                    if prendi_z and p.get("z") is None and o.get("z") is not None:
                        p["z"] = o["z"]
                        p["fonte"] = p.get("fonte", "SIMBAD") + " (z da NED)"
                    continue
            out.append(o)
    elif aggiunti:
        out = list(aggiunti)
    out.sort(key=lambda o: (not o["catalogo"], -o.get("riferimenti", 0)))
    return out


def oggetti_simbad(w, shape, timeout=60):
    """Oggetti SIMBAD nel campo. Restituisce una lista di dizionari ordinata per
    numero di riferimenti bibliografici (= quanto è famoso l'oggetto)."""
    ny, nx = shape
    ra0, de0, raggio = _campo(w, shape)
    q1 = ("SELECT TOP 3000 b.oid, b.main_id, b.ra, b.dec, b.otype, b.nbref FROM basic AS b "
          f"WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {ra0:.6f}, {de0:.6f}, {raggio:.5f})) = 1 "
          "AND b.ra IS NOT NULL ORDER BY b.nbref DESC")
    q2 = ("SELECT i.id, b.oid FROM ident AS i JOIN basic AS b ON i.oidref = b.oid "
          f"WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {ra0:.6f}, {de0:.6f}, {raggio:.5f})) = 1 "
          "AND (i.id LIKE 'M %' OR i.id LIKE 'NGC %' OR i.id LIKE 'IC %' OR i.id LIKE 'NAME %')")
    # forma semplice (senza alias né ordinamento): alcuni server, come il mirror
    # di Harvard, rifiutano la forma completa
    q1b = ("SELECT TOP 3000 oid, main_id, ra, dec, otype, nbref FROM basic "
           f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra0:.6f}, {de0:.6f}, {raggio:.5f})) = 1")
    import requests
    sessione = requests.Session()
    sessione.headers["User-Agent"] = f"{NOME_PROGRAMMA}/{VERSIONE} (Python)"
    sessione.request = _con_timeout(sessione.request, timeout)
    errori = []
    t1 = None
    global SIMBAD_USATO
    for url in URL_SIMBAD_TAP:
        for forma, q in (("domanda completa", q1), ("domanda semplice", q1b)):
            try:
                t1 = _tap_righe(url, q, sessione)
                SIMBAD_USATO = f"{url.split('/')[2]}, {forma}"
                break
            except Exception as e:
                errori.append(f"{url.split('/')[2]}: {e}")
        if t1 is not None:
            break
    if t1 is None:
        raise ConnectionError("SIMBAD non raggiungibile (" + "; ".join(errori) +
                              "). Controlla la connessione a Internet e riprova più tardi.")
    t1 = sorted(t1, key=lambda r: -float(r.get("nbref") or 0) if not _vuoto(r.get("nbref")) else 0)
    try:
        t2 = _tap_righe(url, q2, sessione, 5000)
    except Exception:
        t2 = []
    nomi = {}
    for r in t2:
        nomi.setdefault(int(r["oid"]), []).append(str(r["id"]).strip())
    oggetti = []
    for r in t1:
        try:
            ra, de = float(r["ra"]), float(r["dec"])
        except (TypeError, ValueError):
            continue
        x, y = (float(np.ravel(v)[0]) for v in radec_a_pix(w, ra, de))
        if not (0 <= x < nx and 0 <= y < ny):
            continue
        principale = str(r["main_id"]).strip()
        altri = sorted(set(nomi.get(int(r["oid"]), [])))
        def priorita(n):
            for k, p in (("M ", 0), ("NGC ", 1), ("IC ", 2), ("NAME ", 3)):
                if n.startswith(k):
                    return p
            return 9
        cat = sorted(altri, key=priorita)
        etichetta = cat[0] if cat else principale
        etichetta = re.sub(r"^NAME\s+", "", re.sub(r"\s+", " ", etichetta))
        try:
            nb = int(r["nbref"])
        except (TypeError, ValueError):
            nb = 0
        oggetti.append(dict(nome=etichetta, nome_simbad=re.sub(r"^NAME\s+", "", principale),
                            altri_nomi="; ".join(a for a in altri if a != principale),
                            ra=ra, dec=de, x=float(x), y=float(y), tipo=str(r["otype"]).strip(),
                            riferimenti=nb, catalogo=bool(cat), magnitudine=None, fonte="SIMBAD"))
    oggetti.sort(key=lambda o: (not o["catalogo"], -o["riferimenti"]))
    return oggetti


FILE_OPENNGC = "openngc_compatto.csv"
TIPI_OPENNGC = {"G": "galassia", "OCl": "ammasso aperto", "GCl": "ammasso globulare",
                "PN": "nebulosa planetaria", "HII": "regione HII", "Cl+N": "ammasso con nebulosa",
                "Neb": "nebulosa", "RfN": "nebulosa a riflessione", "EmN": "nebulosa a emissione",
                "SNR": "resto di supernova", "DrkN": "nebulosa oscura", "*": "stella", "**": "stella doppia",
                "*Ass": "associazione stellare", "GPair": "coppia di galassie", "GTrpl": "tripletta di galassie",
                "GGroup": "gruppo di galassie", "Nova": "nova", "Other": "altro"}
_CATALOGO_NGC = None


def separazione_deg(ra1, dec1, ra2, dec2):
    """Distanza angolare in gradi tra due punti del cielo (formula dell'aversina)."""
    a1, d1, a2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    h = math.sin((d2 - d1) / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin((a2 - a1) / 2) ** 2
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(h))))


def percorso_dati(nome):
    """File di dati accanto al programma (o dentro l'eseguibile PyInstaller)."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / nome


def _sessagesimale_deg(testo, ore):
    t = testo.strip()
    segno = -1 if t.startswith("-") else 1
    parti = [float(x) for x in t.lstrip("+-").split(":")]
    v = parti[0] + (parti[1] if len(parti) > 1 else 0) / 60 + (parti[2] if len(parti) > 2 else 0) / 3600
    return segno * v * (15 if ore else 1)


def catalogo_openngc():
    """Catalogo OpenNGC (NGC, IC e addendum) incluso nel programma: funziona senza Internet."""
    global _CATALOGO_NGC
    if _CATALOGO_NGC is None:
        p = percorso_dati(FILE_OPENNGC)
        if not p.exists():
            raise FileNotFoundError(f"catalogo {FILE_OPENNGC} non trovato accanto al programma")
        righe = []
        with open(p, encoding="utf-8") as f:
            lettore = csv.DictReader((r for r in f if not r.startswith("#")), delimiter=";")
            for r in lettore:
                try:
                    r["ra_deg"] = _sessagesimale_deg(r["ra"], True)
                    r["dec_deg"] = _sessagesimale_deg(r["dec"], False)
                except (ValueError, IndexError):
                    continue
                righe.append(r)
        _CATALOGO_NGC = righe
    return _CATALOGO_NGC


def _nome_ngc(n):
    m = re.match(r"^(NGC|IC)0*(\d+)(.*)$", n)
    if m:
        return f"{m.group(1)} {m.group(2)}{m.group(3)}"
    m = re.match(r"^M0*(\d+)$", n)
    return f"M {m.group(1)}" if m else n


def oggetti_openngc(w, shape):
    """Oggetti NGC/IC/Messier nel campo, dal catalogo incluso. Ordinati per
    importanza: prima i Messier, poi quelli con un nome comune, poi i più luminosi."""
    cat = catalogo_openngc()
    ny, nx = shape
    ra = np.array([r["ra_deg"] for r in cat])
    de = np.array([r["dec_deg"] for r in cat])
    ra0, de0 = (float(np.ravel(v)[0]) for v in pix_a_radec(w, (nx - 1) / 2, (ny - 1) / 2))
    ang = impronta_radec(w, nx, ny)
    raggio = max(separazione_deg(ra0, de0, a, b) for a, b in ang) * 1.05
    a1, d1, a2, d2 = map(np.radians, (ra, de, ra0, de0))
    sep = np.degrees(np.arccos(np.clip(np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(a1 - a2), -1, 1)))
    vicini = np.where(sep <= raggio)[0]
    oggetti = []
    if vicini.size:
        x, y = radec_a_pix(w, ra[vicini], de[vicini])
        for k, xx, yy in zip(vicini, x, y):
            if not (np.isfinite(xx) and np.isfinite(yy) and 0 <= xx < nx and 0 <= yy < ny):
                continue
            r = cat[k]
            nome_cat = _nome_ngc(r["nome"])
            messier = f"M {int(r['messier'])}" if r.get("messier", "").strip().isdigit() else None
            comuni = [c.strip() for c in r.get("nomi_comuni", "").split(",") if c.strip()]
            altri = [n for n in ([nome_cat] if messier else []) + comuni]
            mag = None
            for chiave in ("mag_v", "mag_b"):
                try:
                    mag = float(r[chiave])
                    break
                except (TypeError, ValueError, KeyError):
                    pass
            oggetti.append(dict(nome=messier or nome_cat, nome_simbad=nome_cat, altri_nomi="; ".join(altri),
                                ra=float(ra[k]), dec=float(de[k]), x=float(xx), y=float(yy),
                                tipo=TIPI_OPENNGC.get(r["tipo"], r["tipo"]), riferimenti=0, catalogo=True,
                                magnitudine=mag, fonte="OpenNGC", _comune=bool(comuni), _messier=bool(messier)))
    oggetti.sort(key=lambda o: (not o["_messier"], not o["_comune"],
                                o["magnitudine"] if o["magnitudine"] is not None else 99))
    return oggetti


def analisi_oggetti(d, w, max_nomi=12, oggetti=None, log=None):
    import matplotlib.patheffects as pe
    fonte = "dati forniti"
    if oggetti is None:
        # SIMBAD e NED interrogati insieme (in parallelo); se un oggetto è in
        # entrambi si tiene quello di SIMBAD, con il redshift di NED se manca
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_sim = ex.submit(oggetti_simbad, w, d.shape)
            f_ned = ex.submit(oggetti_ned, w, d.shape)
            sim = ned = None
            try:
                sim = f_sim.result()
                if log:
                    log(f"  SIMBAD: {len(sim)} oggetti nel campo")
            except Exception as e:
                if log:
                    log(f"  SIMBAD non raggiungibile ({str(e).splitlines()[0][:300]})")
            try:
                ned = f_ned.result()
                if log:
                    con_z = sum(1 for o in ned if o.get("z") is not None)
                    log(f"  NED: {len(ned)} oggetti nel campo, {con_z} con il redshift")
            except Exception as e2:
                if log:
                    log(f"  NED non raggiungibile ({str(e2).splitlines()[0][:300]})")
        try:
            ngc = oggetti_openngc(w, d.shape)
        except Exception:
            ngc = []
        if sim is not None and ned is not None:
            oggetti = unisci_oggetti(sim, ned, distanza_arcsec=2, prendi_z=True)
            if log:
                log(f"  SIMBAD + NED: {len(oggetti)} oggetti diversi "
                    f"({len(oggetti) - len(sim)} solo in NED)")
            fonte = "SIMBAD + NED"
        elif sim is not None:
            oggetti, fonte = sim, "SIMBAD"
        elif ned is not None:
            oggetti = unisci_oggetti(ned, ngc)
            fonte = "NED + OpenNGC" if ngc else "NED"
        else:
            if log:
                log("  uso il catalogo OpenNGC incluso nel programma (NGC, IC, Messier; senza Internet)")
            oggetti = ngc
            fonte = "OpenNGC, catalogo incluso"
    fig = _fig()
    ax, _ = _assi_wcs(fig, w, d, "Oggetti con nome nel campo")
    ny, nx = d.shape
    distanza_min = 0.07 * max(nx, ny)
    messi = []
    for o in oggetti:
        if len(messi) >= max_nomi:
            break
        if not o["catalogo"] and o["riferimenti"] < 20 and o.get("z") is None:
            continue
        if any(math.hypot(o["x"] - a, o["y"] - b) < distanza_min for a, b in messi):
            continue
        ax.plot(o["x"], o["y"], marker="+", color="#00e5ff", markersize=9, markeredgewidth=1.2)
        # sull'immagine solo il nome: tipo, redshift e il resto sono nel CSV
        t = ax.text(o["x"] + 0.012 * nx, o["y"] + 0.012 * ny, o["nome"], color="white", fontsize=9)
        t.set_path_effects([pe.withStroke(linewidth=2.2, foreground="black")])
        messi.append((o["x"], o["y"]))
    fig.suptitle(f"Oggetti con nome nel campo: {len(messi)} mostrati su {len(oggetti)} trovati (fonte: {fonte})",
                 fontsize=13, fontweight="bold")
    righe = []
    from astropy.coordinates import Angle
    import astropy.units as u
    for o in oggetti:
        righe.append({"nome": o["nome"], "nome_simbad": o["nome_simbad"], "altri_nomi": o["altri_nomi"],
                      "ra_deg": f"{o['ra']:.6f}", "dec_deg": f"{o['dec']:.6f}",
                      "ra_hms": Angle(o["ra"], u.deg).to_string(unit=u.hour, sep=":", precision=2),
                      "dec_dms": Angle(o["dec"], u.deg).to_string(unit=u.deg, sep=":", precision=1, alwayssign=True),
                      "tipo": o["tipo"],
                      "riferimenti_bibliografici": o["riferimenti"] if o.get("fonte", "SIMBAD").startswith(("SIMBAD", "NED")) else "",
                      "magnitudine": f"{o['magnitudine']:.2f}" if o.get("magnitudine") is not None else "",
                      "redshift_z": f"{o['z']:.5g}" if o.get("z") is not None else "",
                      "fonte": o.get("fonte", "SIMBAD")})
    return fig, righe


def analisi_raggi_x(immagini, log=None):
    """Analisi dei file a raggi X (conteggi di fotoni), sui dati originali:
    mappa del flusso (fotoni al secondo per secondo d'arco quadrato, levigata),
    profilo di brillanza superficiale attorno al punto più brillante e una
    tabella con conteggi, esposizione e flussi di ogni file."""
    if not immagini:
        raise ValueError("nessun file a raggi X (Chandra, XMM...) tra quelli inclusi")
    righe = []
    for img in immagini:
        ny, nx = img.shape
        d = img.blocco(0, ny, 0, nx)
        dentro = np.isfinite(d)
        scala = scala_pixel_deg(img.wcs) * 3600 if img.wcs is not None else float("nan")
        tot = float(np.nansum(d))
        area = float(dentro.sum()) * scala ** 2 / 3600 if math.isfinite(scala) else float("nan")
        esp = img.esposizione
        liscia = ndi.gaussian_filter(np.nan_to_num(d), 2.0)
        yp, xp = np.unravel_index(int(np.argmax(np.where(dentro, liscia, -1))), d.shape)
        fondo = float(np.median(liscia[dentro])) if dentro.any() else 0.0
        picco = img.wcs.pixel_to_world(xp, yp) if img.wcs is not None else None
        righe.append({"file": img.percorso.name, "telescopio": img.telescopio, "strumento": img.strumento,
                      "banda": img.filtro, "energia_keV": descrivi_lunghezza(img.lambda_um).replace(" keV", ""),
                      "esposizione_s": f"{esp:.1f}", "conteggi_totali": f"{tot:.0f}",
                      "conteggi_al_secondo": f"{tot / esp:.4g}" if esp > 0 else "",
                      "area_campo_arcmin2": f"{area:.2f}" if math.isfinite(area) else "",
                      "scala_arcsec_pixel": f"{scala:.3f}" if math.isfinite(scala) else "",
                      "picco_ra_deg": f"{picco.ra.deg:.6f}" if picco is not None else "",
                      "picco_dec_deg": f"{picco.dec.deg:.6f}" if picco is not None else "",
                      "fondo_conteggi_pixel": f"{fondo:.4g}"})
        if log:
            log(f"  {img.percorso.name}: {tot:.0f} fotoni in {esp:.0f} s")
    # figura sul file più dettagliato
    img = min(immagini, key=lambda i: scala_pixel_deg(i.wcs) if i.wcs is not None else 1e9)
    ny, nx = img.shape
    d = img.blocco(0, ny, 0, nx)
    dentro = np.isfinite(d)
    scala = scala_pixel_deg(img.wcs) * 3600
    esp = img.esposizione if img.esposizione > 0 else 1.0
    flusso = ndi.gaussian_filter(np.nan_to_num(d), 2.0) / esp / scala ** 2      # fotoni / s / arcsec²
    flusso[~dentro] = np.nan
    from matplotlib.colors import LogNorm
    fig = _fig((15, 7.5))
    ax = fig.add_subplot(1, 2, 1, projection=img.wcs)
    pos = flusso[np.isfinite(flusso) & (flusso > 0)]
    vmin = float(np.percentile(pos, 5)) if pos.size else 1e-6
    vmax = float(np.percentile(pos, 99.9)) if pos.size else 1.0
    im = ax.imshow(np.where(np.isfinite(flusso) & (flusso > 0), flusso, vmin), origin="lower", cmap="magma",
                   norm=LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10)), interpolation="nearest")
    ax.coords[0].set_axislabel("Ascensione retta (J2000)")
    ax.coords[1].set_axislabel("Declinazione (J2000)")
    ax.coords[0].set_major_formatter("hh:mm:ss")
    ax.coords[1].set_major_formatter("dd:mm:ss")
    fig.suptitle(f"Raggi X — {img.telescopio} {img.strumento} {img.filtro}\n"
                 f"flusso levigato (σ = 2 pixel), esposizione {img.esposizione:.0f} s")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("fotoni / s / arcsec²")
    liscia = ndi.gaussian_filter(np.nan_to_num(d), 2.0)
    yp, xp = np.unravel_index(int(np.argmax(np.where(dentro, liscia, -1))), d.shape)
    ax.plot([xp], [yp], marker="+", color="cyan", markersize=14, mew=1.5)
    Y, X = np.indices(d.shape)
    r = np.hypot(X - xp, Y - yp) * scala
    rmax = float(np.nanmax(np.where(dentro, r, np.nan)))
    bordi = np.linspace(0, rmax, 60)
    raggi, brill, err = [], [], []
    for a, b in zip(bordi[:-1], bordi[1:]):
        sel = dentro & (r >= a) & (r < b)
        n = int(sel.sum())
        if n < 5:
            continue
        c = float(np.nansum(d[sel]))
        raggi.append((a + b) / 2)
        brill.append(c / esp / (n * scala ** 2))
        err.append(math.sqrt(max(c, 1)) / esp / (n * scala ** 2))       # errore di Poisson
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.errorbar(raggi, brill, yerr=err, fmt="o", ms=3, color="#c03030", ecolor="#e09090", capsize=0)
    ax2.set_yscale("log")
    ax2.set_xlabel("Distanza dal punto più brillante (″)")
    ax2.set_ylabel("Brillanza superficiale (fotoni / s / arcsec²)")
    ax2.set_title("Profilo di brillanza superficiale (errori di Poisson)")
    ax2.grid(True, which="both", alpha=0.3)
    return fig, righe


def nome_da_cartella(cartella):
    """«NGC 7635», «IC 1396», «M 31» dal nome della cartella madre o delle cartelle
    sopra (es. ~/Astronomia/NGC_7635/SPHEREX_DR1_CUBE -> NGC 7635), oppure None."""
    if not cartella:
        return None
    for parte in reversed(Path(cartella).parts):
        m = re.search(r"(?<![A-Za-z])(NGC|IC|M|Messier)[ _\-]?0*(\d{1,4})(?!\d)", parte, re.IGNORECASE)
        if m:
            sigla = m.group(1).upper()
            return f"{'M' if sigla in ('M', 'MESSIER') else sigla} {int(m.group(2))}"
    return None


def analisi_cielo(d, w, log=None, nome=None, cartella=None):
    """Mappa della costellazione che contiene l'immagine, nello stile delle carte IAU
    (mappa_cielo.py). Il nome scritto accanto al campo è quello dell'oggetto NGC/IC/M
    più vicino al centro (catalogo incluso), se c'è."""
    import mappa_cielo as MC
    from astropy.wcs.utils import pixel_to_skycoord
    ny, nx = d.shape
    # anche le mappe in coordinate galattiche (es. Herschel) vengono portate in AR/Dec
    xs = np.array([(nx - 1) / 2, -0.5, nx - 0.5, nx - 0.5, -0.5])
    ys = np.array([(ny - 1) / 2, -0.5, -0.5, ny - 0.5, ny - 0.5])
    c = pixel_to_skycoord(xs, ys, w).icrs
    ra, de = float(c.ra.deg[0]), float(c.dec.deg[0])
    contorno = np.column_stack([c.ra.deg[1:], c.dec.deg[1:]])
    if nome is None:
        # 1) l'oggetto scritto nel nome della cartella (NGC_7635 -> NGC 7635), se è nel campo;
        # 2) altrimenti il più importante del campo (Messier, poi con nome comune, poi il più brillante)
        try:
            ogg = oggetti_openngc(w, d.shape)
        except Exception:
            ogg = []
        dalla_cartella = nome_da_cartella(cartella)
        scelto = None
        if dalla_cartella:
            chiave = dalla_cartella.replace(" ", "").upper()
            scelto = next((o for o in ogg if chiave in [o["nome"].replace(" ", "").upper(),
                                                         str(o.get("nome_simbad", "")).replace(" ", "").upper()]
                           + [x.replace(" ", "").upper() for x in str(o.get("altri_nomi", "")).split(";")]), None)
            if scelto is None:
                nome = dalla_cartella            # fuori dal catalogo incluso: il nome scelto dall'utente
        if nome is None:
            nome = (scelto or (ogg[0] if ogg else {})).get("nome", "")
    dati = MC.carica_dati(log=log)
    fig, info = MC.disegna_mappa(dati, ra, de, contorno, nome)
    from astropy.coordinates import Angle
    import astropy.units as u
    lato = scala_pixel_deg(w) * 60 * max(nx, ny)
    righe = [{"oggetto": nome or "centro dell'immagine", "ra_deg": f"{ra:.6f}", "dec_deg": f"{de:.6f}",
              "ra_hms": Angle(ra, u.deg).to_string(unit=u.hour, sep=":", precision=2),
              "dec_dms": Angle(de, u.deg).to_string(unit=u.deg, sep=":", precision=1, alwayssign=True),
              "costellazione": info["costellazione"], "nome_latino": info["latino"],
              "sigla_iau": info["abbreviazione"], "lato_campo_arcmin": f"{lato:.2f}",
              "stelle_con_nome_vicine": "; ".join(
                  f"{v['stella']} ({v['sigla']}, mag {v['magnitudine']}, {v['distanza_gradi']}°)"
                  for v in info["vicine"])}]
    if log:
        log(f"  costellazione: {info['costellazione']} ({info['latino']})")
    return fig, righe


FUNZIONI_ANALISI = {"griglia": analisi_griglia, "snr": analisi_snr, "gradiente": analisi_gradiente,
                    "strutture": analisi_strutture, "isofote": analisi_isofote,
                    "sorgenti": analisi_sorgenti, "istogramma": analisi_istogramma,
                    "profilo": analisi_profilo, "oggetti": analisi_oggetti, "cielo": analisi_cielo}
CSV_ANALISI = {"sorgenti": "analisi_sorgenti.csv", "profilo": "analisi_profilo_radiale.csv",
               "oggetti": "oggetti_nel_campo.csv", "raggi_x": "analisi_raggi_x.csv",
               "cielo": "posizione_nel_cielo.csv"}


def scrivi_csv(percorso, righe):
    if not righe:
        return None
    with open(percorso, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(righe[0].keys()), delimiter=";")
        wr.writeheader()
        wr.writerows(righe)
    return Path(percorso)


def figura_in_rgba(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def salva_figura(fig, percorso, formato, firma=False):
    """Salva una figura delle analisi; con firma=True scrive «© Edoardo Battistel»
    in piccolo nell'angolo in basso a destra (solo nel file, non sullo schermo)."""
    est = FORMATI_IMMAGINE.get(formato, ".png")
    p = Path(str(Path(percorso).with_suffix("")) + est)
    kw = {"dpi": 200}
    if est == ".jpg":
        kw["pil_kwargs"] = {"quality": 92}
    t = fig.text(0.995, 0.005, FIRMA, ha="right", va="bottom", fontsize=8,
                 color="0.45", alpha=0.8) if firma else None
    try:
        if formato == "JPEG per forum":
            import io
            from PIL import Image
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            _jpeg_forum(Image.open(buf), p)
        else:
            fig.savefig(p, **kw)
    finally:
        if t is not None:
            t.remove()
    return p


# =============================================================================
# STATO DELLA PIPELINE E LOG
# =============================================================================
class Stato:
    """Scelte salvate in <madre>/aap_stato.json e riassunte in aap_log.txt.
    Se c'è solo il file della versione 1.4 (pipeline_stato.json) viene letto
    quello, così il lavoro fatto con Pipeline FITS si riprende."""

    def __init__(self, madre):
        self.madre = Path(madre)
        self.p = self.madre / FILE_STATO
        self.dati = {"versione": 1, "file": {}, "eventi": []}
        sorgente = self.p if self.p.exists() else self.madre / FILE_STATO_VECCHIO
        if sorgente.exists():
            try:
                self.dati.update(json.loads(sorgente.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass

    def file(self, nome):
        return self.dati["file"].setdefault(nome, {"incluso": True})

    def imposta(self, nome, chiave, valore, evento=None):
        self.file(nome)[chiave] = valore
        if evento:
            self.evento(f"{nome}: {evento}")
        self.salva()

    def globale(self, chiave, predefinito=None):
        return self.dati.get(chiave, predefinito)

    def imposta_globale(self, chiave, valore, evento=None):
        self.dati[chiave] = valore
        if evento:
            self.evento(evento)
        self.salva()

    def evento(self, testo):
        self.dati["eventi"].append(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {testo}")
        self.dati["eventi"] = self.dati["eventi"][-1000:]

    def salva(self):
        try:
            tmp = self.p.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.dati, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            os.replace(tmp, self.p)
            self.scrivi_log()
        except OSError:
            pass

    def scrivi_log(self):
        r = [f"LOG DI {NOME_PROGRAMMA.upper()} ({SIGLA}) - {self.madre}",
             f"aggiornato il {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"{NOME_PROGRAMMA} {VERSIONE} - {COPYRIGHT}", "",
             "SCELTE PER FILE", "=" * 100,
             f"{'file':<14}{'incluso':<9}{'noise reduction':<26}{'fondo':<24}{'stretching':<28}{'riparazione'}",
             "-" * 100]
        for nome in sorted(self.dati["file"]):
            s = self.dati["file"][nome]
            nr = s.get("nr") or {}
            bg = s.get("bg") or {}
            st = s.get("stretch") or {}
            rp = s.get("riparazione") or {}
            t_nr = f"{ALGORITMI_NR.get(nr.get('algo'), {}).get('etichetta', '-')} {nr.get('valore', '')}" if nr else "-"
            if not s.get("usa_bg", True):
                t_bg = "non applicato"
            elif bg.get("box"):
                t_bg = f"{ALGORITMI_FONDO.get(bg.get('algo'), {}).get('etichetta', '')} box {bg['box']}"
            elif bg:
                t_bg = "nessuna sottrazione"
            else:
                t_bg = "-"
            t_st = descrivi_stretch(st["algo"], st.get("valore")) if st else "-"
            t_rp = []
            if rp.get("auto"):
                t_rp.append(f"auto ({len(rp.get('zone_auto', []))} zone)")
            if rp.get("rettangoli"):
                t_rp.append(f"{len(rp['rettangoli'])} manuali")
            r.append(f"{nome:<14}{'sì' if s.get('incluso', True) else 'NO':<9}{t_nr:<26}{t_bg:<24}{t_st:<28}"
                     f"{', '.join(t_rp) or '-'}")
        for chiave, titolo in (("stack", "STACK BIANCO E NERO"), ("rgb", "STACK RGB")):
            if self.dati.get(chiave):
                r += ["", titolo, "=" * 100]
                for k, v in self.dati[chiave].items():
                    r.append(f"  {k}: {v}")
        r += ["", "CRONOLOGIA", "=" * 100] + self.dati["eventi"]
        (self.madre / FILE_LOG).write_text("\n".join(r) + "\n", encoding="utf-8")


# =============================================================================
# CSV CON TUTTE LE SCELTE
# =============================================================================
def scrivi_csv_scelte(stato, percorso=None):
    """Scrive <madre>/aap_scelte.csv: una riga per file con tutte le scelte
    (rumore, fondo, stretching, riparazione, peso nello stack, pesi RGB) e,
    ripetute su ogni riga, le impostazioni di stack, ritaglio e RGB.
    Separatore ';', si apre con LibreOffice o Excel."""
    d = stato.dati
    st = d.get("stack") or {}
    rgb = d.get("rgb") or {}
    pesi_rgb = rgb.get("pesi") or {}
    pesi_stack = dict(zip(st.get("file", []), st.get("pesi", []))) or (d.get("pesi_stack") or {})
    rit = d.get("ritaglio") or {}
    box = rit.get("box")
    t_rit = f"x {box[2]}-{box[3]}, y {box[0]}-{box[1]} ({box[3] - box[2]}x{box[1] - box[0]})" if box else "nessuno"
    ora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    righe = []
    for nome in sorted(d.get("file", {})):
        s = d["file"][nome]
        nr = s.get("nr") or {}
        bg = s.get("bg") or {}
        stc = s.get("stretch") or {}
        rp = s.get("riparazione") or {}
        prgb = pesi_rgb.get(nome)
        righe.append({
            "file": nome,
            "incluso": "sì" if s.get("incluso", True) else "no",
            "filtro": s.get("filtro", ""),
            "lunghezza_onda_um": s.get("lambda_um", ""),
            "riduzione_rumore": (ALGORITMI_NR.get(nr.get("algo"), {}).get("etichetta", "") if nr.get("valore") is not None
                                 else ("nessuna" if nr else "")),
            "rumore_valore": "" if nr.get("valore") is None else nr["valore"],
            "fondo_stimatore": (ALGORITMI_FONDO.get(bg.get("algo"), {}).get("etichetta", "") if bg.get("box")
                                else ("nessuna sottrazione" if bg else "")),
            "fondo_box_px": bg.get("box") or "",
            "fondo_applicato": "sì" if (s.get("usa_bg", bool(bg.get("box"))) and bg.get("box")) else "no",
            "stretching": stc.get("algo", ""),
            "stretching_valore": "" if stc.get("valore") is None else stc["valore"],
            "stretching_descrizione": descrivi_stretch(stc["algo"], stc.get("valore")) if stc else "",
            "riparazione_automatica": "sì" if rp.get("auto") else "no",
            "zone_automatiche": len(rp.get("zone_auto", [])) if rp.get("auto") else 0,
            "zone_disegnate": len(rp.get("rettangoli", [])),
            "nello_stack_bn": "sì" if nome in st.get("file", []) else "no",
            "peso_stack_bn": pesi_stack.get(nome, ""),
            "peso_R": prgb[0] if prgb else "",
            "peso_G": prgb[1] if prgb else "",
            "peso_B": prgb[2] if prgb else "",
            "stack_metodo": st.get("metodo", ""),
            "stack_griglia": st.get("griglia", ""),
            "stack_riferimento": st.get("riferimento", ""),
            "stack_riduzione": st.get("riduzione", ""),
            "stack_dimensione": st.get("dimensione", ""),
            "ritaglio": t_rit,
            "rgb_scala_canali": " ".join(str(x) for x in rgb.get("scala_canali", [])),
            "programma": f"{NOME_PROGRAMMA} {VERSIONE}",
            "aggiornato_il": ora,
        })
    if not righe:
        return None
    return scrivi_csv(Path(percorso) if percorso else stato.madre / FILE_SCELTE_CSV, righe)
