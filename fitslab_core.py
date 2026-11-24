# -*- coding: utf-8 -*-
"""
FITS LAB - MOTORE
=================
Tutto ciò che non è finestra:
  - riconoscimento della missione, del filtro e delle unità (anche Euclid,
    Roman, Planck, radio e raggi X; filtri e lunghezze d'onda da AutoAstroPipe);
  - apertura dei file: immagini 2D, cubi (piano o media), file .asdf di Roman,
    cubi SPHEREx, mappe HEALPix di Planck (cielo intero o ritaglio);
  - lettura di una zona del file ridotta di un fattore (file enormi);
  - geometria e WCS: ogni ritaglio o riduzione aggiorna le coordinate
    (anche le distorsioni SIP), così i FITS salvati restano «in cielo»;
  - referto di qualità (fondo, rumore, gradiente, stelle, saturazione, buchi);
  - esecuzione della ricetta (passi non distruttivi) e uscite separate:
    science/ (FITS lineari misurabili) e aesthetic/ (immagini e FITS stirati).

Le uscite (nomi dei file, intestazioni, rapporti) sono sempre in inglese.

© 2026 Edoardo Battistel - FITS Lab
"""

import json
import math
import os
import re
import sys
import tempfile
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

import fitslab_operazioni as O
from fitslab_operazioni import DTYPE, Geometria, Contesto
import fitslab_esterni  # noqa: F401  (registra StarNet2 e GraXpert tra le operazioni)

NOME = "FITS Lab"
VERSIONE = "1.7"
AUTORE = "Edoardo Battistel"
ANNO = "2026"
COPYRIGHT = f"© {ANNO} {AUTORE}"
COPYRIGHT_ASCII = f"(C) {ANNO} {AUTORE}"
FIRMA = f"© {AUTORE}"
FILE_IMPOSTAZIONI = Path.home() / ".fitslab.json"
CARTELLA_CACHE = Path.home() / ".cache" / "fitslab"
LATO_LAVORO = 3000          # lato massimo dell'immagine di lavoro con riduzione «auto»

ESTENSIONI = (".fits", ".fit", ".fts", ".fits.gz", ".fit.gz", ".fts.gz", ".fits.fz", ".fz", ".fits.bz2",
              ".asdf", ".imh")


def _M():
    """Motore di AutoAstroPipe 2.5 (copia congelata): filtri, firma, SIMBAD, NED..."""
    import motore_aap
    return motore_aap


def ascii_fits(t):
    return _M()._ascii(t)


# =============================================================================
# IMPOSTAZIONI
# =============================================================================
PREDEFINITE = dict(lingua="it", testo=150, riduzione="auto", ultima_cartella=str(Path.home()),
                   firma=True, formati=["science_fits", "png", "tiff16"], pannello=True)


def carica_impostazioni():
    d = dict(PREDEFINITE)
    try:
        d.update(json.loads(FILE_IMPOSTAZIONI.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    return d


def salva_impostazioni(d):
    try:
        FILE_IMPOSTAZIONI.write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


# =============================================================================
# MISSIONI
# =============================================================================
# (nome mostrato, parole nell'intestazione o nel nome del file)
MISSIONI = [
    ("JWST", ("JWST",), ("jw0", "jw1", "jw2", "jw3", "jw4", "jw5", "jw6", "jw7", "jw8", "jw9")),
    ("Hubble", ("HST", "HUBBLE"), ("hst_", "_drz", "_drc", "_flc", "_flt")),
    ("Euclid", ("EUCLID",), ("euc_",)),
    ("Roman", ("ROMAN", "WFIRST"), ("roman", "_wfi")),
    ("Spitzer", ("SPITZER",), ("spitzer", "glimpse", "smog", "seip", "irac", "mips")),
    ("Herschel", ("HERSCHEL",), ("hpacs", "hspire", "hppjs", "herschel")),
    ("WISE", ("WISE", "NEOWISE"), ("wise", "unwise")),
    ("SPHEREx", ("SPHEREX",), ("spherex", "_spx_")),
    ("Planck", ("PLANCK",), ("hfi_", "lfi_", "com_cmb", "planck")),
    ("Pan-STARRS", ("PS1", "PAN-STARRS", "GPC1"), ("rings.v3", "skycell", "ps1")),
    ("SDSS", ("SDSS", "SLOAN"), ("frame-", "sdss")),
    ("Legacy Survey", ("DECAM", "BASS", "MZLS", "LEGACY"), ("legacysurvey", "decals", "-image-")),
    ("DES", ("DES", "BLANCO"), ("des_",)),
    ("2MASS", ("2MASS",), ("2mass", "aj_", "ah_", "ak_")),
    ("GALEX", ("GALEX",), ("-nd-", "-fd-", "galex")),
    ("Swift", ("SWIFT",), ("swift", "uvot", "_sk.img")),
    ("Chandra", ("CHANDRA",), ("acis", "hrc", "chandra")),
    ("XMM-Newton", ("XMM", "NEWTON"), ("pn_", "mos1", "mos2", "xmm")),
    ("eROSITA", ("EROSITA", "SRG"), ("erosita", "em01_", "eb01_")),
    ("TESS", ("TESS",), ("tess",)),
    ("Kepler", ("KEPLER", "K2"), ("kplr", "ktwo")),
    ("ZTF", ("ZTF", "P48"), ("ztf_",)),
    ("DSS", ("POSS", "DSS", "UKST", "ESO-SERC", "PALOMAR"), ("dss",)),
    ("Subaru", ("SUBARU", "HSC"), ("hsc", "calexp")),
    ("VISTA", ("VISTA", "VIRCAM"), ("vista", "vvv", "viking")),
    ("UKIDSS", ("UKIRT", "WFCAM"), ("ukidss",)),
    ("GALEX", ("GALEX",), ()),
    ("VLA", ("VLA", "EVLA", "JVLA"), ("nvss", "first", "vlass")),
    ("LOFAR", ("LOFAR",), ("lotss", "lofar")),
    ("ALMA", ("ALMA",), ("alma", "member.uid")),
    ("MeerKAT", ("MEERKAT",), ("meerkat",)),
    ("ASKAP", ("ASKAP",), ("racs", "askap", "emu_")),
    ("Fermi", ("GLAST", "FERMI"), ("fermi", "gll_")),
    ("Gaia", ("GAIA",), ("gaia",)),
    ("SDO", ("SDO",), ("aia", "hmi")),
]


def riconosci_missione(h, nome_file=""):
    tel = " ".join(str(h.get(k, "")) for k in ("TELESCOP", "MISSION", "OBSERVAT", "ORIGIN", "SURVEY",
                                                  "FPA.TELESCOPE", "INSTRUME")).upper()
    nf = Path(nome_file).name.lower()
    for nome, parole, nomi in MISSIONI:
        if any(re.search(r"(^|[^A-Z])" + re.escape(p) + r"([^A-Z]|$)", tel) for p in parole):
            return nome
    for nome, parole, nomi in MISSIONI:
        if any(n in nf for n in nomi):
            return nome
    t = str(h.get("TELESCOP", "")).strip()
    return t or "unknown"


def filtro_euclid(nome_file, h):
    f = str(h.get("FILTER", "")).strip().upper()
    if f in ("VIS", "NIR_Y", "NIR_J", "NIR_H"):
        return f
    m = re.search(r"(?:^|[-_])(VIS|NIR[-_]?([YJH]))(?:[-_.]|$)", Path(nome_file).name.upper())
    if m:
        return "VIS" if m.group(1) == "VIS" else f"NIR_{m.group(2)}"
    return None


def descrivi_lunghezza(um):
    if um is None:
        return "?"
    if um < 0.01:
        return f"{12.398 / (um * 1e4):.2f} keV"
    if um < 1:
        return f"{um * 1000:.0f} nm"
    if um < 1000:
        return f"{um:.2f} µm"
    if um < 1e5:
        return f"{um / 1000:.2f} mm"
    return f"{299792.458 / um:.3g} GHz"


def lunghezza_da_frequenza(h):
    """Radio/millimetrico: frequenza (FREQ, RESTFRQ, CRVAL3 in Hz) -> µm."""
    for k in ("FREQ", "RESTFRQ", "RESTFREQ", "OBSFREQ"):
        try:
            v = float(h.get(k))
        except (TypeError, ValueError):
            continue
        if v > 0:
            hz = v if v > 1e6 else v * 1e9          # Planck scrive FREQ = 143 (GHz)
            return 2.99792458e14 / hz
    for i in (3, 4):
        if str(h.get(f"CTYPE{i}", "")).upper().startswith("FREQ"):
            try:
                return 2.99792458e14 / float(h[f"CRVAL{i}"])
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                pass
    return None


# =============================================================================
# APERTURA DEI FILE
# =============================================================================
def e_supportato(p):
    n = str(p).lower()
    return n.endswith(ESTENSIONI)


def ispeziona(percorso):
    """Cosa c'è nel file, per decidere se chiedere qualcosa prima di aprirlo.
    {'kind': 'image'|'cube'|'healpix'|'asdf'|'spherex', 'planes': n, 'columns': [...],
     'shape': (ny, nx), 'mission': ..., 'size_mb': ...}"""
    p = Path(percorso)
    info = dict(kind="image", size_mb=p.stat().st_size / 1e6, path=str(p))
    if p.name.lower().endswith(".asdf"):
        info["kind"] = "asdf"
        info["mission"] = "Roman"
        return info
    from astropy.io import fits
    import fitslab_healpix as HP
    with fits.open(p, memmap=True, lazy_load_hdus=True) as hdul:
        i = HP.e_mappa_healpix(hdul)
        if i is not None:
            h = hdul[i].header
            info.update(kind="healpix", hdu=i, nside=int(h.get("NSIDE", 0)), columns=HP.colonne_mappa(hdul[i]),
                        coordsys=str(h.get("COORDSYS", "G")).strip(),
                        mission=riconosci_missione({**dict(hdul[0].header), **dict(h)}, p.name))
            return info
        idx, hdu = _hdu_immagine(hdul)
        h = hdu.header
        dims = [int(h.get(f"NAXIS{k}", 1)) for k in range(1, int(h.get("NAXIS", 0)) + 1)]
        utili = [d for d in dims[2:] if d > 1]
        hp = hdul[0].header
        info.update(hdu=idx, shape=(dims[1], dims[0]), mission=riconosci_missione({**dict(hp), **dict(h)}, p.name))
        if utili:
            info.update(kind="cube", planes=int(np.prod(utili)))
            if info["mission"] == "SPHEREx" or "SPECTRAL_CHANNELS" in [x.name for x in hdul]:
                info["kind"] = "spherex"
    return info


def _hdu_immagine(hdul):
    """(indice, hdu) della prima immagine con almeno 2 assi > 1 (anche cubi e 4D radio)."""
    if "SCI" in hdul and hdul["SCI"].header.get("NAXIS", 0) >= 2:
        return hdul.index_of("SCI"), hdul["SCI"]
    for i, hdu in enumerate(hdul):
        h = hdu.header
        x = str(h.get("XTENSION", "IMAGE")).strip().upper()
        if x in ("BINTABLE", "TABLE", "A3DTABLE") and not (h.get("ZIMAGE") in (True, "T")):
            continue
        n = int(h.get("NAXIS", 0) or h.get("ZNAXIS", 0) or 0)
        if n >= 2 and int(h.get("NAXIS1", h.get("ZNAXIS1", 0)) or 0) > 1 and int(h.get("NAXIS2", h.get("ZNAXIS2", 0)) or 0) > 1:
            return i, hdu
    raise ValueError("no 2D image in this file")


class Sorgente:
    """Il file aperto: metadati, WCS completo e lettura di zone ridotte.
    I dati possono stare nel file (memmap) o in memoria (Planck, cubi già ridotti)."""

    def __init__(self, percorso, scelte=None, progresso=None):
        scelte = scelte or {}
        self.percorso = Path(percorso)
        self.scelte = dict(scelte)
        self.in_memoria = None
        self.piano = None
        self.nota_apertura = ""
        info = ispeziona(percorso)
        self.tipo = info["kind"]
        if self.tipo == "asdf":
            self._apri_asdf()
        elif self.tipo == "healpix":
            self._apri_healpix(scelte, progresso)
            return
        elif self.tipo == "spherex" and scelte.get("spherex_band"):
            self._apri_spherex(scelte)
        self._apri_immagine(scelte)

    # ---------------------------------------------------------------- immagini
    def _apri_immagine(self, scelte):
        from astropy.io import fits
        from astropy.wcs import WCS
        M = _M()
        with fits.open(self.percorso, memmap=True) as hdul:
            self.indice, hdu = _hdu_immagine(hdul)
            self.hdr_s = hdu.header.copy()
            self.hdr_p = hdul[0].header.copy() if self.indice != 0 else fits.Header()
        h = {**dict(self.hdr_p), **dict(self.hdr_s)}
        self.h = h
        dims = [int(self.hdr_s.get(f"NAXIS{k}", 1)) for k in range(1, int(self.hdr_s.get("NAXIS", 0)) + 1)]
        self.shape = (dims[1], dims[0])
        self.extra_assi = dims[2:]
        self.missione = riconosci_missione(h, self.percorso.name)
        filtro = filtro_euclid(self.percorso.name, h) if self.missione == "Euclid" else None
        if filtro is None:
            filtro, _ = M.rileva_filtro(self.hdr_p, self.hdr_s, self.percorso.name)
        self.filtro = filtro if filtro != "sconosciuto" else "unknown"
        self.lambda_um = M.lunghezza_onda_um(self.filtro, self.hdr_p, self.hdr_s) or lunghezza_da_frequenza(h)
        self.unita = str(h.get("BUNIT", h.get("UNITS", ""))).strip()
        self.conteggi = bool(M.e_conteggi_x(self.hdr_p, self.hdr_s))
        try:
            self.asinh_ps1 = (float(h["BSOFTEN"]), float(h["BOFFSET"])) if "BSOFTEN" in h and "BOFFSET" in h else None
        except (TypeError, ValueError):
            self.asinh_ps1 = None
        if self.asinh_ps1:
            self.unita = self.unita or "counts"
        try:
            w = WCS(self.hdr_s, naxis=2)
            self.wcs = w.celestial if w.has_celestial else None
        except Exception:
            self.wcs = None
        self.distorsioni_tabella = self.wcs is not None and (
            getattr(self.wcs, "cpdis1", None) is not None or getattr(self.wcs, "det2im1", None) is not None)
        self.cubo = None
        utili = [d for d in self.extra_assi if d > 1]
        if utili:
            self.cubo = dict(planes=int(np.prod(utili)), mode=scelte.get("cube_mode", "plane"),
                             plane=int(scelte.get("cube_plane", 0)))
            self._prepara_cubo()

    def _prepara_cubo(self):
        """Cubi (radio, ALMA, SPHEREx, IFU): un piano scelto, oppure media/somma
        (momento 0) di un intervallo di piani, tenuto in memoria."""
        from astropy.io import fits
        c = self.cubo
        with fits.open(self.percorso, memmap=True) as hdul:
            d = hdul[self.indice].data
            d = d.reshape((-1,) + d.shape[-2:])
            if c["mode"] == "plane":
                k = int(min(max(c["plane"], 0), d.shape[0] - 1))
                a = np.array(d[k], dtype=DTYPE)
                self.piano = k
            else:
                p0 = int(self.scelte.get("cube_from", 0))
                p1 = int(self.scelte.get("cube_to", d.shape[0] - 1))
                p0, p1 = max(0, min(p0, p1)), min(d.shape[0] - 1, max(p0, p1))
                acc = np.zeros(d.shape[-2:], dtype=np.float64)
                n = np.zeros(d.shape[-2:], dtype=np.int32)
                for k in range(p0, p1 + 1):
                    b = np.asarray(d[k], dtype=np.float64)
                    ok = np.isfinite(b)
                    acc[ok] += b[ok]
                    n += ok
                with np.errstate(invalid="ignore", divide="ignore"):
                    a = (acc / n if c["mode"] == "mean" else np.where(n > 0, acc, np.nan)).astype(DTYPE)
                self.piano = (p0, p1)
        self.in_memoria = a
        self.nota_apertura = (f"cube: plane {self.piano}" if c["mode"] == "plane"
                              else f"cube: {c['mode']} of planes {self.piano[0]}-{self.piano[1]}")

    # ---------------------------------------------------------------- Roman / SPHEREx
    def _apri_asdf(self):
        import roman_asdf
        CARTELLA_CACHE.mkdir(parents=True, exist_ok=True)
        dest = CARTELLA_CACHE / (self.percorso.stem + "_roman.fits")
        if not dest.exists() or dest.stat().st_mtime < self.percorso.stat().st_mtime:
            roman_asdf.converti_in_fits(self.percorso, dest, log=lambda *a: None, creatore=f"{NOME} {VERSIONE}",
                                        autore=AUTORE, nota_ascii=COPYRIGHT_ASCII)
        self.originale = self.percorso
        self.percorso = dest
        self.nota_apertura = f"Roman ASDF converted to FITS ({self.originale.name})"

    def _apri_spherex(self, scelte):
        M = _M()
        bande = M.converti_cubo_spherex(self.percorso, log=lambda *a: None)
        k = int(scelte.get("spherex_band", 1))
        scelti = [b for b in (bande or []) if f"_D{k}_" in Path(b).name]
        if scelti:
            self.originale = self.percorso
            self.percorso = Path(scelti[0])
            self.nota_apertura = f"SPHEREx band D{k} from {self.originale.name}"

    # ---------------------------------------------------------------- Planck / HEALPix
    def _apri_healpix(self, scelte, progresso=None):
        import fitslab_healpix as HP
        mappa = HP.MappaHEALPix(self.percorso, scelte.get("column"))
        modo = scelte.get("projection", "allsky")
        sistema = scelte.get("frame", mappa.sistema)
        if modo == "allsky":
            w, shape = HP.wcs_cielo_intero(sistema, int(scelte.get("width", 2400)))
        else:
            w, shape = HP.wcs_ritaglio(sistema, float(scelte["lon"]), float(scelte["lat"]),
                                       float(scelte.get("size_deg", 10)), float(scelte.get("pixel_arcmin",
                                       max(0.5, mappa.risoluzione_deg() * 60 / 2))))
        a = HP.proietta(mappa, w, shape, sistema, progresso=progresso)
        self.in_memoria = a
        self.shape = a.shape
        self.extra_assi = []
        self.cubo = None
        self.wcs = w
        self.distorsioni_tabella = False
        from astropy.io import fits
        self.hdr_p = mappa.header_primario
        self.hdr_s = fits.Header()
        for k in ("TELESCOP", "INSTRUME", "FREQ", "BAND", "DATE-OBS", "ORIGIN", "FILENAME"):
            if k in mappa.header:
                self.hdr_s[k] = mappa.header[k]
        self.h = {**dict(self.hdr_p), **dict(mappa.header)}
        self.missione = riconosci_missione(self.h, self.percorso.name)
        if self.missione == "unknown":
            self.missione = "HEALPix map"
        m = re.search(r"(\d{2,3})[-_]", self.percorso.name)
        freq = mappa.frequenza or (m.group(1) if m else "")
        self.filtro = f"{freq} GHz" if freq and str(freq).replace(".", "").isdigit() else (mappa.colonna or "map")
        self.lambda_um = lunghezza_da_frequenza({"FREQ": float(freq)}) if str(freq).replace(".", "").isdigit() else None
        self.unita = mappa.unita
        self.conteggi = False
        self.asinh_ps1 = None
        self.piano = None
        self.nota_apertura = (f"HEALPix nside {mappa.nside} ({'NESTED' if mappa.nested else 'RING'}), column "
                              f"{mappa.colonna}, " + ("all-sky Mollweide" if modo == "allsky" else
                              f"TAN cutout {scelte.get('size_deg', 10)} deg at ({scelte['lon']}, {scelte['lat']})"))
        self.healpix = dict(nside=mappa.nside, column=mappa.colonna, frame=sistema, projection=modo)

    # ---------------------------------------------------------------- lettura
    def _prepara_blocco(self, b):
        b = np.asarray(b, dtype=DTYPE)
        if self.asinh_ps1 is not None:
            b = _M().flusso_ps1(b, *self.asinh_ps1).astype(DTYPE)
        if not self.conteggi and self.in_memoria is None:
            b = b.copy() if not b.flags.writeable else b
            b[b == 0] = np.nan                 # ottico/IR: lo 0 esatto è «nessun dato»
        return b

    def leggi(self, zona=None, fattore=1, progresso=None):
        """Array float32 della zona (y0, y1, x0, x1) ridotta del fattore (media a blocchi)."""
        ny, nx = self.shape
        y0, y1, x0, x1 = zona or (0, ny, 0, nx)
        y0, x0 = max(0, int(y0)), max(0, int(x0))
        y1, x1 = min(ny, int(y1)), min(nx, int(x1))
        f = max(1, int(fattore))
        y1 = y0 + (y1 - y0) // f * f
        x1 = x0 + (x1 - x0) // f * f
        if self.in_memoria is not None:
            src_ctx = None
            d = self.in_memoria
        else:
            from astropy.io import fits
            try:
                src_ctx = fits.open(self.percorso, memmap=True)
                d = src_ctx[self.indice].data
            except ValueError:
                src_ctx = fits.open(self.percorso, memmap=False)
                d = src_ctx[self.indice].data
            while d.ndim > 2:
                d = d[0]
        try:
            out = np.empty(((y1 - y0) // f, (x1 - x0) // f), dtype=DTYPE)
            passo = max(f, (64 * 1024 * 1024 // 4 // max(1, x1 - x0)) // f * f)
            for ya in range(y0, y1, passo):
                yb = min(y1, ya + passo)
                b = self._prepara_blocco(d[ya:yb, x0:x1])
                if f > 1:
                    b = O.riduci_blocchi(b, f, "mean")
                out[(ya - y0) // f:(yb - y0) // f] = b
                if progresso:
                    progresso((yb - y0) / max(1, y1 - y0))
        finally:
            if src_ctx is not None:
                src_ctx.close()
        return out

    def fattore_auto(self, zona=None, lato=LATO_LAVORO):
        ny, nx = self.shape
        if zona:
            ny, nx = zona[1] - zona[0], zona[3] - zona[2]
        return max(1, int(math.ceil(max(ny, nx) / lato)))

    def memoria_mb(self, zona=None):
        ny, nx = self.shape
        if zona:
            ny, nx = zona[1] - zona[0], zona[3] - zona[2]
        return ny * nx * 4 / 1e6

    # ---------------------------------------------------------------- metadati
    def meta(self):
        h = self.h
        def num(*chiavi):
            for k in chiavi:
                try:
                    v = float(h.get(k))
                    if v > 0:
                        return v
                except (TypeError, ValueError):
                    pass
            return None
        scala = scala_arcsec(self.wcs)
        return dict(
            file=self.percorso.name, mission=self.missione, instrument=str(h.get("INSTRUME", "")).strip(),
            filter=self.filtro, wavelength_um=self.lambda_um, wavelength=descrivi_lunghezza(self.lambda_um),
            unit=self.unita, exposure_s=num("EXPTIME", "EXPOSURE", "XPOSURE", "TEXPTIME", "LIVETIME", "ONTIME"),
            date=str(h.get("DATE-OBS", h.get("DATE_OBS", h.get("DATE-BEG", "")))).strip(),
            object=str(h.get("OBJECT", h.get("TARGNAME", ""))).strip(), shape=self.shape,
            pixel_arcsec=scala, wcs=self.wcs is not None, counts=self.conteggi,
            saturate=num("SATURATE", "SATLEVEL", "MAXLIN"), note=self.nota_apertura,
            beam_arcsec=(num("BMAJ") * 3600 if num("BMAJ") else None))


def scala_arcsec(w):
    if w is None:
        return None
    try:
        from astropy.wcs.utils import proj_plane_pixel_scales
        return float(np.mean(np.abs(proj_plane_pixel_scales(w)))) * 3600
    except Exception:
        return None


# =============================================================================
# WCS DOPO RITAGLI E RIDUZIONI
# =============================================================================
def crpix_per_geometria(crpix, ox, oy, s):
    """CRPIX (1-based) nel nuovo array: X_orig = s*X_arr + o - s/2 + 1/2."""
    return ((crpix[0] - ox + s / 2 - 0.5) / s, (crpix[1] - oy + s / 2 - 0.5) / s)


def wcs_per_geometria(w, geom, shape=None):
    """WCS dell'array descritto da 'geom' a partire dal WCS del file intero.
    Ritagli: si sposta CRPIX. Riduzioni: anche CD/CDELT e i coefficienti SIP
    (A_pq * s^(p+q-1)). Le tabelle di distorsione (HST flt) non si possono
    scalare e vengono tolte (errore tipico < 1 pixel originale)."""
    if w is None:
        return None
    from astropy.wcs import Sip
    w2 = w.deepcopy()
    s = geom.s
    cx, cy = crpix_per_geometria(w.wcs.crpix, geom.ox, geom.oy, s)
    w2.wcs.crpix = [cx, cy]
    if s != 1:
        if w2.wcs.has_cd():
            w2.wcs.cd = np.array(w.wcs.cd) * s
        else:
            w2.wcs.cdelt = np.array(w.wcs.cdelt) * s
        try:
            w2.cpdis1 = w2.cpdis2 = None
            w2.det2im1 = w2.det2im2 = None
        except Exception:
            pass
    if w.sip is not None:
        def scala(m):
            if m is None:
                return None
            m = np.array(m, dtype=float)
            out = m.copy()
            for p in range(m.shape[0]):
                for q in range(m.shape[1]):
                    out[p, q] = m[p, q] * s ** (p + q - 1)
            return out
        w2.sip = Sip(scala(w.sip.a), scala(w.sip.b), scala(w.sip.ap), scala(w.sip.bp), [cx, cy])
    w2.wcs.set()
    if shape is not None:
        w2.pixel_shape = (shape[1], shape[0])
    return w2


# =============================================================================
# REFERTO DI QUALITÀ
# =============================================================================
def misura_stelle(a, sigma, max_stelle=400):
    """Stelle isolate e non sature: FWHM (pixel), allungamento, numero."""
    from scipy import ndimage as ndi
    m, _ = O.statistiche(a)
    b = np.where(np.isfinite(a), a - m, 0).astype(np.float32)
    liscia = ndi.gaussian_filter(b, 1.0)
    picchi = (liscia == ndi.maximum_filter(liscia, size=9)) & (liscia > 10 * sigma)
    ys, xs = np.nonzero(picchi)
    if ys.size == 0:
        return dict(n=0)
    ny, nx = a.shape
    r = 6
    tieni = (ys > r) & (ys < ny - r - 1) & (xs > r) & (xs < nx - r - 1)
    ys, xs = ys[tieni], xs[tieni]
    vmax = np.nanmax(a) if np.isfinite(a).any() else 0
    ordine = np.argsort(-liscia[ys, xs])
    ys, xs = ys[ordine], xs[ordine]
    # niente le più brillanti (spesso sature) e niente vicine tra loro
    salta = min(len(ys) // 20, 20)
    ys, xs = ys[salta:], xs[salta:]
    fwhm, ell, snr = [], [], []
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1]
    for y, x in zip(ys[: max_stelle * 3], xs[: max_stelle * 3]):
        fin = a[y - r:y + r + 1, x - r:x + r + 1]
        if not np.isfinite(fin).all() or fin.max() >= 0.98 * vmax:
            continue
        f = np.clip(fin - m, 0, None).astype(np.float64)
        # una sola sorgente nella finestra
        if (f > 0.5 * f[r, r]).sum() > 0.5 * f.size:
            continue
        t = f.sum()
        if t <= 0:
            continue
        cx, cy = (f * dx).sum() / t, (f * dy).sum() / t
        if abs(cx) > 1.5 or abs(cy) > 1.5:
            continue
        mxx = (f * (dx - cx) ** 2).sum() / t
        myy = (f * (dy - cy) ** 2).sum() / t
        mxy = (f * (dx - cx) * (dy - cy)).sum() / t
        tr, det = mxx + myy, mxx * myy - mxy ** 2
        disc = math.sqrt(max(tr * tr / 4 - det, 0))
        l1, l2 = tr / 2 + disc, tr / 2 - disc
        if l2 <= 0:
            continue
        fwhm.append(2.3548 * math.sqrt((l1 + l2) / 2))
        snr.append(f[r, r] / sigma)
        ell.append(math.sqrt(l1 / l2))
        if len(fwhm) >= max_stelle:
            break
    if not fwhm:
        return dict(n=0)
    return dict(n=len(fwhm), fwhm_px=float(np.median(fwhm)), elongation=float(np.median(ell)),
                peak_snr=float(np.median(snr)))


def qualita(a, meta, geom=None, lineare=True):
    """Lista di voci del referto: (chiave, valore, testo valore, giudizio).
    Giudizio: 'good', 'warn', 'bad' o '' (solo informazione)."""
    geom = geom or Geometria()
    voci = []
    def voce(k, v, testo, giud=""):
        voci.append(dict(key=k, value=v, text=testo, verdict=giud))
    ny, nx = a.shape[:2]
    scala = (meta.get("pixel_arcsec") or 0) * geom.s if meta.get("pixel_arcsec") else None
    voce("mission", meta.get("mission"), f"{meta.get('mission')} {meta.get('instrument') or ''}".strip())
    voce("filter", meta.get("filter"), f"{meta.get('filter')} ({meta.get('wavelength')})")
    voce("size", (ny, nx), f"{nx} x {ny} px" + (f" (reduced x{geom.s:g})" if geom.s > 1 else ""))
    if scala:
        voce("pixel_scale", scala, f"{scala:.3f}\"/px")
        voce("field", (nx * scala / 60, ny * scala / 60), f"{nx * scala / 60:.2f}' x {ny * scala / 60:.2f}'")
    voce("wcs", meta.get("wcs"), "yes" if meta.get("wcs") else "no", "good" if meta.get("wcs") else "warn")
    voce("unit", meta.get("unit"), meta.get("unit") or "-", "" if meta.get("unit") else "warn")
    if meta.get("exposure_s"):
        voce("exposure", meta["exposure_s"], f"{meta['exposure_s']:.1f} s")
    if meta.get("date"):
        voce("date", meta["date"], meta["date"])
    if a.ndim == 3:
        return voci
    fin = np.isfinite(a)
    cop = float(fin.mean())
    voce("coverage", cop, f"{cop * 100:.1f}%", "good" if cop > 0.9 else "warn" if cop > 0.5 else "bad")
    if not fin.any():
        return voci
    y0, y1, x0, x1 = O.riquadro_dati(a)
    dentro = a[y0:y1, x0:x1]
    buchi = O.buchi_interni(dentro, 10 ** 6)
    fb = float(buchi.mean())
    voce("holes", fb, f"{fb * 100:.3f}%", "good" if fb < 1e-4 else "warn" if fb < 1e-2 else "bad")
    m, s_glob = O.statistiche(a)
    s = O.rumore_pixel(a)
    voce("background", m, f"{m:.4g}")
    voce("noise", s, f"{s:.4g}")
    if lineare:
        v = O.campione(a)
        din = float((np.percentile(v, 99.9) - m) / s) if v.size else 0
        voce("dynamic_range", din, f"{din:.0f} sigma")
        # gradiente: fondo su una griglia 8x8 (senza sorgenti)
        try:
            mask = O.maschera_sorgenti(a, m, s_glob)
            xs, ys, vs = O.campioni_griglia(a, mask, 8)
            if vs.size >= 6:
                g = float((np.percentile(vs, 95) - np.percentile(vs, 5)) / s)
                voce("gradient", g, f"{g:.1f} sigma", "good" if g < 3 else "warn" if g < 15 else "bad")
        except Exception:
            pass
        caldi, freddi = O.pixel_caldi(a if a.size < 9e6 else a[:3000, :3000], 8.0, s)
        fc = float((caldi.sum() + freddi.sum()) / max(1, caldi.size))
        voce("hot_pixels", fc, f"{fc * 100:.3f}%", "good" if fc < 1e-4 else "warn" if fc < 1e-3 else "bad")
        vmax = float(np.nanmax(a))
        sat = meta.get("saturate")
        lim = sat if sat else vmax
        fs = float((a >= 0.999 * lim).sum() / fin.sum()) if lim > m + 10 * s else 0.0
        if not sat and fs < 2e-6:
            fs = 0.0
        voce("saturation", fs, f"{fs * 100:.3f}%", "good" if fs < 1e-4 else "warn" if fs < 1e-3 else "bad")
        st = misura_stelle(a if a.size < 2.5e7 else a[ny // 2 - 2500: ny // 2 + 2500, nx // 2 - 2500: nx // 2 + 2500], s)
        voce("stars", st.get("n", 0), str(st.get("n", 0)))
        if st.get("n"):
            fw = st["fwhm_px"]
            testo = f"{fw:.2f} px" + (f" = {fw * scala:.2f}\"" if scala else "")
            giud = "good" if 1.5 <= fw <= 6 else "warn"
            voce("fwhm", fw, testo, giud)
            sn = st["peak_snr"]
            voce("star_snr", sn, f"{sn:.0f}", "good" if sn > 30 else "warn" if sn > 12 else "bad")
            el = st["elongation"]
            voce("elongation", el, f"{el:.2f}", "good" if el < 1.2 else "warn" if el < 1.5 else "bad")
    return voci


def referto_testo(voci, titolo):
    """Referto in inglese (file .txt)."""
    righe = [titolo, "=" * len(titolo)]
    for v in voci:
        g = {"good": "OK", "warn": "CHECK", "bad": "POOR"}.get(v["verdict"], "")
        righe.append(f"{v['key']:<15} {v['text']:<40} {g}")
    return "\n".join(righe)


# =============================================================================
# RICETTA
# =============================================================================
def ricetta_vuota(sorgente=None, zona=None, fattore=1):
    return dict(program=NOME, version=VERSIONE, source=str(sorgente.percorso) if sorgente else "",
                open_choices=sorgente.scelte if sorgente else {}, zone=list(zona) if zona else None,
                factor=fattore, steps=[])


def esegui(a0, geom0, passi, meta, da=0, cache=None, progresso=None, anteprima=False, ferma=None):
    """Esegue i passi dal numero 'da' usando la cache dei risultati precedenti.
    cache[i] = (array, geometria, lineare, avvisi, modello) DOPO il passo i.
    Restituisce la cache aggiornata."""
    cache = list(cache or [])[:da]
    if da == 0:
        a, geom, lineare, avvisi = a0, geom0.copia(), True, set()
    else:
        a, g, lineare, avvisi, _ = cache[da - 1]
        geom = g.copia()
    for i in range(da, len(passi)):
        if ferma and ferma():
            return None
        ctx = Contesto(geom, dict(meta), lineare, anteprima=anteprima)
        ctx.avvisi = set(avvisi)
        a = O.applica_passo(a, passi[i], ctx)
        geom, lineare, avvisi = ctx.geom.copia(), ctx.lineare, ctx.avvisi
        cache.append((a, geom.copia(), lineare, set(avvisi), ctx.modello))
        if progresso:
            progresso((i + 1 - da) / max(1, len(passi) - da))
    return cache


def stato_scienza(a0, geom0, passi, cache):
    """Ultimo stato lineare: prima del primo passo "display" acceso."""
    ultimo = (a0, geom0, True, set())
    for i, p in enumerate(passi):
        if not p.get("on", True):
            continue
        if O.OPERAZIONI[p["op"]].famiglia == "display":
            break
        if i < len(cache):
            a, g, lin, avv, _ = cache[i]
            if not lin:
                break
            ultimo = (a, g, lin, avv)
    return ultimo


def per_visione(a, lineare):
    """Array [0,1] per lo schermo (2D o HxWx3)."""
    if lineare:
        if a.ndim == 3:
            return np.stack([O.stira_auto(a[..., c]) for c in range(a.shape[2])], axis=2)
        return O.stira_auto(a)
    return np.clip(np.nan_to_num(a, nan=0.0), 0, 1)


# =============================================================================
# USCITE
# =============================================================================
CHIAVI_STRUTTURA = re.compile(r"^(SIMPLE|BITPIX|NAXIS\d*|EXTEND|PCOUNT|GCOUNT|XTENSION|BSCALE|BZERO|BLANK|"
                              r"CHECKSUM|DATASUM|ZIMAGE|ZSIMPLE|ZTENSION|ZEXTEND|ZBLOCKED|ZBITPIX|ZNAXIS\d*|"
                              r"ZTILE\d+|ZCMPTYPE|ZNAME\d+|ZVAL\d+|ZQUANTIZ|ZDITHER0|ZPCOUNT|ZGCOUNT|ZHECKSUM|"
                              r"ZDATASUM|ZBLANK|ZSCALE|ZZERO|ZMASKCMP|TFIELDS|TTYPE\d+|TFORM\d+|TUNIT\d+|TDIM\d+|"
                              r"END|EXTNAME|EXTVER|INHERIT|DATAMIN|DATAMAX|LTV\d|LTM\d_\d)$")
CHIAVI_WCS = re.compile(r"^(WCSAXES\w?|CRPIX\d\w?|CRVAL\d\w?|CDELT\d\w?|CTYPE\d\w?|CUNIT\d\w?|CROTA\d|"
                        r"CD\d_\d\w?|PC\d_\d\w?|PV\d_\d+\w?|PS\d_\d+\w?|LONPOLE\w?|LATPOLE\w?|RADESYS\w?|"
                        r"RADECSYS|EQUINOX\w?|EPOCH|A_ORDER|B_ORDER|AP_ORDER|BP_ORDER|A_\d_\d|B_\d_\d|"
                        r"AP_\d_\d|BP_\d_\d|A_DMAX|B_DMAX|CPDIS\d|CPERR\d|DP\d|DQ\d|D2IMDIS\d|D2IMERR\d|"
                        r"D2IM\d|D2IMEXT|NPOL\w*|WCSNAME\w?|MJDREF|CRDER\d|CSYER\d|"
                        r"OCRPIX\d|OCRVAL\d|OCD\d_\d|ORIENTAT|IDCTAB|WCSAXESO)$")


def intestazione_uscita(sorgente, w, storia, extra=None):
    """Intestazione con le parole chiave del file originale (non strutturali e non WCS),
    il WCS nuovo e la storia della lavorazione."""
    from astropy.io import fits
    h = fits.Header()
    for hdr in (sorgente.hdr_p, sorgente.hdr_s):
        for card in hdr.cards:
            k = card.keyword
            if not k or k in ("COMMENT", "HISTORY", "") or CHIAVI_STRUTTURA.match(k) or CHIAVI_WCS.match(k):
                continue
            try:
                h[k] = (card.value, card.comment)
            except Exception:
                pass
    if w is not None:
        h.extend(w.to_header(relax=True), update=True)
    h["CREATOR"] = (f"{NOME} {VERSIONE}", "software that created this file")
    h["FLABVER"] = (VERSIONE, "FITS Lab version")
    h["AUTHOR"] = (AUTORE, "author of the software")
    h["DATE"] = (datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), "file creation date")
    h["FLABSRC"] = (ascii_fits(Path(getattr(sorgente, "originale", sorgente.percorso)).name)[:60], "source file")
    for k, v in (extra or {}).items():
        h[k] = v
    h.add_comment(f"{COPYRIGHT_ASCII} - created with {NOME} {VERSIONE}")
    for s in storia:
        testo = ascii_fits(s)
        while testo:
            h.add_history(testo[:70])
            testo = testo[70:]
    return h


def scrivi_fits(percorso, dati, h):
    from astropy.io import fits
    percorso = Path(percorso)
    percorso.parent.mkdir(parents=True, exist_ok=True)
    if dati.ndim == 3:                      # RGB HxWx3 -> cubo 3 x H x W
        dati = np.moveaxis(dati, 2, 0)
    fits.PrimaryHDU(np.asarray(dati, dtype=np.float32), header=h).writeto(percorso, overwrite=True)
    return percorso


FORMATI = {
    "science_fits": ("science", ".fits"),
    "aesthetic_fits": ("aesthetic", ".fits"),
    "png": ("aesthetic", ".png"),
    "tiff16": ("aesthetic", ".tif"),
    "jpeg": ("aesthetic", ".jpg"),
    "forum": ("aesthetic", "_forum.jpg"),
}
NOMI_PIL = {"png": "PNG 8 bit", "tiff16": "TIFF 16 bit", "jpeg": "JPEG", "forum": "JPEG per forum"}


def storia_ricetta(sorgente, ricetta, fino_a=None):
    storia = [f"{NOME} {VERSIONE} {datetime.now():%Y-%m-%d %H:%M}"]
    if sorgente.nota_apertura:
        storia.append("open: " + sorgente.nota_apertura)
    if ricetta.get("zone"):
        y0, y1, x0, x1 = ricetta["zone"]
        storia.append(f"zone: x {x0}-{x1}, y {y0}-{y1} (original pixels)")
    for i, p in enumerate(ricetta["steps"]):
        if fino_a is not None and i >= fino_a:
            break
        if not p.get("on", True):
            continue
        op = O.OPERAZIONI[p["op"]]
        par = op.predefiniti()
        par.update(p.get("params", {}))
        storia.append(f"step {i + 1}: {op.descrivi(par)}")
    return storia


def elabora_e_salva(sorgente, ricetta, cartella, formati, firma=True, fattore=1, progresso=None, log=print):
    """Rifà la ricetta sul file (zona scelta, fattore dato: 1 = piena qualità) e
    salva le uscite. Restituisce l'elenco dei file scritti."""
    zona = ricetta.get("zone")
    ny, nx = sorgente.shape
    z = zona or (0, ny, 0, nx)
    log(f"reading {z} x{fattore}")
    a0 = sorgente.leggi(z, fattore, progresso=(lambda f: progresso(0.25 * f)) if progresso else None)
    geom0 = Geometria(z[0], z[2], fattore, 1)
    meta = sorgente.meta()
    passi = ricetta["steps"]
    cache = esegui(a0, geom0, passi, meta, progresso=(lambda f: progresso(0.25 + 0.55 * f)) if progresso else None)
    return salva_uscite(sorgente, ricetta, a0, geom0, passi, cache, cartella, formati, firma, progresso, log)


def salva_uscite(sorgente, ricetta, a0, geom0, passi, cache, cartella, formati, firma=True, progresso=None, log=print):
    cartella = Path(cartella)
    base = re.sub(r"[^A-Za-z0-9_.+-]+", "_", Path(getattr(sorgente, "originale", sorgente.percorso)).name)
    for est in sorted(ESTENSIONI, key=len, reverse=True):
        if base.lower().endswith(est):
            base = base[: -len(est)]
            break
    scritti = []
    meta = sorgente.meta()
    # --- scienza: ultimo stato lineare
    a_sci, g_sci, _, avv = stato_scienza(a0, geom0, passi, cache)
    n_sci = 0
    for i, p in enumerate(passi):
        if p.get("on", True) and O.OPERAZIONI[p["op"]].famiglia == "display":
            break
        n_sci = i + 1
    storia_sci = storia_ricetta(sorgente, ricetta, n_sci)
    if "science_fits" in formati:
        w = wcs_per_geometria(sorgente.wcs, g_sci, a_sci.shape[:2])
        extra = {"FLABTYPE": ("science", "linear data, still measurable")}
        if "denoised" in avv:
            extra["FLABDNS"] = (True, "noise reduced: pixel errors no longer valid")
        if "deconvolved" in avv:
            extra["FLABDCV"] = (True, "deconvolved: flux kept, PSF and errors changed")
        if g_sci.s > 1:
            extra["FLABBIN"] = (g_sci.s, "original pixels per pixel")
        if sorgente.distorsioni_tabella and (g_sci.s != 1):
            storia_sci.append("lookup-table distortion removed (cannot be rescaled)")
        h = intestazione_uscita(sorgente, w, storia_sci, extra)
        p = scrivi_fits(cartella / "science" / f"{base}_science.fits", a_sci, h)
        scritti.append(p)
        log(f"written {p}")
    # --- estetica: ultimo stato (stirato se non c'è uno stiramento)
    if cache:
        a_fin, g_fin, lin_fin, _, _ = cache[-1]
    else:
        a_fin, g_fin, lin_fin = a0, geom0, True
    vis = per_visione(a_fin, lin_fin)
    storia_est = storia_ricetta(sorgente, ricetta)
    if lin_fin:
        storia_est.append("display: automatic STF stretch")
    if "aesthetic_fits" in formati:
        w = wcs_per_geometria(sorgente.wcs, g_fin, vis.shape[:2])
        h = intestazione_uscita(sorgente, w, storia_est, {"FLABTYPE": ("aesthetic", "stretched, for display only")})
        h["BUNIT"] = ("", "display values 0-1")
        p = scrivi_fits(cartella / "aesthetic" / f"{base}_aesthetic.fits", vis, h)
        scritti.append(p)
    M = _M()
    for f in ("png", "tiff16", "jpeg", "forum"):
        if f in formati:
            sotto, est = FORMATI[f]
            p = cartella / sotto / f"{base}_aesthetic{est}"
            p.parent.mkdir(parents=True, exist_ok=True)
            if f == "tiff16" and vis.ndim == 3:
                _tiff16_rgb(np.flipud(vis), p, firma, M)
            else:
                M.salva_immagine(vis.copy(), p, NOMI_PIL[f], firma=firma)
            scritti.append(p)
    # --- ricetta e referto
    cartella.mkdir(parents=True, exist_ok=True)
    pr = cartella / f"{base}_recipe.json"
    pr.write_text(json.dumps(ricetta, indent=1, ensure_ascii=False), encoding="utf-8")
    scritti.append(pr)
    try:
        q0 = qualita(a0, meta, geom0)
        q1 = qualita(a_sci, meta, g_sci)
        testo = referto_testo(q0, f"QUALITY - original ({base})") + "\n\n" + \
            referto_testo(q1, "QUALITY - science result") + "\n\nSteps:\n" + "\n".join(storia_est) + \
            f"\n\n{COPYRIGHT_ASCII}\n"
        pq = cartella / f"{base}_quality.txt"
        pq.write_text(testo, encoding="utf-8")
        scritti.append(pq)
    except Exception as e:
        log(f"quality report not written: {e}")
    if progresso:
        progresso(1.0)
    return scritti


def _tiff16_rgb(a, percorso, firma, M):
    """TIFF RGB a 16 bit (senza tifffile: scrittura diretta non compressa)."""
    a = np.clip(np.nan_to_num(a), 0, 1).astype(np.float32)
    if firma:
        a = np.ascontiguousarray(a)
        M.applica_firma(a)
    try:
        import tifffile
        tifffile.imwrite(percorso, (a * 65535 + 0.5).astype(np.uint16), photometric="rgb")
        return
    except ImportError:
        pass
    import struct
    a16 = (a * 65535 + 0.5).astype("<u2")
    h, w, _ = a16.shape
    dati = a16.tobytes()
    voci = [(256, 4, 1, w), (257, 4, 1, h), (258, 3, 3, None), (259, 3, 1, 1), (262, 3, 1, 2),
            (273, 4, 1, None), (277, 3, 1, 3), (278, 4, 1, h), (279, 4, 1, len(dati)), (284, 3, 1, 1)]
    off_ifd = 8
    n = len(voci)
    off_bps = off_ifd + 2 + n * 12 + 4
    off_dati = off_bps + 6
    out = bytearray(b"II*\x00" + struct.pack("<I", off_ifd) + struct.pack("<H", n))
    for tag, tipo, cnt, val in voci:
        if tag == 258:
            out += struct.pack("<HHII", tag, tipo, cnt, off_bps)
        elif tag == 273:
            out += struct.pack("<HHII", tag, tipo, cnt, off_dati)
        elif tipo == 3:
            out += struct.pack("<HHIHH", tag, tipo, cnt, val, 0)
        else:
            out += struct.pack("<HHII", tag, tipo, cnt, val)
    out += struct.pack("<I", 0) + struct.pack("<HHH", 16, 16, 16) + dati
    Path(percorso).write_bytes(bytes(out))


# =============================================================================
# OGGETTI NEL CAMPO E NOMI
# =============================================================================
def oggetti_nel_campo(w, shape, timeout=60, massimo=60):
    """SIMBAD + NED in parallelo, poi i Messier/NGC/IC del catalogo incluso se la rete manca."""
    from concurrent.futures import ThreadPoolExecutor
    M = _M()
    with ThreadPoolExecutor(2) as ex:
        fs = ex.submit(M.oggetti_simbad, w, shape, timeout)
        fn = ex.submit(M.oggetti_ned, w, shape, timeout)
        errori = []
        try:
            sim = fs.result()
        except Exception as e:
            sim, _ = [], errori.append(f"SIMBAD: {e}")
        try:
            ned = fn.result()
        except Exception as e:
            ned, _ = [], errori.append(f"NED: {e}")
    try:
        famosi = oggetti_famosi(w, shape, timeout)
    except Exception as e:
        famosi = []
        errori.append(f"SIMBAD (famous objects): {e}")
    if famosi:
        sim = M.unisci_oggetti(famosi, sim, 5.0) if sim else famosi
    tutti = M.unisci_oggetti(sim, ned, 2.0, prendi_z=True) if sim else ned
    if not tutti:
        try:
            tutti = M.oggetti_openngc(w, shape)
        except Exception as e:
            errori.append(f"OpenNGC: {e}")
    return ordina_oggetti(tutti)[:massimo], errori


def oggetti_famosi(w, shape, timeout=60, minimo_articoli=40):
    """Solo gli oggetti molto studiati del campo (M 51, NGC 5195...). Serve perché lo
    specchio SIMBAD di Harvard accetta solo la domanda semplice senza ORDER BY: nei campi
    affollati le prime 3000 righe possono non contenere l'oggetto principale."""
    import requests
    M = _M()
    ny, nx = shape
    ra0, de0, raggio = M._campo(w, shape)
    q = ("SELECT TOP 500 oid, main_id, ra, dec, otype, nbref FROM basic "
         f"WHERE CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra0:.6f}, {de0:.6f}, {raggio:.5f})) = 1 "
         f"AND nbref >= {int(minimo_articoli)}")
    s = requests.Session()
    s.headers["User-Agent"] = f"FITSLab/{VERSIONE} (Python)"
    s.request = M._con_timeout(s.request, timeout)
    righe, errori = None, []
    for url in M.URL_SIMBAD_TAP:
        try:
            righe = M._tap_righe(url, q, s, 500)
            break
        except Exception as e:
            errori.append(f"{url.split('/')[2]}: {e}")
    if righe is None:
        raise ConnectionError("; ".join(errori))
    out = []
    for r in righe:
        try:
            ra, de = float(r["ra"]), float(r["dec"])
            nb = int(float(r.get("nbref") or 0))
        except (TypeError, ValueError, KeyError):
            continue
        x, y = (float(np.ravel(v)[0]) for v in M.radec_a_pix(w, ra, de))
        if not (0 <= x < nx and 0 <= y < ny):
            continue
        nome = re.sub(r"^NAME\s+", "", re.sub(r"\s+", " ", str(r["main_id"]).strip()))
        out.append(dict(nome=nome, nome_simbad=nome, altri_nomi="", ra=ra, dec=de, x=x, y=y,
                        tipo=str(r.get("otype", "")).strip(), riferimenti=nb, catalogo=bool(_CATALOGO.match(nome)),
                        magnitudine=None, fonte="SIMBAD"))
    out.sort(key=lambda o: -o["riferimenti"])
    return out


_CATALOGO = re.compile(r"^(M|NGC|IC|UGC|PGC|Sh\s?2-|Abell|Arp|Mrk|Cl|Collinder|Melotte|Barnard|B|LDN|vdB|RCW|Gum)"
                       r"\s*\d+[A-Za-z]?$")


def ordina_oggetti(oggetti):
    """Prima gli oggetti principali con un nome di catalogo vero («M 51», «NGC 5195»),
    poi i nomi comuni, poi il resto per numero di articoli. Le sottoregioni come
    «NGC 5194:[CCM69] 095» (regioni HII dentro M 51) non contano come catalogo."""
    def nome_migliore(o):
        nomi = [o.get("nome", "")] + [x.strip() for x in str(o.get("altri_nomi", "")).split(";")]
        nomi = [re.sub(r"^NAME\s+", "", re.sub(r"\s+", " ", n)) for n in nomi if n]
        for n in nomi:
            if _CATALOGO.match(n):
                return n, 0
        for n in nomi:
            if ":" not in n and "[" not in n and not re.search(r"\d{3,}", n):
                return n, 1
        return (nomi[0] if nomi else "?"), 2
    out = []
    for o in oggetti:
        o = dict(o)
        o["nome"], o["rango"] = nome_migliore(o)
        out.append(o)
    def articoli(o):
        try:
            return float(o.get("riferimenti") or o.get("nrefs") or 0)
        except (TypeError, ValueError):
            return 0.0
    out.sort(key=lambda o: (o["rango"], -articoli(o)))
    return out


def risolvi_nome(nome, timeout=20):
    """Nome -> (ra, dec) in gradi con NED ObjectLookup (raggiungibile dalla rete di Edoardo)."""
    import requests
    r = requests.post("https://ned.ipac.caltech.edu/srs/ObjectLookup", json={"name": {"v": nome}},
                      timeout=timeout, headers={"User-Agent": f"FITSLab/{VERSIONE}"})
    r.raise_for_status()
    d = r.json()
    pos = (d.get("Preferred") or {}).get("Position") or {}
    if "RA" in pos and "Dec" in pos:
        return float(pos["RA"]), float(pos["Dec"])
    raise ValueError(f"name not found: {nome}")
