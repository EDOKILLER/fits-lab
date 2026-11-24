# -*- coding: utf-8 -*-
"""
ROMAN ASDF - lettura dei file del Nancy Grace Roman Space Telescope
==================================================================
I dati di Roman (strumento WFI) non sono FITS ma ASDF: un'intestazione di
testo (YAML) con tutti i metadati, seguita da blocchi binari con le immagini.
Questo modulo legge i file .asdf e li converte in FITS, così tutto il resto di
AutoAstroPipe (rumore, fondo, stretching, stack, RGB, analisi) li tratta come
gli altri file.

Due modi di lettura:
  1. con le librerie ufficiali (asdf + gwcs), se sono installate: le coordinate
     vengono dal modello di distorsione completo (GWCS), convertito in un WCS
     FITS con correzione SIP;
  2. con il lettore interno (solo numpy e PyYAML), se mancano: i dati sono gli
     stessi, le coordinate vengono dal contorno del campo (s_region) e sono
     approssimate (niente correzione della distorsione).

Livelli dei prodotti Roman:
  L1  _uncal   rampe grezze (3D)            -> non adatte, vengono saltate
  L2  _cal     un rivelatore calibrato (2D)  -> sì (18 rivelatori per esposizione)
  L3  _coadd   mosaici combinati (2D)        -> sì

© 2026 Edoardo Battistel - AutoAstroPipe versione 2.2
"""

import bz2
import hashlib
import math
import re
import struct
import zlib
from pathlib import Path

import numpy as np

ESTENSIONI_ASDF = (".asdf",)
MAGIA_BLOCCO = b"\xd3BLK"
DO_NOT_USE = 1                      # bit 0 della maschera di qualità (dq) di Roman

# lunghezze d'onda centrali dei filtri del WFI (micron)
FILTRI_ROMAN_UM = {"F062": 0.620, "F087": 0.869, "F106": 1.060, "F129": 1.293, "F146": 1.464,
                   "F158": 1.577, "F184": 1.842, "F213": 2.125}
ELEMENTI_NON_IMMAGINE = ("GRISM", "PRISM", "DARK", "BLANK")


def e_asdf(p):
    return Path(p).name.lower().endswith(ESTENSIONI_ASDF)


# =============================================================================
# LETTORE INTERNO (senza la libreria asdf)
# =============================================================================
class _Etichettato(dict):
    """Nodo YAML con un'etichetta ASDF (es. core/ndarray-1.0.0)."""
    tag = ""


class _TestoEtichettato(str):
    tag = ""


class _ListaEtichettata(list):
    tag = ""


def _caricatore_yaml():
    import yaml
    base = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

    class Caricatore(base):
        pass

    def costruisci(loader, _suffisso, nodo):
        if isinstance(nodo, yaml.MappingNode):
            v = _Etichettato(loader.construct_mapping(nodo, deep=True))
        elif isinstance(nodo, yaml.SequenceNode):
            v = _ListaEtichettata(loader.construct_sequence(nodo, deep=True))
        else:
            v = _TestoEtichettato(loader.construct_scalar(nodo))
        v.tag = nodo.tag
        return v

    Caricatore.add_multi_constructor("", costruisci)
    return Caricatore


def _lz4_blocco(src, dimensione):
    """Decompressione di un blocco LZ4 (formato 'block') in puro Python.
    Usata solo se manca il pacchetto lz4: più lenta, ma senza dipendenze."""
    dst = bytearray(dimensione)
    i = o = 0
    n = len(src)
    while i < n:
        tok = src[i]
        i += 1
        lit = tok >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        dst[o:o + lit] = src[i:i + lit]
        i += lit
        o += lit
        if i >= n:
            break
        off = src[i] | (src[i + 1] << 8)
        i += 2
        ml = tok & 15
        if ml == 15:
            while True:
                b = src[i]
                i += 1
                ml += b
                if b != 255:
                    break
        ml += 4
        s = o - off
        if off >= ml:
            dst[o:o + ml] = dst[s:s + ml]
        else:                                   # copia sovrapposta: il motivo si ripete
            motivo = bytes(dst[s:o])
            dst[o:o + ml] = (motivo * (ml // off + 1))[:ml]
        o += ml
    return bytes(dst[:o])


def _decomprimi_lz4_asdf(buf):
    """Formato lz4 di ASDF: pezzi [4 byte big-endian = lunghezza][blocco lz4 con
    i 4 byte little-endian della dimensione originale in testa]."""
    try:
        import lz4.block as lz4b
    except ImportError:
        lz4b = None
    out = []
    i = 0
    n = len(buf)
    while i < n:
        lung = struct.unpack(">I", buf[i:i + 4])[0]
        i += 4
        pezzo = bytes(buf[i:i + lung])
        i += lung
        if lz4b is not None:
            out.append(lz4b.decompress(pezzo))
        else:
            dim = struct.unpack("<I", pezzo[:4])[0]
            out.append(_lz4_blocco(pezzo[4:], dim))
    return b"".join(out)


_TIPI = {"int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2", "int32": "i4", "uint32": "u4",
         "int64": "i8", "uint64": "u8", "float16": "f2", "float32": "f4", "float64": "f8",
         "complex64": "c8", "complex128": "c16", "bool8": "?"}


class _FileAsdfInterno:
    """Legge struttura e blocchi binari di un file ASDF senza la libreria asdf."""

    def __init__(self, percorso):
        self.percorso = Path(percorso)
        self.dimensione_file = self.percorso.stat().st_size
        import yaml
        testo, self.inizio_blocchi = self._intestazione()
        self.albero = yaml.load(testo, Loader=_caricatore_yaml())
        if not isinstance(self.albero, dict):
            raise ValueError("struttura ASDF non valida")
        self._blocchi = None
        self._cache = {}

    def _intestazione(self):
        letti = bytearray()
        inizio_blocchi = None
        with open(self.percorso, "rb") as f:
            while True:
                pezzo = f.read(1 << 20)
                if not pezzo:
                    break
                cerca_da = max(0, len(letti) - 4)
                letti += pezzo
                k = letti.find(MAGIA_BLOCCO, cerca_da)
                if k >= 0:
                    inizio_blocchi = k
                    break
                if len(letti) > (256 << 20):
                    raise ValueError("intestazione ASDF troppo lunga")
        if not letti.startswith(b"#ASDF"):
            raise ValueError("non è un file ASDF (manca '#ASDF' all'inizio)")
        testa = bytes(letti[:inizio_blocchi] if inizio_blocchi is not None else letti)
        a = testa.find(b"%YAML")
        if a < 0:
            raise ValueError("file ASDF senza albero YAML")
        fine = None
        for m in re.finditer(rb"\r?\n\.\.\.\r?\n", testa):
            fine = m.end()
        testo = testa[a:fine] if fine else testa[a:]
        return testo.decode("utf-8", "replace"), inizio_blocchi

    def blocchi(self):
        if self._blocchi is None:
            self._blocchi = []
            pos = self.inizio_blocchi
            with open(self.percorso, "rb") as f:
                while pos is not None and pos + 6 <= self.dimensione_file:
                    f.seek(pos)
                    if f.read(4) != MAGIA_BLOCCO:
                        break
                    hs = struct.unpack(">H", f.read(2))[0]
                    h = f.read(hs)
                    flags, comp, alloc, usati, dim = struct.unpack(">I4sQQQ", h[:32])
                    inizio = pos + 6 + hs
                    if flags & 1:                       # blocco in streaming: fino alla fine del file
                        usati = dim = self.dimensione_file - inizio
                        alloc = usati
                    self._blocchi.append(dict(inizio=inizio, compressione=comp, usati=usati, dimensione=dim))
                    if flags & 1:
                        break
                    pos = inizio + alloc
        return self._blocchi

    def dati_blocco(self, indice):
        b = self.blocchi()[indice]
        comp = b["compressione"]
        if comp in (b"\0\0\0\0", b"    "):
            if b["usati"] == 0:
                return np.zeros(0, np.uint8)
            return np.memmap(self.percorso, dtype=np.uint8, mode="r", offset=b["inizio"], shape=(b["usati"],))
        if indice in self._cache:
            return self._cache[indice]
        with open(self.percorso, "rb") as f:
            f.seek(b["inizio"])
            grezzi = f.read(b["usati"])
        if comp == b"zlib":
            d = zlib.decompress(grezzi)
        elif comp == b"bzp2":
            d = bz2.decompress(grezzi)
        elif comp in (b"lz4\0", b"lz4 ", b"lz4"):
            d = _decomprimi_lz4_asdf(grezzi)
        else:
            raise ValueError(f"compressione ASDF non supportata: {comp!r} (installa il pacchetto 'asdf')")
        a = np.frombuffer(d, dtype=np.uint8)
        self._cache[indice] = a
        return a

    def array(self, nodo):
        """Converte un nodo core/ndarray in un array numpy (senza copiarlo, se possibile)."""
        if "data" in nodo and nodo.get("source") is None:
            return np.array(nodo["data"])
        sorgente = nodo.get("source")
        if isinstance(sorgente, str):
            raise ValueError(f"array in un file esterno ({sorgente}): non supportato")
        tipo = nodo.get("datatype", "float32")
        if not isinstance(tipo, str) or tipo not in _TIPI:
            raise ValueError(f"tipo di dato non supportato: {tipo}")
        ordine = ">" if str(nodo.get("byteorder", "little")) == "big" else "<"
        dt = np.dtype(ordine + _TIPI[tipo]) if _TIPI[tipo] != "?" else np.dtype("?")
        buf = self.dati_blocco(int(sorgente))
        offset = int(nodo.get("offset", 0) or 0)
        forma = list(nodo.get("shape", []))
        if "*" in [str(s) for s in forma]:
            resto = int(np.prod([int(s) for s in forma if str(s) != "*"]) or 1)
            forma = [(buf.size - offset) // dt.itemsize // resto if str(s) == "*" else int(s) for s in forma]
        forma = tuple(int(s) for s in forma)
        passi = nodo.get("strides")
        if passi:
            return np.ndarray(forma, dtype=dt, buffer=buf, offset=offset, strides=tuple(int(s) for s in passi))
        n = int(np.prod(forma)) if forma else 1
        return np.frombuffer(buf, dtype=dt, count=n, offset=offset).reshape(forma)

    def chiudi(self):
        self._cache.clear()
        self._blocchi = None


# =============================================================================
# ACCESSO UNIFORME ALL'ALBERO (libreria asdf o lettore interno)
# =============================================================================
def _prendi(nodo, chiave):
    if nodo is None:
        return None
    try:
        return nodo[chiave]
    except (KeyError, TypeError, IndexError, AttributeError):
        pass
    except Exception:
        pass
    try:
        return getattr(nodo, chiave)
    except Exception:
        return None


def _valore(nodo, *percorso):
    for k in percorso:
        nodo = _prendi(nodo, k)
        if nodo is None:
            return None
    return nodo


def _e_mappa(nodo):
    return isinstance(nodo, dict) or (hasattr(nodo, "keys") and hasattr(nodo, "__getitem__"))


def _cerca(nodo, chiave, profondita=6):
    """Primo valore con questo nome a qualsiasi profondità dell'albero dei metadati."""
    if nodo is None or profondita < 0 or not _e_mappa(nodo):
        return None
    try:
        chiavi = list(nodo.keys())
    except Exception:
        return None
    if chiave in chiavi:
        v = _prendi(nodo, chiave)
        if v is not None:
            return v
    for k in chiavi:
        if k in ("wcs", "data", "dq", "err", "context", "weight"):
            continue
        v = _prendi(nodo, k)
        if _e_mappa(v):
            r = _cerca(v, chiave, profondita - 1)
            if r is not None:
                return r
    return None


def _testo(v):
    """Valore dei metadati come testo semplice (gestisce Time, enumerazioni, quantità)."""
    if v is None:
        return ""
    if hasattr(v, "isot"):
        try:
            return str(v.isot)
        except Exception:
            pass
    if isinstance(v, dict):
        for k in ("value", "base"):
            if k in v:
                return _testo(v[k])
    if hasattr(v, "value") and not isinstance(v, (str, bytes)):
        try:
            return str(v.value)
        except Exception:
            pass
    return str(v).strip()


def _numero(v):
    if v is None:
        return None
    if hasattr(v, "value") and not isinstance(v, (str, bytes, dict)):
        v = v.value
    if isinstance(v, dict):
        v = v.get("value")
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _e_ndarray_interno(n):
    return isinstance(n, _Etichettato) and "ndarray" in n.tag


def _e_quantita_interna(n):
    return isinstance(n, _Etichettato) and "quantity" in n.tag


def unita_lunghezza_onda_filtro(filtro):
    f = str(filtro or "").upper().strip()
    m = re.match(r"^F(\d{3})", f)
    if not m:
        return None
    return FILTRI_ROMAN_UM.get(f"F{m.group(1)}", int(m.group(1)) / 100.0)


# =============================================================================
# COORDINATE (WCS)
# =============================================================================
def _proietta_tan(ra, dec, ra0, dec0):
    """Coordinate standard (ξ, η) in gradi della proiezione gnomonica attorno a (ra0, dec0)."""
    a, d, a0, d0 = map(np.radians, (ra, dec, ra0, dec0))
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - a0)
    xi = np.cos(d) * np.sin(a - a0) / cosc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - a0)) / cosc
    return np.degrees(xi), np.degrees(eta)


def _deproietta_tan(xi, eta, ra0, dec0):
    x, y = np.radians(xi), np.radians(eta)
    a0, d0 = np.radians(ra0), np.radians(dec0)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.where(rho == 0, d0, np.arcsin(np.cos(c) * np.sin(d0) + y * np.sin(c) * np.cos(d0) / np.where(rho == 0, 1, rho)))
        ra = a0 + np.arctan2(x * np.sin(c), rho * np.cos(d0) * np.cos(c) - y * np.sin(d0) * np.sin(c))
    return np.degrees(ra) % 360, np.degrees(dec)


def vertici_s_region(s_region):
    """Vertici (ra, dec) di un 'POLYGON ICRS ra1 dec1 ra2 dec2 ...'."""
    t = str(s_region or "")
    if "POLYGON" not in t.upper():
        return None
    nums = [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", t)]
    if len(nums) < 6 or len(nums) % 2:
        return None
    v = np.array(nums, dtype=np.float64).reshape(-1, 2)
    if len(v) > 3 and np.allclose(v[0], v[-1]):
        v = v[:-1]
    return v


def wcs_da_s_region(s_region, forma):
    """WCS TAN approssimato dai 4 vertici del campo (s_region). I vertici sono
    nell'ordine di gwcs.footprint: (x0,y0), (x0,y1), (x1,y1), (x1,y0).
    Restituisce (dizionario di parole chiave FITS, errore massimo in pixel)."""
    v = vertici_s_region(s_region)
    if v is None or len(v) != 4:
        return None, None
    ny, nx = forma
    pix = np.array([[-0.5, -0.5], [-0.5, ny - 0.5], [nx - 0.5, ny - 0.5], [nx - 0.5, -0.5]])
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    # punto di tangenza iniziale: baricentro dei vertici sulla sfera
    a, d = np.radians(v[:, 0]), np.radians(v[:, 1])
    s = np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)]).mean(axis=1)
    ra0 = math.degrees(math.atan2(s[1], s[0])) % 360
    dec0 = math.degrees(math.atan2(s[2], math.hypot(s[0], s[1])))
    A = np.column_stack([np.ones(4), pix[:, 0] - cx, pix[:, 1] - cy])
    for _ in range(3):
        xi, eta = _proietta_tan(v[:, 0], v[:, 1], ra0, dec0)
        cxi, *_ = np.linalg.lstsq(A, xi, rcond=None)
        ceta, *_ = np.linalg.lstsq(A, eta, rcond=None)
        ra0, dec0 = [float(t) for t in _deproietta_tan(cxi[0], ceta[0], ra0, dec0)]
    xi, eta = _proietta_tan(v[:, 0], v[:, 1], ra0, dec0)
    cxi, *_ = np.linalg.lstsq(A, xi, rcond=None)
    ceta, *_ = np.linalg.lstsq(A, eta, rcond=None)
    cd = np.array([[cxi[1], cxi[2]], [ceta[1], ceta[2]]])
    res = np.hypot(A @ cxi - xi, A @ ceta - eta)
    scala = math.sqrt(abs(np.linalg.det(cd)))
    errore_px = float(res.max() / scala) if scala > 0 else float("nan")
    h = {"WCSAXES": 2, "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN", "CUNIT1": "deg", "CUNIT2": "deg",
         "CRPIX1": cx + 1.0, "CRPIX2": cy + 1.0, "CRVAL1": ra0, "CRVAL2": dec0,
         "CD1_1": float(cd[0, 0]), "CD1_2": float(cd[0, 1]), "CD2_1": float(cd[1, 0]), "CD2_2": float(cd[1, 1]),
         "RADESYS": "ICRS"}
    return h, errore_px


def _wcs_da_gwcs(gw, forma, log=None):
    """Converte un GWCS in un'intestazione FITS con correzione SIP (gwcs.to_fits_sip)."""
    if gw is None or not hasattr(gw, "to_fits_sip"):
        return None, None
    ny, nx = forma
    bbox = None
    try:
        if getattr(gw, "bounding_box", None) is None:
            bbox = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))
    except Exception:
        bbox = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))
    for errore in (0.05, 0.25, 1.0):
        try:
            h = gw.to_fits_sip(bounding_box=bbox, max_pix_error=errore, degree=range(1, 7),
                               max_inv_pix_error=errore, inv_degree=range(1, 7), npoints=32)
            return h, errore
        except TypeError:
            break                                  # versione di gwcs con altri parametri
        except Exception as e:
            if log:
                log(f"  gwcs -> SIP con errore {errore} px non riuscito: {type(e).__name__}")
    try:
        h = gw.to_fits_sip(bounding_box=bbox) if bbox is not None else gw.to_fits_sip()
        return h, 0.25
    except Exception as e:
        if log:
            log(f"  gwcs -> SIP non riuscito: {type(e).__name__}: {e}")
    return None, None


# =============================================================================
# FILE ROMAN
# =============================================================================
class FileRoman:
    """Un file .asdf di Roman: metadati, immagine e coordinate."""

    def __init__(self, percorso, usa_libreria=True):
        self.percorso = Path(percorso)
        self._af = None
        self._interno = None
        self.lettore = "interno"
        if usa_libreria:
            try:
                import asdf
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")        # etichette Roman sconosciute senza roman_datamodels
                    self._af = asdf.open(self.percorso, lazy_load=True, memmap=False)
                self.albero = self._af.tree
                self.lettore = f"asdf {getattr(asdf, '__version__', '')}".strip()
            except ImportError:
                self._af = None
            except Exception:
                self._af = None
        if self._af is None:
            self._interno = _FileAsdfInterno(self.percorso)
            self.albero = self._interno.albero
        self.radice = _prendi(self.albero, "roman")
        if self.radice is None:
            # file simulati con uno schema più semplice: l'immagine può stare in cima all'albero
            self.radice = self.albero
        self.meta = _prendi(self.radice, "meta") or {}

    # ------------------------------------------------------------ array
    def _nodo_dati(self):
        n = _prendi(self.radice, "data")
        if n is None:
            raise ValueError("nel file non c'è un'immagine ('data')")
        return n

    def _array(self, nodo):
        if self._interno is not None:
            if _e_quantita_interna(nodo):
                nodo = nodo.get("value")
            if _e_ndarray_interno(nodo):
                return self._interno.array(nodo)
            return np.asarray(nodo)
        if hasattr(nodo, "unit") and hasattr(nodo, "value"):
            nodo = nodo.value
        return nodo

    def unita(self):
        n = _prendi(self.radice, "data")
        if self._interno is not None and _e_quantita_interna(n):
            return _testo(n.get("unit"))
        if hasattr(n, "unit"):
            try:
                return str(n.unit)
            except Exception:
                pass
        for k in ("unit", "bunit"):
            u = _cerca(self.meta, k)
            if u is not None and not _e_mappa(u):
                return _testo(u)
        # prodotti L2 di romancal: immagine in DN/s, con il fattore per MJy/sr in meta.photometry
        if _valore(self.meta, "photometry", "conversion_megajanskys") is not None and \
                str(_testo(_cerca(self.meta, "product_type"))).lower() == "l2":
            return "DN/s"
        return ""

    def forma(self):
        n = self._nodo_dati()
        if self._interno is not None:
            if _e_quantita_interna(n):
                n = n.get("value")
            if _e_ndarray_interno(n):
                return tuple(int(s) for s in n.get("shape", []))
        return tuple(int(s) for s in np.shape(self._array(n)))

    def immagine(self):
        """Immagine 2D float32; i pixel segnati DO_NOT_USE nella maschera dq diventano NaN."""
        d = np.array(self._array(self._nodo_dati()), dtype=np.float32)
        if d.ndim == 3 and d.shape[0] == 1:
            d = d[0]
        if d.ndim != 2:
            raise ValueError(f"immagine con {d.ndim} dimensioni {d.shape}: servono i prodotti calibrati "
                             f"(_cal di livello 2 o mosaici di livello 3), non le rampe grezze (_uncal)")
        dq = _prendi(self.radice, "dq")
        if dq is not None:
            try:
                q = np.asarray(self._array(dq))
                if q.shape == d.shape:
                    d[(q.astype(np.uint32) & DO_NOT_USE) != 0] = np.nan
            except Exception:
                pass
        d[~np.isfinite(d)] = np.nan
        return d

    # ------------------------------------------------------------ metadati
    def info(self):
        m = self.meta
        filtro = _testo(_valore(m, "instrument", "optical_element") or _cerca(m, "optical_element")
                        or _cerca(m, "filter")).upper()
        rivelatore = _testo(_valore(m, "instrument", "detector") or _cerca(m, "detector")).upper()
        esposizione = None
        for k in ("exposure_time", "effective_exposure_time", "max_exposure_time", "exptime"):
            esposizione = _numero(_valore(m, "exposure", k)) or _numero(_cerca(m, k))
            if esposizione:
                break
        inizio = _testo(_valore(m, "exposure", "start_time") or _cerca(m, "start_time"))
        forma = self.forma()
        nome = self.percorso.name.lower()
        tipo = _testo(_cerca(m, "model_type"))
        if len(forma) == 3 and forma[0] > 1 or "_uncal" in nome or "ramp" in tipo.lower():
            livello = "L1"
        elif re.search(r"_(coadd|mosaic|i2d|skycell|l3)", nome) or "mosaic" in tipo.lower() or not rivelatore:
            livello = "L3"
        else:
            livello = "L2"
        bersaglio = _testo(_valore(m, "target", "name") or _cerca(m, "target_name") or _cerca(m, "proposer_name"))
        return dict(telescopio=_testo(_cerca(m, "telescope")) or "ROMAN", strumento=_testo(_valore(m, "instrument", "name")) or "WFI",
                    filtro=filtro or "sconosciuto", rivelatore=rivelatore, esposizione=esposizione or 0.0,
                    inizio=inizio, forma=forma, livello=livello, tipo=tipo, bersaglio=bersaglio,
                    lambda_um=unita_lunghezza_onda_filtro(filtro), unita=self.unita(),
                    s_region=_testo(_cerca(m, "s_region")), lettore=self.lettore)

    def wcs_fits(self, forma, log=None):
        """(intestazione WCS FITS, descrizione). Prima il GWCS completo, poi s_region."""
        gw = _valore(self.meta, "wcs")
        if gw is None:
            gw = _prendi(self.radice, "wcs")
        h, err = _wcs_da_gwcs(gw, forma, log)
        if h is not None:
            return h, f"GWCS convertito in FITS SIP (errore < {err:g} pixel)"
        s = _testo(_cerca(self.meta, "s_region"))
        h, err = wcs_da_s_region(s, forma)
        if h is not None:
            motivo = ("libreria gwcs non installata" if gw is not None and not hasattr(gw, "to_fits_sip")
                      else "nessun GWCS nel file" if gw is None else "conversione del GWCS non riuscita")
            return h, (f"APPROSSIMATO dal contorno del campo (s_region; {motivo}): "
                       f"scarto ai vertici {err:.2f} px, distorsione non corretta")
        return None, "nessuna coordinata nel file"

    def chiudi(self):
        if self._af is not None:
            try:
                self._af.close()
            except Exception:
                pass
            self._af = None
        if self._interno is not None:
            self._interno.chiudi()


def leggi_info(percorso, usa_libreria=True):
    f = FileRoman(percorso, usa_libreria)
    try:
        return f.info()
    finally:
        f.chiudi()


def problema_immagine(info):
    """Motivo per cui un file Roman non è un'immagine adatta (None se va bene)."""
    if info["livello"] == "L1" or len(info["forma"]) not in (2,) and not (len(info["forma"]) == 3 and info["forma"][0] == 1):
        return "rampe grezze di livello 1 (_uncal): usa i file _cal o i mosaici"
    if any(e in info["filtro"] for e in ELEMENTI_NON_IMMAGINE):
        return f"elemento ottico {info['filtro']}: è uno spettro o una calibrazione, non un'immagine"
    return None


def nome_base(info):
    """Nome base del file convertito: filtro e rivelatore (f158_wfi07), solo filtro per i mosaici."""
    m = re.match(r"^F(\d{3})", info["filtro"])
    base = "f" + m.group(1) if m else (re.sub(r"[^a-z0-9]+", "_", info["filtro"].lower()).strip("_") or "roman")
    if info["livello"] == "L2" and info["rivelatore"]:
        base += "_" + re.sub(r"[^a-z0-9]+", "", info["rivelatore"].lower())
    return base


# =============================================================================
# CONVERSIONE IN FITS
# =============================================================================
def converti_in_fits(sorgente, destinazione, log=print, creatore="AutoAstroPipe", autore="", nota_ascii=""):
    """Scrive un FITS (estensione SCI con il WCS) da un file Roman .asdf e verifica
    che i pixel scritti siano identici a quelli letti. Restituisce (ok, descrizione_wcs, info)."""
    from astropy.io import fits
    f = FileRoman(sorgente)
    try:
        info = f.info()
        d = f.immagine()
        h_wcs, descr = f.wcs_fits(d.shape, log)
    finally:
        f.chiudi()
    hp = _IntestazioneCorta(fits.Header())
    hp["TELESCOP"] = (_ascii(info["telescopio"]).upper() or "ROMAN", "Nancy Grace Roman Space Telescope")
    hp["INSTRUME"] = (_ascii(info["strumento"]).upper() or "WFI", "Wide Field Instrument")
    hp["FILTER"] = (_ascii(info["filtro"]), "elemento ottico (filtro)")
    if info["rivelatore"]:
        hp["DETECTOR"] = (_ascii(info["rivelatore"]), "rivelatore del WFI")
    if info["esposizione"]:
        hp["EXPTIME"] = (float(info["esposizione"]), "tempo di esposizione (s)")
    if info["inizio"]:
        hp["DATE-OBS"] = (_ascii(info["inizio"])[:68], "inizio dell'esposizione")
    if info["bersaglio"]:
        hp["OBJECT"] = _ascii(info["bersaglio"])[:68]
    if info["lambda_um"]:
        hp["PHOTPLAM"] = (info["lambda_um"] * 1e4, "lunghezza d'onda centrale (Angstrom)")
    hp["ROMANLIV"] = (info["livello"], "livello del prodotto Roman (L2 rivelatore, L3 mosaico)")
    hp["ROMANWCS"] = (_ascii(descr)[:68], "origine delle coordinate")
    hp["ORIGFILE"] = (_ascii(Path(sorgente).name)[:68], "file ASDF originale")
    hp["ASDFREAD"] = (_ascii(info["lettore"])[:68], "lettore usato")
    hp["CREATOR"] = (_ascii(creatore)[:68], "programma che ha convertito il file")
    if autore:
        hp["AUTHOR"] = _ascii(autore)[:68]
    if nota_ascii:
        hp.add_comment(_ascii(nota_ascii))
    hp.add_history(_ascii(f"Convertito da ASDF Roman: {Path(sorgente).name}")[:70])
    hp.add_history(_ascii(f"WCS: {descr}")[:70])
    hs = fits.Header()
    if h_wcs is not None:
        for k in list(h_wcs.keys()):
            if k in ("NAXIS", "NAXIS1", "NAXIS2", "SIMPLE", "BITPIX", "EXTEND", "COMMENT", "HISTORY", ""):
                continue
            try:
                hs[k] = h_wcs[k]
            except Exception:
                pass
    if info["unita"]:
        hs["BUNIT"] = _ascii(info["unita"])[:68]
    hs["EXTNAME"] = "SCI"
    destinazione = Path(destinazione)
    fits.HDUList([fits.PrimaryHDU(header=hp.h), fits.ImageHDU(d, header=hs, name="SCI")]).writeto(destinazione)
    with fits.open(destinazione, memmap=True) as hdul:
        letto = np.asarray(hdul["SCI"].data)
        ok = letto.shape == d.shape and bool(np.array_equal(letto, d, equal_nan=True))
    return ok, descr, info


class _IntestazioneCorta:
    """Scrive (valore, commento) togliendo il commento se la riga FITS supererebbe
    80 caratteri (evita gli avvisi 'Card is too long' di astropy)."""

    def __init__(self, h):
        self.h = h

    def __setitem__(self, k, v):
        if isinstance(v, tuple):
            valore, commento = v
            lung = len(repr(valore)) if isinstance(valore, str) else 20
            spazio = 80 - 10 - max(lung, 20) - 3
            v = (valore, commento[:spazio]) if spazio >= 8 else valore
        self.h[k] = v

    def __getattr__(self, nome):
        return getattr(self.h, nome)


def _ascii(testo):
    import unicodedata
    t = str(testo).replace("©", "(C)").replace("×", "x").replace("µ", "u").replace("—", "-").replace("–", "-")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return "".join(c for c in t if 32 <= ord(c) < 127)


# =============================================================================
# SCRITTURA DI FILE ASDF DI PROVA (per l'autotest, senza la libreria asdf)
# =============================================================================
class NodoProva(dict):
    def __init__(self, tag, *a, **k):
        super().__init__(*a, **k)
        self.tag = tag


class _ArrayProva:
    def __init__(self, a, indice):
        self.a, self.indice = a, indice


def _lz4_solo_letterali(raw):
    """Blocco lz4 valido fatto di soli letterali (per provare il decompressore)."""
    n = len(raw)
    if n < 15:
        tok = bytes([n << 4])
        corpo = tok
    else:
        resto = n - 15
        ext = b"\xff" * (resto // 255) + bytes([resto % 255])
        corpo = b"\xf0" + ext
    blocco = struct.pack("<I", n) + corpo + raw
    return struct.pack(">I", len(blocco)) + blocco


def scrivi_asdf_prova(percorso, radice, arrays, compressione=None):
    """Scrive un file ASDF 1.x valido: 'radice' è l'albero sotto 'roman' (dizionari,
    NodoProva con etichetta) in cui gli array numpy di 'arrays' (nome -> array)
    vengono messi nei blocchi binari. compressione: None, 'zlib' o 'lz4'."""
    import yaml
    blocchi = []

    def sostituisci(x):
        if isinstance(x, np.ndarray):
            blocchi.append(np.ascontiguousarray(x.astype(x.dtype.newbyteorder("<"))))
            return _ArrayProva(blocchi[-1], len(blocchi) - 1)
        if isinstance(x, NodoProva):
            return NodoProva(x.tag, {k: sostituisci(v) for k, v in x.items()})
        if isinstance(x, dict):
            return {k: sostituisci(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [sostituisci(v) for v in x]
        return x

    albero = sostituisci(dict(radice, **arrays))

    class Scrittore(yaml.SafeDumper):
        pass

    def rappr_nodo(d, o):
        return d.represent_mapping(o.tag, dict(o))

    def rappr_array(d, o):
        inv = {v: k for k, v in _TIPI.items()}
        return d.represent_mapping("tag:stsci.edu:asdf/core/ndarray-1.0.0",
                                   {"source": o.indice, "datatype": inv[o.a.dtype.str[1:]], "byteorder": "little",
                                    "shape": list(o.a.shape)})

    Scrittore.add_representer(NodoProva, rappr_nodo)
    Scrittore.add_representer(_ArrayProva, rappr_array)
    radice_asdf = NodoProva("tag:stsci.edu:asdf/core/asdf-1.1.0",
                            {"asdf_library": NodoProva("tag:stsci.edu:asdf/core/software-1.0.0",
                                                       {"name": "AutoAstroPipe (prova)", "version": "2.2"}),
                             "roman": NodoProva("asdf://stsci.edu/datamodels/roman/tags/wfi_image-1.0.0", albero)})
    testo = yaml.dump(radice_asdf, Dumper=Scrittore, explicit_start=True, explicit_end=True, version=(1, 1),
                      tags={"!": "tag:stsci.edu:asdf/"}, sort_keys=False, default_flow_style=False)
    with open(percorso, "wb") as f:
        f.write(b"#ASDF 1.0.0\n#ASDF_STANDARD 1.5.0\n")
        f.write(testo.encode("utf-8"))
        for a in blocchi:
            raw = a.tobytes()
            if compressione == "zlib":
                dati, comp = zlib.compress(raw), b"zlib"
            elif compressione == "lz4":
                passo = 1 << 20
                dati = b"".join(_lz4_solo_letterali(raw[i:i + passo]) for i in range(0, len(raw), passo))
                comp = b"lz4\0"
            else:
                dati, comp = raw, b"\0\0\0\0"
            h = struct.pack(">I4sQQQ", 0, comp, len(dati), len(dati), len(raw)) + hashlib.md5(raw).digest()
            f.write(MAGIA_BLOCCO + struct.pack(">H", len(h)) + h + dati)
    return Path(percorso)
