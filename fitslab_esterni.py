# -*- coding: utf-8 -*-
"""
FITS LAB - OPERAZIONI CON STRUMENTI ESTERNI GRATUITI
====================================================
StarNet2 (togli o tieni solo le stelle) e GraXpert (fondo e rumore con AI)
come passi della ricetta. Ogni passo:
  1. riempie i NaN (i programmi esterni non li accettano) e scrive un FITS
     float32 temporaneo (B/N, oppure cubo 3 x H x W se a colori);
  2. lancia il programma installato dal gestore degli strumenti;
  3. rilegge il risultato, controlla forma, verso (alcuni programmi capovolgono
     le righe) e scala dei valori, e rimette i NaN dove erano.
I risultati restano in memoria (stesso ingresso e stessi parametri = nessun
nuovo calcolo), perché questi programmi sono lenti.

© 2026 Edoardo Battistel - FITS Lab
"""

import hashlib
import json
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np

import fitslab_operazioni as O
import fitslab_strumenti as S
from fitslab_operazioni import P, DTYPE

_CACHE = OrderedDict()
_MAX_CACHE = 8


class StrumentoMancante(RuntimeError):
    def __init__(self, strumento):
        self.strumento = strumento
        super().__init__(f"{S.STRUMENTI[strumento].nome} is not installed")


def _chiave(a, op_id, par, lineare):
    h = hashlib.sha1()
    h.update(op_id.encode())
    h.update(json.dumps(par, sort_keys=True, default=str).encode())
    h.update(str(lineare).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _scrivi(a, percorso):
    from astropy.io import fits
    d = np.moveaxis(a, 2, 0) if a.ndim == 3 else a
    fits.PrimaryHDU(np.ascontiguousarray(d, dtype=np.float32)).writeto(percorso, overwrite=True)


def _leggi(percorso, forma):
    from astropy.io import fits
    with fits.open(percorso) as h:
        dati = None
        for hdu in h:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) >= 2:
                dati = np.array(hdu.data, dtype=np.float32)
                break
    if dati is None:
        raise ValueError(f"no image in {Path(percorso).name}")
    dati = np.squeeze(dati)
    if dati.ndim == 3 and len(forma) == 3 and dati.shape[0] == forma[2]:
        dati = np.moveaxis(dati, 0, 2)
    if dati.ndim == 3 and len(forma) == 2:
        dati = dati.mean(axis=0 if dati.shape[0] in (3, 4) else 2)
    if dati.ndim == 2 and len(forma) == 3:
        dati = np.repeat(dati[..., None], forma[2], axis=2)
    if dati.shape != tuple(forma):
        raise ValueError(f"the external program returned an image of size {dati.shape}, expected {forma}")
    return dati


def _stesso_verso(out, inp):
    """Alcuni programmi scrivono le righe al contrario: si sceglie il verso che
    somiglia di più all'ingresso."""
    def corr(x, y):
        x = x[::4, ::4].ravel().astype(np.float64)
        y = y[::4, ::4].ravel().astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 100:
            return 0.0
        x, y = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
        return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-30))
    a = out if out.ndim == 2 else out.mean(axis=2)
    b = inp if inp.ndim == 2 else inp.mean(axis=2)
    if corr(np.flipud(a), b) > corr(a, b) + 0.05:
        return np.flipud(out).copy()
    return out


def _stessa_scala(out, inp, metodo="regressione"):
    """Riporta i valori alla scala dell'ingresso se il programma li ha normalizzati.
    GraXpert scrive (dati - min) / (max - min): out = a·in + b quasi ovunque (il fondo
    tolto è liscio), quindi a e b si stimano con una retta sui pixel e si invertono
    (provato il 15/11 su un drizzle WFPC2 vero: a = 0,048, b = 0,0157).
    Per StarNet (le stelle spariscono, la retta non vale) si confronta il fondo."""
    x = inp if inp.ndim == 2 else inp.mean(axis=2)
    y = out if out.ndim == 2 else out.mean(axis=2)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 100:
        return out
    passo = max(1, int(np.sqrt(ok.sum() / 400_000)))
    xs = x[::passo, ::passo][ok[::passo, ::passo]].astype(np.float64)
    ys = y[::passo, ::passo][ok[::passo, ::passo]].astype(np.float64)
    if metodo == "regressione":
        vx = xs.var()
        if vx <= 0:
            return out
        a = float(((xs - xs.mean()) * (ys - ys.mean())).mean() / vx)
        b = float(ys.mean() - a * xs.mean())
        if a <= 0 or (abs(a - 1) < 0.02 and abs(b) < 0.02 * xs.std()):
            return out
        return ((out - b) / a).astype(DTYPE)
    mi, mo = np.median(xs), np.median(ys)
    si = 1.4826 * np.median(np.abs(xs - mi))
    so = 1.4826 * np.median(np.abs(ys - mo))
    if si <= 0 or so <= 0:
        return out
    k = si / so
    if 0.8 < k < 1.25 and abs(mo - mi) < 0.5 * si:
        return out
    return ((out - mo) * k + mi).astype(DTYPE)


class OperazioneEsterna(O.Operazione):
    famiglia = "science"
    strumento = ""
    lento = True

    def argomenti(self, ingresso, uscita, par, ctx):
        raise NotImplementedError

    def file_uscita(self, cartella, uscita):
        return uscita

    def applica(self, a, par, ctx):
        st = S.STRUMENTI[self.strumento]
        if st.eseguibile() is None:
            raise StrumentoMancante(self.strumento)
        chiave = _chiave(a, self.id, par, ctx.lineare)
        if chiave in _CACHE:
            _CACHE.move_to_end(chiave)
            ctx.note.append(f"{st.nome} {st.versione_installata()}")
            return _CACHE[chiave].copy()
        nan_tutto = ~np.isfinite(a)
        if nan_tutto.all():
            return a
        # solo il rettangolo che contiene dati (+ margine): i drizzle HST/JWST hanno spesso
        # grandi bordi vuoti che, riempiti, confondono i modelli AI e rallentano
        piano = nan_tutto if a.ndim == 2 else nan_tutto.all(axis=2)
        y0, y1, x0, x1 = O.riquadro_dati(np.where(piano, np.nan, 0.0))
        m = 16
        y0, x0 = max(0, y0 - m), max(0, x0 - m)
        y1, x1 = min(a.shape[0], y1 + m), min(a.shape[1], x1 + m)
        intero = a
        a = np.ascontiguousarray(a[y0:y1, x0:x1])
        nan = ~np.isfinite(a)
        pieno = np.where(nan, np.nanmedian(a), a).astype(DTYPE)
        controllo = S.cartella_base() / f"last_{self.id}"
        with tempfile.TemporaryDirectory(prefix="fitslab_ext_") as tmp:
            ingresso = Path(tmp) / "input.fits"
            uscita = Path(tmp) / "output.fits"
            _scrivi(pieno, ingresso)
            args = self.argomenti(ingresso, uscita, par, ctx)
            def riga(r):
                if ctx.progresso and r.startswith("{"):
                    try:
                        d = json.loads(r)
                        f = d.get("progress", d.get("fraction"))
                        if f is not None:
                            ctx.progresso(float(f) / (100.0 if float(f) > 1 else 1.0))
                    except (ValueError, TypeError):
                        pass
            righe = []
            def registra(r):
                righe.append(r)
                riga(r)
            try:
                st.esegui(args, timeout=7200, riga=registra, cwd=tmp)
            finally:
                _salva_controllo(controllo, tmp, args, righe, (y0, y1, x0, x1), intero.shape)
            percorso = self.file_uscita(Path(tmp), uscita)
            out = _leggi(percorso, a.shape)
        out = _stesso_verso(out, pieno)
        if ctx.lineare:
            out = _stessa_scala(out, pieno, "fondo" if self.id == "starnet" else "regressione")
        out = out.astype(DTYPE)
        out[nan] = np.nan
        _scrivi_statistiche(controllo, pieno, out)
        completo = np.full(intero.shape, np.nan, dtype=DTYPE)
        completo[y0:y1, x0:x1] = out
        out = completo
        _CACHE[chiave] = out.copy()
        while len(_CACHE) > _MAX_CACHE:
            _CACHE.popitem(last=False)
        ctx.note.append(f"{st.nome} {st.versione_installata()}")
        return out


class StarNetOp(OperazioneEsterna):
    id = "starnet"
    famiglia = "display"          # senza stelle non è più misurabile: da qui l'uscita è estetica
    strumento = "starnet2"
    parametri = [
        P("result", "choice", "starless", scelte=["starless", "stars"]),
        P("upsample", "bool", False),
        P("stride", "choice", "auto", scelte=["auto", 64, 128, 256]),
        P("highlights", "bool", True),
    ]

    def descrivi(self, par):
        return (f"StarNet2: {'starless image' if par['result'] == 'starless' else 'stars only'}"
                + (", 2x upsampling" if par["upsample"] else ""))

    def argomenti(self, ingresso, uscita, par, ctx):
        args = ["--input", ingresso, "--output", uscita]
        if par["result"] == "stars":
            args += ["--unscreen", Path(uscita).with_name("stars.fits")]
        if ctx.lineare:
            args.append("--linear")
        if par["upsample"]:
            args.append("--upsample")
        if par["stride"] != "auto":
            args += ["--stride", str(par["stride"])]
        if not par["highlights"]:
            args.append("--disable-highlights-protection")
        args.append("--machine-progress")
        return args

    def file_uscita(self, cartella, uscita):
        return cartella / "stars.fits" if (cartella / "stars.fits").exists() and self._stelle else uscita

    def applica(self, a, par, ctx):
        self._stelle = par["result"] == "stars"
        lin = ctx.lineare
        out = super().applica(a, par, ctx)
        ctx.lineare = lin             # con --linear il risultato resta lineare
        return out


class GraXpertFondo(OperazioneEsterna):
    id = "graxpert_bg"
    strumento = "graxpert"
    parametri = [
        P("correction", "choice", "Subtraction", scelte=["Subtraction", "Division"]),
        P("smoothing", "float", 0.1, 0.0, 1.0, 0.01),
    ]

    def descrivi(self, par):
        return f"GraXpert AI background extraction: {par['correction'].lower()}, smoothing {par['smoothing']:g}"

    def argomenti(self, ingresso, uscita, par, ctx):
        return [ingresso, "-cli", "-cmd", "background-extraction", "-output", Path(uscita).with_suffix(""),
                "-correction", par["correction"], "-smoothing", f"{float(par['smoothing']):g}"]

    def file_uscita(self, cartella, uscita):
        return _trova_uscita(cartella, uscita)


class GraXpertRumore(OperazioneEsterna):
    id = "graxpert_denoise"
    strumento = "graxpert"
    parametri = [
        P("strength", "float", 0.5, 0.0, 1.0, 0.01),
        P("batch_size", "int", 4, 1, 32, 1),
        P("gpu", "bool", True),
    ]

    def descrivi(self, par):
        return f"GraXpert AI noise reduction: strength {par['strength']:g}"

    def argomenti(self, ingresso, uscita, par, ctx):
        return [ingresso, "-cli", "-cmd", "denoising", "-output", Path(uscita).with_suffix(""),
                "-strength", f"{float(par['strength']):g}", "-batch_size", str(int(par["batch_size"])),
                "-gpu", "true" if par["gpu"] else "false"]

    def applica(self, a, par, ctx):
        out = super().applica(a, par, ctx)
        ctx.avvisi.add("denoised")
        return out

    def file_uscita(self, cartella, uscita):
        return _trova_uscita(cartella, uscita)


def _salva_controllo(cartella, tmp, args, righe, riquadro, forma):
    """Copia ingresso, uscita e messaggi dell'ultima esecuzione in
    ~/.local/share/fitslab/strumenti/last_<operazione>/ (per capire i problemi)."""
    import shutil
    try:
        if cartella.exists():
            shutil.rmtree(cartella)
        cartella.mkdir(parents=True)
        for p in Path(tmp).iterdir():
            if p.is_file() and p.suffix.lower() in (".fits", ".fit", ".fts", ".tif", ".tiff", ".png", ".txt", ".log"):
                shutil.copy2(p, cartella / p.name)
        (cartella / "command.txt").write_text(
            "command: " + " ".join(str(x) for x in args) + f"\nzone y0,y1,x0,x1 = {riquadro} of image {forma}\n\n"
            + "\n".join(righe), encoding="utf-8")
    except OSError:
        pass


def _scrivi_statistiche(cartella, inp, out):
    try:
        def st(x):
            v = x[np.isfinite(x)]
            if v.size == 0:
                return "all NaN"
            p = np.percentile(v, [0.5, 50, 99.5])
            return (f"finite {v.size}/{x.size}, min {v.min():.5g}, p0.5 {p[0]:.5g}, median {p[1]:.5g}, "
                    f"p99.5 {p[2]:.5g}, max {v.max():.5g}")
        with open(cartella / "command.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\ninput  (sent):     {st(inp)}\noutput (read back, after checks): {st(out)}\n")
    except OSError:
        pass


def _trova_uscita(cartella, uscita):
    """GraXpert aggiunge da sé l'estensione: output.fits, output.fit, ..."""
    for p in [uscita] + sorted(cartella.glob(Path(uscita).stem + "*")):
        if p.exists() and p.name != "input.fits" and p.suffix.lower() in (".fits", ".fit", ".fts"):
            return p
    altri = [p for p in cartella.iterdir() if p.suffix.lower() in (".fits", ".fit", ".fts") and p.name != "input.fits"]
    if altri:
        return altri[0]
    raise FileNotFoundError("the external program did not write a FITS result")


for _c in (StarNetOp, GraXpertFondo, GraXpertRumore):
    O.OPERAZIONI[_c.id] = _c()
if not any(f == "external" for f, _ in O.ORDINE_MENU):
    O.ORDINE_MENU.insert(1, ("external", ["graxpert_bg", "graxpert_denoise", "starnet"]))
