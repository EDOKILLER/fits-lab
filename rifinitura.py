# -*- coding: utf-8 -*-
"""
RIFINITURA - dall'immagine a colori "grezza" a quella da pubblicare
===================================================================
I passaggi che gli elaboratori di immagini dello STScI fanno a mano in
PixInsight e Photoshop, resi automatici e regolabili con le barre:

  1. colori fedeli (Lupton et al. 2004): un solo stretching asinh sulla
     luminosità totale, così il colore di stelle e galassie non sbiadisce
     verso il bianco quando sono brillanti
  2. fondo cielo neutro: il fondo di ogni canale viene portato allo stesso
     grigio scuro (niente più sfondo bluastro o verdognolo) e, a scelta, il
     bianco viene bilanciato
  3. riduzione del rumore a colori (la "grana" colorata) e, solo nelle zone
     deboli, del rumore di luminosità
  4. curve: mezzitoni (gamma) e contrasto, applicati alla luminosità per non
     cambiare i colori
  5. saturazione dei colori
  6. contrasto locale (maschera di contrasto a raggio grande) e nitidezza
  7. riduzione delle stelle (fa risaltare nebulose e galassie)
  8. riparazione dei centri neri delle stelle sature

Tutte le funzioni lavorano su array numpy e si possono applicare a tasselli
(con un margine): l'immagine intera viene così elaborata a piena risoluzione
con poca memoria, con lo stesso risultato dell'anteprima.

© 2026 Edoardo Battistel - AutoAstroPipe versione 2.3
"""

import math

import numpy as np
from scipy import ndimage as ndi

PARAMETRI_PREDEFINITI = {
    "metodo": "lupton",        # lupton / stf / trilogy / asinh_canali (dati lineari) o classico (sezione 6)
    "q": 8.0,                  # Lupton: morbidezza dello stretching asinh
    "bianco": 1.0,             # Lupton: moltiplica il livello che diventa bianco (1 = automatico)
    "nero": 0.0,               # Lupton: punto del nero, in unità di rumore (σ)
    "neutro": True,            # fondo cielo neutro
    "bilancia": True,          # bilanciamento del bianco
    "piedistallo": 0.04,       # livello del fondo cielo (0 = nero)
    "rumore_colore": 3.0,      # raggio (px) della riduzione del rumore a colori
    "rumore_luce": 0.4,        # intensità della riduzione del rumore di luminosità nelle zone deboli
    "gamma": 1.0,              # mezzitoni: >1 più chiari
    "contrasto": 0.2,          # curva a S
    "saturazione": 1.3,
    "contrasto_locale": 0.3,
    "raggio_locale": 60.0,     # px
    "nitidezza": 0.3,
    "stelle": 0.0,             # riduzione delle stelle 0..1
    "dimensione_stelle": 7.0,  # px
    "centri": True,            # ripara i centri neri delle stelle sature
}

ETICHETTE = {
    "q": "Morbidezza (Q)", "bianco": "Luminosità", "nero": "Punto del nero (σ)",
    "piedistallo": "Livello del fondo cielo", "rumore_colore": "Rumore a colori: raggio (px)",
    "rumore_luce": "Rumore di luminosità", "gamma": "Mezzitoni (gamma)", "contrasto": "Contrasto",
    "saturazione": "Saturazione", "contrasto_locale": "Contrasto locale", "raggio_locale": "Raggio contrasto locale (px)",
    "nitidezza": "Nitidezza", "stelle": "Riduzione delle stelle", "dimensione_stelle": "Dimensione delle stelle (px)",
}

PESI_Y = np.array([0.299, 0.587, 0.114], dtype=np.float32)

# Metodi che partono dai dati lineari (in unità di rumore σ di ogni filtro)
LINEARI = ("lupton", "stf", "trilogy", "asinh_canali")


def margine(p):
    """Pixel di bordo necessari per elaborare a tasselli senza cuciture."""
    return int(math.ceil(max(4 * float(p.get("rumore_colore", 0)), 4 * 2.0, 3 * float(p.get("raggio_locale", 0))
                             if p.get("contrasto_locale", 0) else 0, 3 * float(p.get("dimensione_stelle", 0)), 8))) + 4


# =============================================================================
# UTILITÀ
# =============================================================================
def _luminanza(rgb):
    return rgb[..., 0] * PESI_Y[0] + rgb[..., 1] * PESI_Y[1] + rgb[..., 2] * PESI_Y[2]


def _con_luminanza(rgb, y_vecchia, y_nuova):
    """Cambia la luminosità tenendo i rapporti tra i canali (cioè il colore)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(y_vecchia > 1e-6, y_nuova / np.maximum(y_vecchia, 1e-6), 1.0)
    out = rgb * f[..., None]
    piatte = y_vecchia <= 1e-6                      # pixel neri: si aggiunge grigio
    if piatte.any():
        out[piatte] = np.maximum(out[piatte], y_nuova[piatte][:, None])
    return out


def _gauss(a, sigma):
    return ndi.gaussian_filter(a, sigma, mode="reflect") if sigma > 0.05 else a


# =============================================================================
# 1. COMBINAZIONE DEI FILTRI
# =============================================================================
def canali_pesati(piani, pesi):
    """R, G, B come medie pesate dei filtri (pesi = (r, g, b) per ogni filtro).
    I NaN (senza dati) non contano; dove nessun filtro ha dati il pixel è NaN."""
    h, w = piani[0].shape
    out = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        num = np.zeros((h, w), dtype=np.float32)
        den = np.zeros((h, w), dtype=np.float32)
        for a, p in zip(piani, pesi):
            peso = float(p[c])
            if peso <= 0:
                continue
            a = np.asarray(a, dtype=np.float32)
            ok = np.isfinite(a)
            num += np.where(ok, a, 0) * peso
            den += ok * peso
        with np.errstate(invalid="ignore", divide="ignore"):
            out[..., c] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def lupton(rgb_lineare, q, bianco_assoluto, nero=0.0):
    """Lupton et al. (2004): x = asinh(Q·I/S)/asinh(Q), applicato all'intensità
    I = media dei tre canali e ripartito tra i canali in proporzione, quindi i
    colori restano quelli dei dati anche nelle parti brillanti. Se un canale
    supera 1, tutti e tre vengono scalati insieme (il colore non cambia).
    I dati sono in unità di rumore (σ) di ciascun filtro."""
    x = np.asarray(rgb_lineare, dtype=np.float32) - np.float32(nero)
    x = np.where(np.isfinite(x), x, np.nan)
    i = np.nanmean(x, axis=-1) if x.ndim == 3 else x
    q = max(float(q), 1e-3)
    s = max(float(bianco_assoluto), 1e-6)
    with np.errstate(invalid="ignore", divide="ignore"):
        fattore = np.where(i > 0, np.arcsinh(q * i / s) / (math.asinh(q) * np.where(i > 0, i, 1)), 0.0)
        out = np.clip(x, 0, None) * fattore[..., None]
        massimo = np.nanmax(out, axis=-1)
        out = np.where((massimo > 1)[..., None], out / np.where(massimo > 1, massimo, 1)[..., None], out)
    return out.astype(np.float32)


def livelli_canali(rgb_lineare, nero=0.0):
    """Per i metodi «per canale»: fondo (mediana), rumore (MAD) e bianco (99,8°
    percentile) di ciascun canale, misurati sull'anteprima intera."""
    out = {"mediana": [], "rumore": [], "bianco": []}
    for c in range(3):
        v = np.asarray(rgb_lineare[..., c], dtype=np.float64) - nero
        v = v[np.isfinite(v)]
        if v.size < 100:
            out["mediana"].append(0.0); out["rumore"].append(1.0); out["bianco"].append(100.0)
            continue
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826
        out["mediana"].append(med)
        out["rumore"].append(max(mad, 1e-6))
        out["bianco"].append(max(float(np.percentile(v, 99.8)), med + 5 * max(mad, 1e-6)))
    return out


def _mtf(m, x):
    """Funzione dei mezzitoni di PixInsight: mtf(m, m) = 0,5."""
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def stf_per_canale(rgb_lineare, liv, nero=0.0, luminosita=1.0, obiettivo=0.25):
    """Come lo «Screen Transfer Function» automatico di PixInsight, separato per canale:
    nero a mediana − 2,8·rumore, bianco al 99,8° percentile, mezzitoni scelti in modo
    che il fondo cielo vada a 'obiettivo'. Ogni filtro viene tirato per conto suo:
    fondo neutro e colori molto marcati anche con filtri di luminosità diversissima (WISE)."""
    out = np.empty(rgb_lineare.shape, dtype=np.float32)
    for c in range(3):
        x = np.asarray(rgb_lineare[..., c], dtype=np.float32) - nero
        med, rum = liv["mediana"][c], liv["rumore"][c]
        c0 = med - 2.8 * rum
        bianco = c0 + (liv["bianco"][c] - c0) / max(luminosita, 1e-3)
        xn = np.clip((x - c0) / max(bianco - c0, 1e-9), 0, 1)
        xm = min(max((med - c0) / max(bianco - c0, 1e-9), 1e-6), 0.999)
        y = obiettivo
        m = xm * (y - 1) / (2 * xm * y - xm - y)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[..., c] = np.clip(_mtf(m, xn), 0, 1)
    return out


def trilogy_per_canale(rgb_lineare, liv, nero=0.0, luminosita=1.0, q=8.0):
    """Scala logaritmica di Trilogy (Dan Coe): y = log10(k·u + 1) / log10(k + 1),
    u = (x − fondo)/(bianco − fondo); k scelto perché il rumore (1σ sopra il fondo)
    arrivi al livello 0,02·Q (Q = 8 → 0,16). Immagini morbide, molto dettaglio nel debole."""
    out = np.empty(rgb_lineare.shape, dtype=np.float32)
    livello = min(max(0.02 * float(q), 0.02), 0.6)
    for c in range(3):
        x = np.asarray(rgb_lineare[..., c], dtype=np.float32) - nero
        x0, rum = liv["mediana"][c], liv["rumore"][c]
        x1 = x0 + (liv["bianco"][c] - x0) / max(luminosita, 1e-3)
        u_r = rum / max(x1 - x0, 1e-9)
        a, b = -3.0, 12.0                    # ricerca di k su scala logaritmica
        for _ in range(60):
            k = 10 ** ((a + b) / 2)
            if math.log10(k * u_r + 1) / math.log10(k + 1) < livello:
                a = (a + b) / 2
            else:
                b = (a + b) / 2
        k = 10 ** ((a + b) / 2)
        u = np.clip((x - x0) / max(x1 - x0, 1e-9), 0, None)
        out[..., c] = np.clip(np.log10(k * u + 1) / math.log10(k + 1), 0, 1)
    return out


def asinh_per_canale(rgb_lineare, liv, nero=0.0, luminosita=1.0, q=8.0):
    """asinh come Lupton ma con il bianco di ciascun canale: i filtri deboli
    vengono rinforzati, i colori sono più accesi e meno «fedeli»."""
    out = np.empty(rgb_lineare.shape, dtype=np.float32)
    q = max(float(q), 1e-3)
    for c in range(3):
        x = np.asarray(rgb_lineare[..., c], dtype=np.float32) - nero - liv["mediana"][c]
        s = max((liv["bianco"][c] - liv["mediana"][c]) / max(luminosita, 1e-3), 1e-6)
        out[..., c] = np.clip(np.arcsinh(q * np.clip(x, 0, None) / s) / math.asinh(q), 0, 1)
    return out


def bianco_automatico(rgb_lineare, nero=0.0):
    """Livello (in σ) che diventa bianco: il 99,8° percentile dell'intensità."""
    i = np.nanmean(np.asarray(rgb_lineare, dtype=np.float32), axis=-1) - nero
    v = i[np.isfinite(i) & (i > 0)]
    return float(np.percentile(v, 99.8)) if v.size > 100 else 100.0


# =============================================================================
# 2. FONDO NEUTRO E BILANCIAMENTO DEL BIANCO
# =============================================================================
def statistiche(rgb):
    """Fondo, rumore e bianco di ogni canale, misurati sull'anteprima dell'immagine
    intera. Il fondo è la mediana delle zone più scure (30% della luminosità,
    misurata dopo una leggera sfocatura), così nebulose, galassie e rumore non lo falsano."""
    ok = np.isfinite(rgb).all(axis=-1)
    # zone scure scelte su una versione sfocata: scegliendo i singoli pixel più
    # scuri si sceglierebbe anche il rumore, e il fondo verrebbe sottostimato
    y = _gauss(_luminanza(np.nan_to_num(rgb, nan=0.0)), 4.0)
    if ok.sum() < 100:
        return {"fondo": [0.0, 0.0, 0.0], "rumore": [0.0, 0.0, 0.0], "bianco": [1.0, 1.0, 1.0]}
    soglia = np.percentile(y[ok], 30)
    scuri = ok & (y <= soglia)
    fondo, rumore, bianco = [], [], []
    for c in range(3):
        v = rgb[..., c][scuri]
        f = float(np.median(v))
        fondo.append(f)
        rumore.append(float(1.4826 * np.median(np.abs(v - f))))
        tutto = rgb[..., c][ok] - f
        bianco.append(max(float(np.percentile(tutto, 99.8)), 1e-4))
    return {"fondo": fondo, "rumore": rumore, "bianco": bianco}


def neutralizza(rgb, stat, piedistallo=0.04, neutro=True, bilancia=True):
    """Toglie a ogni canale il suo fondo (sfondo grigio neutro) e porta i bianchi
    allo stesso livello (bilanciamento) o allo stesso fattore per tutti."""
    out = np.empty_like(rgb, dtype=np.float32)
    bianchi = stat["bianco"]
    comune = max(bianchi)
    for c in range(3):
        f = stat["fondo"][c] if neutro else 0.0
        b = bianchi[c] if bilancia else comune
        out[..., c] = (rgb[..., c] - f) * ((1.0 - piedistallo) / b) + (piedistallo if neutro else 0.0)
    return out


# =============================================================================
# 3. RUMORE
# =============================================================================
def riduci_rumore(rgb, raggio_colore, intensita_luce, scala=1.0, piedistallo=0.04):
    """Separa luminosità e colore (Y, Cb, Cr): il colore viene levigato col
    raggio dato (toglie la grana colorata, che l'occhio non perdona), la
    luminosità solo nelle zone deboli e in misura regolabile (i dettagli restano)."""
    y = _luminanza(rgb)
    cb = rgb[..., 2] - y
    cr = rgb[..., 0] - y
    s = float(raggio_colore) / scala
    if s > 0.05:
        cb = _gauss(cb, s)
        cr = _gauss(cr, s)
    if intensita_luce > 0:
        liscia = _gauss(y, max(1.2 / scala, 0.6))
        debole = np.clip(1.0 - (y - piedistallo) / 0.25, 0, 1) * float(intensita_luce)
        y = y * (1 - debole) + liscia * debole
    r = y + cr
    b = y + cb
    g = (y - PESI_Y[0] * r - PESI_Y[2] * b) / PESI_Y[1]
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# =============================================================================
# 4-6. CURVE, SATURAZIONE, CONTRASTO LOCALE, NITIDEZZA
# =============================================================================
def curve(rgb, gamma=1.0, contrasto=0.0):
    """Mezzitoni (gamma) e curva a S, sulla luminosità (i colori non cambiano)."""
    y = _luminanza(rgb)
    x = np.clip(y, 0, 1)
    if abs(gamma - 1.0) > 1e-3:
        x = x ** (1.0 / max(gamma, 0.05))
    if abs(contrasto) > 1e-3:
        c = float(np.clip(contrasto, -1, 1))
        x = np.clip(x - c * x * (1 - x) * (1 - 2 * x) * 2.0, 0, 1)
    return _con_luminanza(rgb, y, x)


def saturazione(rgb, fattore):
    if abs(fattore - 1.0) < 1e-3:
        return rgb
    y = _luminanza(rgb)[..., None]
    return (y + (rgb - y) * float(fattore)).astype(np.float32)


def contrasto_locale(rgb, intensita, raggio, scala=1.0):
    """Maschera di contrasto a raggio grande: separa meglio bracci, filamenti e
    polveri dal fondo vicino, senza bruciare le zone luminose."""
    if intensita <= 0:
        return rgb
    y = _luminanza(rgb)
    liscia = _gauss(y, float(raggio) / scala)
    nuova = np.clip(y + float(intensita) * (y - liscia) * np.clip(1.5 - y, 0, 1), 0, None)
    return _con_luminanza(rgb, y, nuova)


def nitidezza(rgb, intensita, scala=1.0):
    if intensita <= 0:
        return rgb
    y = _luminanza(rgb)
    s = max(1.2 / scala, 0.5)
    nuova = np.clip(y + float(intensita) * (y - _gauss(y, s)), 0, None)
    return _con_luminanza(rgb, y, nuova)


# =============================================================================
# 7-8. STELLE
# =============================================================================
def riduci_stelle(rgb, intensita, dimensione=7.0, scala=1.0):
    """Le stelle sono piccole e molto più luminose dei dintorni: un'apertura
    morfologica le toglie; la differenza (top-hat) dice dove sono e quanto
    attenuarle. Nebulose e galassie, più estese, non vengono toccate."""
    if intensita <= 0:
        return rgb
    k = max(3, int(round(float(dimensione) / scala)) | 1)
    y = _luminanza(rgb)
    aperta = ndi.grey_opening(y, size=(k, k), mode="reflect")
    cima = y - aperta
    soglia = float(np.percentile(cima, 90)) if cima.size else 0.0
    maschera = np.clip((cima - soglia) / max(4 * soglia, 1e-3), 0, 1)
    maschera = _gauss(maschera, max(k / 4, 0.7))
    out = np.empty_like(rgb)
    for c in range(3):
        ap = ndi.grey_opening(rgb[..., c], size=(k, k), mode="reflect")
        out[..., c] = rgb[..., c] - float(intensita) * maschera * (rgb[..., c] - ap)
    return out


def ripara_centri(arr, area_max=400, scala=1.0):
    """Centri neri delle stelle sature: piccoli buchi senza dati (NaN) circondati
    da pixel luminosi vengono riempiti con il valore più alto del bordo
    (come lo script PixelClip usato per le immagini di Webb)."""
    a = np.array(arr, dtype=np.float32, copy=True)
    nan = ~np.isfinite(a)
    if not nan.any() or nan.all():
        return a
    lab, n = ndi.label(nan)
    if n == 0:
        return a
    limite = max(4, int(area_max / max(scala, 1) ** 2))
    valide = a[np.isfinite(a)]
    soglia = float(np.percentile(valide, 90)) if valide.size else 0.0
    for k, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - 2, 0), min(sl[0].stop + 2, a.shape[0])
        x0, x1 = max(sl[1].start - 2, 0), min(sl[1].stop + 2, a.shape[1])
        if y0 == 0 or x0 == 0 or y1 == a.shape[0] or x1 == a.shape[1]:
            continue                                   # tocca il bordo: non è un centro di stella
        comp = lab[y0:y1, x0:x1] == k
        if comp.sum() > limite:
            continue
        anello = ndi.binary_dilation(comp, iterations=2) & ~comp
        vicini = a[y0:y1, x0:x1][anello]
        vicini = vicini[np.isfinite(vicini)]
        if vicini.size and float(np.median(vicini)) > soglia:
            blocco = a[y0:y1, x0:x1]
            blocco[comp] = float(vicini.max())
    return a


# =============================================================================
# TUTTO INSIEME
# =============================================================================
def immagine_base(piani, pesi, p, bianco_assoluto=None, scala=1.0):
    """Immagine a colori prima della rifinitura: classica (piani già stirati,
    0..1) oppure Lupton (piani lineari in σ)."""
    if p.get("centri", True):
        piani = [ripara_centri(a, scala=scala) for a in piani]
    rgb = canali_pesati(piani, pesi)
    metodo = p.get("metodo")
    nero = float(p.get("nero", 0.0))
    if metodo == "lupton":
        s = (bianco_assoluto if bianco_assoluto else bianco_automatico(rgb, nero)) / max(float(p.get("bianco", 1.0)), 1e-3)
        rgb = lupton(rgb, p.get("q", 8.0), s, nero)
    elif metodo in LINEARI:
        liv = bianco_assoluto if isinstance(bianco_assoluto, dict) else livelli_canali(rgb, nero)
        lum = float(p.get("bianco", 1.0))
        if metodo == "stf":
            rgb = stf_per_canale(rgb, liv, nero, lum)
        elif metodo == "trilogy":
            rgb = trilogy_per_canale(rgb, liv, nero, lum, p.get("q", 8.0))
        else:
            rgb = asinh_per_canale(rgb, liv, nero, lum, p.get("q", 8.0))
    return rgb


def rifinisci(rgb_base, p, stat, scala=1.0):
    """Applica i passaggi 2-7 all'immagine base. 'stat' viene da statistiche()
    sull'anteprima intera: così ogni tassello usa gli stessi livelli."""
    nan = ~np.isfinite(rgb_base).all(axis=-1)
    rgb = np.nan_to_num(rgb_base, nan=0.0).astype(np.float32)
    pied = float(p.get("piedistallo", 0.04))
    rgb = neutralizza(rgb, stat, pied, p.get("neutro", True), p.get("bilancia", True))
    rgb = riduci_rumore(rgb, p.get("rumore_colore", 0), p.get("rumore_luce", 0), scala, pied)
    rgb = riduci_stelle(rgb, p.get("stelle", 0), p.get("dimensione_stelle", 7), scala)
    rgb = contrasto_locale(rgb, p.get("contrasto_locale", 0), p.get("raggio_locale", 60), scala)
    rgb = nitidezza(rgb, p.get("nitidezza", 0), scala)
    rgb = curve(rgb, p.get("gamma", 1.0), p.get("contrasto", 0.0))
    rgb = saturazione(rgb, p.get("saturazione", 1.0))
    rgb = np.clip(rgb, 0, 1)
    rgb[nan] = 0.0
    return rgb


def prepara_statistiche(piani_anteprima, pesi, p, scala):
    """Livelli comuni (bianco Lupton e statistiche dei canali) calcolati una volta
    sull'anteprima intera."""
    base = immagine_base(piani_anteprima, pesi, dict(p, centri=False), scala=scala)
    bianco = None
    if p.get("metodo") in LINEARI:
        lin = canali_pesati(piani_anteprima, pesi)
        nero = float(p.get("nero", 0.0))
        bianco = bianco_automatico(lin, nero) if p.get("metodo") == "lupton" else livelli_canali(lin, nero)
        base = immagine_base(piani_anteprima, pesi, dict(p, centri=False), bianco, scala)
    return {"bianco_lupton": bianco, "canali": statistiche(base)}


def rifinisci_a_tasselli(leggi_piani, forma, pesi, p, stat, dst, tassello=2048, progresso=None):
    """Elabora l'immagine intera a tasselli con margine. leggi_piani(y0, y1, x0, x1)
    restituisce la lista dei piani dei filtri per quel riquadro; dst è (ny, nx, 3)."""
    ny, nx = forma
    m = margine(p)
    tot = math.ceil(ny / tassello) * math.ceil(nx / tassello)
    k = 0
    for y0 in range(0, ny, tassello):
        y1 = min(y0 + tassello, ny)
        for x0 in range(0, nx, tassello):
            x1 = min(x0 + tassello, nx)
            ya, yb, xa, xb = max(y0 - m, 0), min(y1 + m, ny), max(x0 - m, 0), min(x1 + m, nx)
            piani = leggi_piani(ya, yb, xa, xb)
            base = immagine_base(piani, pesi, p, stat.get("bianco_lupton"))
            r = rifinisci(base, p, stat["canali"])
            dst[y0:y1, x0:x1] = r[y0 - ya:y1 - ya, x0 - xa:x1 - xa]
            k += 1
            if progresso:
                progresso(k, tot)
    if hasattr(dst, "flush"):
        dst.flush()
    return dst
