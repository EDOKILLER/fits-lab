# -*- coding: utf-8 -*-
"""
FITS LAB - laboratorio interattivo per i FITS di tutti i telescopi
==================================================================
A sinistra l'immagine originale, a destra quella che si modifica.
Ogni operazione diventa un passo di una ricetta non distruttiva (si può
spegnere, cambiare, spostare); il file originale non viene mai toccato.
Uscite separate: science/ (FITS lineari, con coordinate) e aesthetic/.

Avvio:  python3 fitslab.py [file.fits]
        python3 fitslab.py --autotest      (prove complete, anche della finestra)
        python3 fitslab.py --diagnostica   (prove di rete: SIMBAD, NED)

© 2026 Edoardo Battistel - FITS Lab
"""

import json
import math
import os
import queue
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fitslab_core as C
import fitslab_operazioni as O
import fitslab_testi as T
import fitslab_strumenti as S
import fitslab_esterni as E

try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
    import tkinter.font as tkfont
except ImportError:          # il motore si può usare anche senza finestra
    tk = None


# =============================================================================
# FINESTRE DI SISTEMA (zenity su Ubuntu: si ingrandiscono)
# =============================================================================
def _ambiente_di_sistema():
    env = dict(os.environ)
    for var in ("LD_LIBRARY_PATH", "GDK_PIXBUF_MODULE_FILE", "GTK_PATH", "GIO_MODULE_DIR"):
        orig = env.pop(var + "_ORIG", None)
        if orig is not None:
            env[var] = orig
        elif getattr(sys, "frozen", False):
            env.pop(var, None)
    return env


def _zenity(padre, argomenti):
    """Risultato di zenity, "" se annullata, None se zenity non c'è o non funziona."""
    import subprocess
    if not sys.platform.startswith("linux") or not shutil.which("zenity"):
        return None
    try:
        proc = subprocess.Popen(["zenity"] + argomenti + ["--width=1100", "--height=750"], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=_ambiente_di_sistema(), text=True)
    except OSError:
        return None
    while proc.poll() is None:
        try:
            padre.update()
        except tk.TclError:
            break
        time.sleep(0.05)
    out, err = proc.communicate()
    if proc.returncode == 0:
        return out.strip().splitlines()[-1] if out.strip() else ""
    if proc.returncode == 1 and not err.strip():
        return ""
    return None


def chiedi_file(padre, titolo, iniziale, filtri):
    """filtri: [(nome, [estensioni])]"""
    arg = ["--file-selection", f"--title={titolo}", f"--filename={str(iniziale).rstrip('/')}/"]
    for nome, est in filtri:
        arg.append(f"--file-filter={nome} | " + " ".join(f"*{e}" for e in est))
    arg.append("--file-filter=* | *")
    r = _zenity(padre, arg)
    if r is not None:
        return r
    return filedialog.askopenfilename(parent=padre, title=titolo, initialdir=str(iniziale),
                                      filetypes=[(n, " ".join(f"*{e}" for e in est)) for n, est in filtri] + [("*", "*")])


def chiedi_cartella(padre, titolo, iniziale):
    r = _zenity(padre, ["--file-selection", "--directory", f"--title={titolo}",
                        f"--filename={str(iniziale).rstrip('/')}/"])
    if r is not None:
        return r
    return filedialog.askdirectory(parent=padre, title=titolo, initialdir=str(iniziale))


def chiedi_salvataggio(padre, titolo, iniziale, nome):
    r = _zenity(padre, ["--file-selection", "--save", "--confirm-overwrite", f"--title={titolo}",
                        f"--filename={Path(iniziale) / nome}"])
    if r is not None:
        return r
    return filedialog.asksaveasfilename(parent=padre, title=titolo, initialdir=str(iniziale), initialfile=nome)


def ram_libera_gb():
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:
        try:
            for r in open("/proc/meminfo"):
                if r.startswith("MemAvailable"):
                    return int(r.split()[1]) / 1e6
        except OSError:
            pass
    return 8.0


# =============================================================================
# VISTA DI UN'IMMAGINE (zoom e spostamento condivisi in pixel originali)
# =============================================================================
class Vista:
    """Converte tra schermo, pixel originali e pixel dell'array mostrato."""

    def __init__(self, canvas):
        self.canvas = canvas
        self.immagine = None         # PIL già capovolta (Nord in alto)
        self.geom = None
        self.piramide = {}
        self.item = None
        self.tk_img = None

    def imposta(self, pil, geom):
        self.immagine = pil
        self.geom = geom
        self.piramide = {0: pil}

    def livello(self, k):
        if k not in self.piramide:
            base = self.livello(k - 1)
            self.piramide[k] = base.reduce(2) if min(base.size) >= 4 else base
        return self.piramide[k]


# =============================================================================
# APPLICAZIONE
# =============================================================================
class FitsLab(tk.Tk if tk else object):
    def __init__(self, file_iniziale=None):
        super().__init__(className="Fitslab")
        self.imp = C.carica_impostazioni()
        self.title(f"{C.NOME} {C.VERSIONE}  —  {C.COPYRIGHT}")
        self._icona()
        self.coda = queue.Queue()
        self.lavoro_id = 0
        self.prev_id = 0
        self._autotest = False
        self._icone = (None, None)
        self._fin_strumenti = None
        self._righe_strumenti = None
        self._ultimo_controllo = {}
        self.sorgente = None
        self.zona = None
        self.fattore = 1
        self.a0 = None
        self.geom0 = None
        self.meta = {}
        self.passi = []
        self.cache = []
        self.op = None
        self.par = None
        self.modifica = None          # indice del passo in modifica
        self.anteprima = None         # (chiave parametri, voce di cache)
        self.passo_mostrato = None    # indice del passo mostrato a destra (None = ultimo)
        self.vista = dict(cx=0.0, cy=0.0, zoom=1.0)
        self.oggetti = []
        self.mouse = tk.StringVar(value="pan")
        self.bersaglio_rett = None    # "zone" o "param"
        self.trascina = None
        self.rett_disegnato = None
        self.mostra_modello = tk.BooleanVar(value=False)
        self.q_orig = []
        self.q_ris = []
        self._timer_anteprima = None
        self.cv_ist = None
        self.cv_curva = None
        self._caratteri()
        self.geometry(f"{int(self.winfo_screenwidth() * 0.92)}x{int(self.winfo_screenheight() * 0.86)}+20+20")
        self.costruisci()
        self.protocol("WM_DELETE_WINDOW", self.esci)
        self.after(50, self._leggi_coda)
        self.after(3000, self.controllo_automatico)
        if file_iniziale:
            self.after(300, lambda: self.apri(file_iniziale))

    # ------------------------------------------------------------------ testi
    def t(self, k, **kw):
        return T.testo(self.imp["lingua"], k, **kw)

    def _icona(self):
        try:
            p = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "fitslab.png"
            if p.exists():
                self._img_icona = tk.PhotoImage(file=str(p))
                self.iconphoto(True, self._img_icona)
        except Exception:
            pass

    def _caratteri(self):
        self.basi = {}
        for nome in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkFixedFont",
                     "TkCaptionFont", "TkSmallCaptionFont", "TkTooltipFont"):
            try:
                f = tkfont.nametofont(nome)
                self.basi[nome] = abs(int(f.cget("size"))) or 10
            except tk.TclError:
                pass
        self.applica_testo(self.imp.get("testo", 150))

    def applica_testo(self, pct):
        self.imp["testo"] = int(pct)
        for nome, base in self.basi.items():
            try:
                tkfont.nametofont(nome).configure(size=max(7, int(round(base * pct / 100))))
            except tk.TclError:
                pass
        self.scala = pct / 100
        try:
            st = ttk.Style(self)
            st.configure("Treeview", rowheight=int(22 * self.scala))
            st.configure("Op.TButton", padding=int(4 * self.scala))
        except tk.TclError:
            pass

    # --- caselle di spunta e pallini grandi (quelli del tema di Tk sono di pochi pixel)
    def _icone_scelta(self):
        lato = max(16, int(round(15 * self.scala)))
        chiave = lato
        if self._icone[0] == chiave:
            return self._icone[1]
        from PIL import Image, ImageDraw, ImageTk
        S = lato * 4                      # disegnate in grande e rimpicciolite: bordi lisci
        def fai(tipo, acceso):
            im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
            d = ImageDraw.Draw(im)
            m, w = S // 10, max(3, S // 12)
            if tipo == "check":
                d.rounded_rectangle([m, m, S - m, S - m], radius=S // 7, fill=(255, 255, 255, 255),
                                    outline=(70, 90, 110, 255), width=w)
                if acceso:
                    d.rounded_rectangle([m, m, S - m, S - m], radius=S // 7, fill=(40, 110, 200, 255),
                                        outline=(30, 80, 150, 255), width=w)
                    d.line([(S * 0.25, S * 0.52), (S * 0.43, S * 0.70), (S * 0.76, S * 0.32)],
                           fill=(255, 255, 255, 255), width=int(S / 9), joint="curve")
            else:
                d.ellipse([m, m, S - m, S - m], fill=(255, 255, 255, 255), outline=(70, 90, 110, 255), width=w)
                if acceso:
                    c = S * 0.28
                    d.ellipse([c, c, S - c, S - c], fill=(40, 110, 200, 255))
            return ImageTk.PhotoImage(im.resize((lato, lato), Image.LANCZOS))
        icone = {(t, a): fai(t, a) for t in ("check", "radio") for a in (False, True)}
        self._icone = (chiave, icone)
        return icone

    def _scelta_tk(self, tipo, padre, **kw):
        icone = self._icone_scelta()
        try:
            sfondo = ttk.Style(self).lookup("TFrame", "background") or self.cget("background")
        except tk.TclError:
            sfondo = None
        classe = tk.Checkbutton if tipo == "check" else tk.Radiobutton
        opz = dict(image=icone[(tipo, False)], selectimage=icone[(tipo, True)], compound="left",
                   indicatoron=False, relief="flat", offrelief="flat", overrelief="flat", bd=0,
                   highlightthickness=0, padx=4, pady=2, font=self.font(), anchor="w")
        if sfondo:
            opz.update(bg=sfondo, activebackground=sfondo, selectcolor=sfondo)
        opz.update(kw)
        testo = opz.get("text")
        if testo:
            opz["text"] = " " + str(testo)
        return classe(padre, **opz)

    def Spunta(self, padre, **kw):
        return self._scelta_tk("check", padre, **kw)

    def Scelta(self, padre, **kw):
        return self._scelta_tk("radio", padre, **kw)

    def adattabile(self, etichetta, margine=12):
        """Etichetta che va a capo seguendo la larghezza del suo contenitore."""
        padre = etichetta.master
        def adatta(e=None):
            try:
                w = padre.winfo_width()
                if w > 20:
                    etichetta.configure(wraplength=max(80, w - margine))
            except tk.TclError:
                pass
        padre.bind("<Configure>", lambda e: adatta(), add="+")
        self.after(50, adatta)
        return etichetta

    def messaggio(self, tipo, titolo, testo, padre=None):
        """Finestra di messaggio ridimensionabile: il testo va a capo seguendo la
        larghezza della finestra e si può selezionare e copiare.
        tipo: info, warning, error, question (restituisce True/False)."""
        if getattr(self, "_autotest", False):
            print(f"     [{tipo}] {titolo}: {testo}", flush=True)
            return True
        padre = padre or self
        d = tk.Toplevel(padre)
        d.title(titolo)
        d.transient(padre)
        d.resizable(True, True)
        try:
            larg = int(max(420 * self.scala, min(self.winfo_width() * 0.5, 1100 * self.scala)))
        except tk.TclError:
            larg = 600
        righe = sum(max(1, math.ceil(len(r) * 9 * self.scala / larg)) for r in (str(testo).splitlines() or [""]))
        alt = int(min(self.winfo_screenheight() * 0.8, (righe + 5) * 26 * self.scala + 60))
        d.geometry(f"{larg}x{alt}+{self.winfo_rootx() + 80}+{self.winfo_rooty() + 80}")
        bt = ttk.Frame(d, padding=(12, 0, 12, 12))
        bt.pack(fill="x", side="bottom")
        fr = ttk.Frame(d, padding=12)
        fr.pack(fill="both", expand=True)
        simbolo, colore = {"error": ("✖", "#c0392b"), "warning": ("⚠", "#b36b00"),
                           "question": ("?", "#1f5fa8")}.get(tipo, ("ℹ", "#1f5fa8"))
        tk.Label(fr, text=simbolo, fg=colore, font=self.font(10, True)).pack(side="left", anchor="n", padx=(0, 12))
        corpo = ttk.Frame(fr)
        corpo.pack(side="left", fill="both", expand=True)
        t = tk.Text(corpo, wrap="word", font=self.font(1), relief="flat", height=1, borderwidth=0,
                    background=d.cget("background"), highlightthickness=0)
        t.insert("1.0", str(testo))
        t.configure(state="disabled")
        t.pack(fill="both", expand=True)
        risposta = {"v": False}
        def chiudi(v):
            risposta["v"] = v
            d.destroy()
        if tipo == "question":
            ttk.Button(bt, text=self.t("yes").capitalize(), command=lambda: chiudi(True)).pack(side="right", padx=4)
            ttk.Button(bt, text=self.t("no").capitalize(), command=lambda: chiudi(False)).pack(side="right")
        else:
            ttk.Button(bt, text=self.t("ok"), command=lambda: chiudi(True)).pack(side="right")
        d.bind("<Return>", lambda e: chiudi(True))
        d.bind("<Escape>", lambda e: chiudi(False))
        try:
            d.wait_visibility()
            d.grab_set()
        except tk.TclError:
            pass
        d.focus_set()
        self.wait_window(d)
        return risposta["v"]

    def font(self, delta=0, grassetto=False):
        f = tkfont.nametofont("TkDefaultFont").actual()
        return (f["family"], max(7, int(f["size"]) + delta), "bold" if grassetto else "normal")

    # ------------------------------------------------------------------ interfaccia
    def costruisci(self):
        for w in self.winfo_children():
            w.destroy()
        self._menu()
        barra = ttk.Frame(self, padding=4)
        barra.pack(side="top", fill="x")
        for k, cmd in (("tb_open", self.chiedi_apri), ("tb_zone", self.chiedi_zona), ("tb_full", self.intera),
                       ("tb_fit", self.adatta), ("tb_objects", self.cerca_oggetti), ("tb_save", self.finestra_salva)):
            ttk.Button(barra, text=self.t(k), command=cmd).pack(side="left", padx=2)
        ttk.Separator(barra, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(barra, text=self.t("mouse")).pack(side="left")
        for v, k in (("pan", "m_pan"), ("rect", "m_rect"), ("points", "m_points")):
            self.Scelta(barra, text=self.t(k), value=v, variable=self.mouse,
                            command=self._cambia_mouse).pack(side="left", padx=2)
        ttk.Separator(barra, orient="vertical").pack(side="left", fill="y", padx=8)
        ttk.Label(barra, text=self.t("red_label")).pack(side="left")
        self.var_rid = tk.StringVar(value=str(self.imp.get("riduzione", "auto")))
        cb = ttk.Combobox(barra, textvariable=self.var_rid, width=5, state="readonly",
                          values=["auto", "1", "2", "4", "8", "16", "32"])
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda e: self.cambia_riduzione(self.var_rid.get()))
        self.lbl_info = ttk.Label(barra, text="")
        self.lbl_info.pack(side="left", padx=12)

        stato = ttk.Frame(self, padding=(6, 2))
        stato.pack(side="bottom", fill="x")
        self.lbl_stato = ttk.Label(stato, text=self.t("status_ready"))
        self.lbl_stato.pack(side="left")
        self.barra_pr = ttk.Progressbar(stato, length=int(220 * self.scala), mode="determinate", maximum=1.0)
        self.barra_pr.pack(side="right")
        self.lbl_coord = ttk.Label(stato, text="")
        self.lbl_coord.pack(side="right", padx=12)

        # pannello a larghezza FISSA (frazione della finestra) e immagini nel resto:
        # il pannello non può più allargarsi da solo e coprire le immagini
        self.corpo = ttk.Frame(self)
        self.corpo.pack(fill="both", expand=True)
        self.cont_pannello = ttk.Frame(self.corpo, width=400)
        self.cont_pannello.pack_propagate(False)
        self.maniglia = tk.Frame(self.corpo, width=max(6, int(7 * self.scala)), cursor="sb_h_double_arrow",
                                 bg="#9aa")
        self.maniglia.bind("<B1-Motion>", self._trascina_divisore)
        self.maniglia.bind("<ButtonRelease-1>", lambda e: C.salva_impostazioni(self.imp))
        immagini = ttk.Frame(self.corpo)
        immagini.columnconfigure(0, weight=1)
        immagini.columnconfigure(1, weight=1)
        immagini.rowconfigure(1, weight=1)
        self.lbl_sx = ttk.Label(immagini, text=self.t("original"), font=self.font(1, True))
        self.lbl_sx.grid(row=0, column=0, sticky="w", padx=4)
        self.lbl_dx = ttk.Label(immagini, text=self.t("result"), font=self.font(1, True))
        self.lbl_dx.grid(row=0, column=1, sticky="w", padx=4)
        self.cv_sx = tk.Canvas(immagini, bg="#111", highlightthickness=0, cursor="crosshair")
        self.cv_dx = tk.Canvas(immagini, bg="#111", highlightthickness=0, cursor="crosshair")
        self.cv_sx.grid(row=1, column=0, sticky="nsew", padx=(0, 2))
        self.cv_dx.grid(row=1, column=1, sticky="nsew", padx=(2, 0))
        self.v_sx, self.v_dx = Vista(self.cv_sx), Vista(self.cv_dx)
        for cv in (self.cv_sx, self.cv_dx):
            cv.bind("<Configure>", lambda e: self.disegna())
            cv.bind("<ButtonPress-1>", self._premi)
            cv.bind("<B1-Motion>", self._muovi)
            cv.bind("<ButtonRelease-1>", self._rilascia)
            cv.bind("<ButtonPress-2>", lambda e: self._premi(e, forza_pan=True))
            cv.bind("<B2-Motion>", self._muovi)
            cv.bind("<ButtonRelease-2>", self._rilascia)
            cv.bind("<ButtonPress-3>", self._destro)
            cv.bind("<Motion>", self._posizione)
            cv.bind("<MouseWheel>", self._rotella)
            cv.bind("<Button-4>", lambda e: self._rotella(e, 1))
            cv.bind("<Button-5>", lambda e: self._rotella(e, -1))
        self.pannello = ttk.Notebook(self.cont_pannello)
        self.pannello.pack(fill="both", expand=True)
        self._pannello()
        self._immagini = immagini
        self._disponi()
        self.bind("<Configure>", self._ridimensiona, add="+")
        self.bind("<F9>", lambda e: self.mostra_pannello())
        self.bind("<Control-z>", lambda e: self.elimina_passo(len(self.passi) - 1) if self.passi else None)
        self.bind("<Control-o>", lambda e: self.chiedi_apri())
        self.bind("<Control-s>", lambda e: self.finestra_salva())
        self.bind("f", lambda e: self.adatta() if not isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Spinbox)) else None)
        self.aggiorna_info()
        self.after(100, self._dopo_costruzione)

    def _dopo_costruzione(self, tentativi=20):
        try:
            if self.winfo_width() < 800 and tentativi > 0:
                self.after(150, lambda: self._dopo_costruzione(tentativi - 1))
                return
            self._ridimensiona()
        except (tk.TclError, ValueError):
            pass
        if self.a0 is not None:
            self.mostra_originale()
            self.mostra_risultato()
            self.adatta()
        else:
            self.disegna()

    def _menu(self):
        m = tk.Menu(self)
        mf = tk.Menu(m, tearoff=False)
        mf.add_command(label=self.t("open"), command=self.chiedi_apri, accelerator="Ctrl+O")
        mf.add_command(label=self.t("open_recipe"), command=self.carica_ricetta)
        mf.add_separator()
        mf.add_command(label=self.t("save"), command=self.finestra_salva, accelerator="Ctrl+S")
        mf.add_command(label=self.t("save_recipe"), command=self.salva_ricetta)
        mf.add_separator()
        mf.add_command(label=self.t("quit"), command=self.esci)
        m.add_cascade(label=self.t("menu_file"), menu=mf)
        mv = tk.Menu(m, tearoff=False)
        mv.add_command(label=self.t("fit"), command=self.adatta, accelerator="F")
        mv.add_command(label=self.t("zoom100"), command=lambda: self.zoom_a(1.0))
        mv.add_command(label=self.t("panel"), command=self.mostra_pannello, accelerator="F9")
        mv.add_separator()
        mv.add_command(label=self.t("annotate"), command=self.cerca_oggetti)
        mv.add_command(label=self.t("clear_annot"), command=lambda: (self.oggetti.clear(), self.disegna()))
        m.add_cascade(label=self.t("menu_view"), menu=mv)
        mo = tk.Menu(m, tearoff=False)
        ml = tk.Menu(mo, tearoff=False)
        self.var_lingua = tk.StringVar(value=self.imp["lingua"])
        for codice, nome in (("it", "Italiano"), ("en", "English")):
            ml.add_radiobutton(label=nome, value=codice, variable=self.var_lingua, command=self.cambia_lingua)
        mo.add_cascade(label=self.t("language"), menu=ml)
        mt = tk.Menu(mo, tearoff=False)
        self.var_testo = tk.IntVar(value=self.imp.get("testo", 150))
        for pct in (100, 125, 150, 175, 200):
            mt.add_radiobutton(label=f"{pct}%", value=pct, variable=self.var_testo,
                               command=lambda: (self.applica_testo(self.var_testo.get()), C.salva_impostazioni(self.imp),
                                                self.costruisci()))
        mo.add_cascade(label=self.t("text_size"), menu=mt)
        self.var_firma = tk.BooleanVar(value=bool(self.imp.get("firma", True)))
        mo.add_checkbutton(label=self.t("signature"), variable=self.var_firma,
                           command=lambda: (self.imp.update(firma=self.var_firma.get()), C.salva_impostazioni(self.imp)))
        m.add_cascade(label=self.t("menu_options"), menu=mo)
        ms = tk.Menu(m, tearoff=False)
        ms.add_command(label=self.t("tools_window"), command=self.finestra_strumenti)
        ms.add_command(label=self.t("tools_check"), command=lambda: self.controlla_strumenti(mostra=True))
        self.var_agg = tk.BooleanVar(value=bool(self.imp.get("aggiorna_strumenti", True)))
        ms.add_checkbutton(label=self.t("tools_auto"), variable=self.var_agg, command=lambda: (
            self.imp.update(aggiorna_strumenti=self.var_agg.get()), C.salva_impostazioni(self.imp)))
        m.add_cascade(label=self.t("menu_tools"), menu=ms)
        ma = tk.Menu(m, tearoff=False)
        ma.add_command(label=self.t("about"), command=lambda: self.messaggio(
            "info", C.NOME, self.t("about_text", v=C.VERSIONE, c=C.COPYRIGHT)))
        m.add_cascade(label=self.t("menu_help"), menu=ma)
        self.config(menu=m)

    def _mostra_parametri(self):
        """Fa scorrere l'area fino ai parametri dell'operazione scelta."""
        try:
            self.update_idletasks()
            tot = max(1, self.fr_tutto.winfo_height())
            self.cv_par.yview_moveto(max(0.0, (self.fr_par.winfo_y() - 8) / tot))
        except (tk.TclError, TypeError):
            pass

    def _disponi_pulsanti(self, larghezza):
        if larghezza < 50 or getattr(self, "_larg_pulsanti", None) == larghezza:
            return
        self._larg_pulsanti = larghezza
        f = tkfont.nametofont("TkDefaultFont")
        piu_lungo = max((f.measure(b.cget("text")) for _, g in self._pulsanti_op for b in g), default=100)
        colonne = 2 if piu_lungo + 40 <= (larghezza - 20) / 2 else (2 if larghezza > 520 else 1)
        larg_col = (larghezza - 20) / colonne
        for lf, gruppo in self._pulsanti_op:
            for i, b in enumerate(gruppo):
                b.grid_forget()
                b.configure(wraplength=max(60, int(larg_col - 24)))
                b.grid(row=i // colonne, column=i % colonne, sticky="nsew", padx=2, pady=2)
            for c in range(2):
                lf.columnconfigure(c, weight=1 if c < colonne else 0, uniform="op" if c < colonne else "")

    def _disponi(self):
        for w in (self.cont_pannello, self.maniglia, self._immagini):
            w.pack_forget()
        if self.imp.get("pannello", True):
            self.cont_pannello.pack(side="right", fill="y")
            self.maniglia.pack(side="right", fill="y")
        self._immagini.pack(side="left", fill="both", expand=True)
        self._ridimensiona()

    def _ridimensiona(self, e=None):
        if e is not None and e.widget is not self:
            return
        try:
            w = self.winfo_width()
        except tk.TclError:
            return
        if w < 200:
            return
        frazione = float(self.imp.get("pannello_frazione", 0.32))
        frazione = min(0.6, max(0.18, frazione))
        larg = int(w * frazione)
        if abs(int(self.cont_pannello.cget("width")) - larg) > 2:
            self.cont_pannello.configure(width=larg)

    def _trascina_divisore(self, e):
        w = max(1, self.winfo_width())
        x = e.x_root - self.winfo_rootx()
        self.imp["pannello_frazione"] = round(min(0.6, max(0.18, 1 - x / w)), 3)
        self._ridimensiona()

    def mostra_pannello(self):
        self.imp["pannello"] = not self.imp.get("pannello", True)
        self._disponi()
        C.salva_impostazioni(self.imp)
        self.after(50, self.disegna)

    def cambia_lingua(self):
        self.imp["lingua"] = self.var_lingua.get()
        C.salva_impostazioni(self.imp)
        op, par, mod = self.op, self.par, self.modifica
        self.costruisci()
        if op:
            self.scegli_op(op, par, mod)

    # ------------------------------------------------------------------ pannello
    def _pannello(self):
        self.tab_op = ttk.Frame(self.pannello, padding=6)
        self.tab_passi = ttk.Frame(self.pannello, padding=6)
        self.tab_q = ttk.Frame(self.pannello, padding=6)
        self.pannello.add(self.tab_op, text=self.t("tab_ops"))
        self.pannello.add(self.tab_passi, text=self.t("tab_steps"))
        self.pannello.add(self.tab_q, text=self.t("tab_quality"))
        # operazioni: pulsanti e parametri nella STESSA area che scorre, con una barra larga
        contenitore = ttk.Frame(self.tab_op)
        contenitore.pack(fill="both", expand=True)
        self.cv_par = tk.Canvas(contenitore, highlightthickness=0)
        sb = tk.Scrollbar(contenitore, orient="vertical", command=self.cv_par.yview,
                          width=max(14, int(16 * self.scala)), bg="#7f97a6", activebackground="#4f6f86",
                          troughcolor="#d8dde2", relief="flat", elementborderwidth=1)
        self.fr_tutto = ttk.Frame(self.cv_par)
        self.fr_tutto.bind("<Configure>", lambda e: self.cv_par.configure(scrollregion=self.cv_par.bbox("all")))
        self._fin_par = self.cv_par.create_window((0, 0), window=self.fr_tutto, anchor="nw")
        self.cv_par.bind("<Configure>", lambda e: self.cv_par.itemconfigure(self._fin_par, width=e.width))
        self.cv_par.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.cv_par.pack(side="left", fill="both", expand=True, padx=(0, 4))
        # rotella del mouse ovunque sopra l'area
        def scorri(e, verso=None):
            if verso is None:
                verso = -1 if e.delta > 0 else 1
            self.cv_par.yview_scroll(verso * 3, "units")
        self.cv_par.bind("<Enter>", lambda e: (self.bind_all("<Button-4>", lambda ev: scorri(ev, -1)),
                                               self.bind_all("<Button-5>", lambda ev: scorri(ev, 1)),
                                               self.bind_all("<MouseWheel>", scorri)))
        self.cv_par.bind("<Leave>", lambda e: (self.unbind_all("<Button-4>"), self.unbind_all("<Button-5>"),
                                               self.unbind_all("<MouseWheel>")))
        self._pulsanti_op = []
        for fam, ids in O.ORDINE_MENU:
            lf = ttk.LabelFrame(self.fr_tutto, text=self.t(f"fam_{fam}"), padding=4)
            lf.pack(fill="x", pady=(0, 6))
            gruppo = []
            for op_id in ids:
                b = tk.Button(lf, text=self.t(f"op_{op_id}"), font=self.font(), justify="center",
                              command=lambda o=op_id: self.scegli_op(o), relief="raised", bd=1,
                              padx=4, pady=max(2, int(3 * self.scala)))
                gruppo.append(b)
            self._pulsanti_op.append((lf, gruppo))
        self.fr_tutto.bind("<Configure>", lambda e: self._disponi_pulsanti(e.width), add="+")
        ttk.Separator(self.fr_tutto).pack(fill="x", pady=4)
        self.fr_par = ttk.Frame(self.fr_tutto)
        self.fr_par.pack(fill="both", expand=True)
        pul = ttk.Frame(self.tab_op)
        pul.pack(fill="x", pady=(6, 0))
        self.btn_applica = ttk.Button(pul, text=self.t("apply"), command=self.applica)
        self.btn_applica.pack(side="left", padx=2)
        ttk.Button(pul, text=self.t("cancel"), command=self.annulla_op).pack(side="left", padx=2)
        ttk.Button(pul, text=self.t("reset"), command=lambda: self.op and self.scegli_op(self.op, None, self.modifica)
                   ).pack(side="left", padx=2)
        self.lbl_prev = ttk.Label(self.tab_op, text="", foreground="#a60")
        self.lbl_prev.pack(fill="x")
        self.adattabile(ttk.Label(self.fr_par, text=self.t("no_file") if self.a0 is None else "")).pack(anchor="w", fill="x")
        # passi
        self.lista = tk.Listbox(self.tab_passi, activestyle="none", font=self.font(), exportselection=False)
        self.lista.pack(fill="both", expand=True)
        self.lista.bind("<<ListboxSelect>>", lambda e: self._seleziona_passo())
        self.lista.bind("<Double-Button-1>", lambda e: self.modifica_passo())
        pp = ttk.Frame(self.tab_passi)
        pp.pack(fill="x", pady=(6, 0))
        for k, cmd in (("st_edit", self.modifica_passo), ("st_toggle", self.accendi_passo),
                       ("st_up", lambda: self.sposta_passo(-1)), ("st_down", lambda: self.sposta_passo(1)),
                       ("st_delete", lambda: self.elimina_passo(self._passo_sel())), ("st_clear", self.elimina_tutti)):
            ttk.Button(pp, text=self.t(k), command=cmd).pack(side="left", padx=1, pady=1)
        # qualità
        ttk.Button(self.tab_q, text=self.t("q_measure"), command=self.misura_risultato).pack(anchor="w")
        self.txt_q = tk.Text(self.tab_q, wrap="none", font=("DejaVu Sans Mono", max(8, int(10 * self.scala))),
                             height=20)
        self.txt_q.pack(fill="both", expand=True, pady=4)
        for tag, col in (("good", "#1a7f1a"), ("warn", "#b36b00"), ("bad", "#c0392b"), ("head", "#000")):
            self.txt_q.tag_configure(tag, foreground=col)
        self.txt_q.tag_configure("head", font=("DejaVu Sans Mono", max(8, int(10 * self.scala)), "bold"))
        self.adattabile(ttk.Label(self.tab_q, text=self.t("q_help"), foreground="#555")).pack(fill="x")
        self.aggiorna_lista()
        self.scrivi_qualita()

    # ------------------------------------------------------------------ coda dei lavori
    def _leggi_coda(self):
        try:
            while True:
                f = self.coda.get_nowait()
                try:
                    f()
                except Exception:
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.after(40, self._leggi_coda)

    def in_thread(self, lavoro, fatto=None, errore=None, testo=None):
        """Esegue lavoro() in un thread; fatto(risultato) torna nella finestra."""
        if testo:
            self.stato(testo)
        def corpo():
            try:
                r = lavoro()
            except Exception as e:
                tb = traceback.format_exc()
                print(tb, file=sys.stderr)
                self.coda.put(lambda: (errore or self._errore)(e))
                return
            if fatto:
                self.coda.put(lambda: fatto(r))
        threading.Thread(target=corpo, daemon=True).start()

    def _errore(self, e):
        self.stato(f"{self.t('err')}: {e}")
        self.messaggio("error", self.t("err"), str(e))

    def progresso(self, f):
        self.coda.put(lambda: self.barra_pr.configure(value=float(f)))

    def stato(self, testo):
        try:
            self.lbl_stato.configure(text=testo)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ apertura
    def chiedi_apri(self):
        p = chiedi_file(self, self.t("open"), self.imp.get("ultima_cartella", str(Path.home())),
                        [("FITS / ASDF", list(C.ESTENSIONI))])
        if p:
            self.apri(p)

    def apri(self, percorso, scelte=None, dopo=None):
        percorso = str(percorso)
        try:
            info = C.ispeziona(percorso)
        except Exception as e:
            self.messaggio("error", self.t("err"), self.t("open_failed", e=e))
            return
        if scelte is None:
            scelte = self.dialogo_apertura(info)
            if scelte is None:
                return
        self.imp["ultima_cartella"] = str(Path(percorso).parent)
        C.salva_impostazioni(self.imp)
        self.barra_pr.configure(value=0)
        def lavoro():
            return C.Sorgente(percorso, scelte, progresso=self.progresso)
        def fatto(s):
            self.sorgente = s
            self.zona = None
            self.oggetti = []
            self.carica_lavoro(dopo=dopo)
        self.in_thread(lavoro, fatto, lambda e: self.messaggio("error", self.t("err"), self.t("open_failed", e=e)),
                       self.t("status_loading", f=Path(percorso).name))

    def dialogo_apertura(self, info):
        """Domande per cubi, SPHEREx e mappe HEALPix. None = annullato."""
        if info["kind"] in ("image", "asdf"):
            return {}
        d = tk.Toplevel(self)
        d.transient(self)
        d.grab_set()
        fr = ttk.Frame(d, padding=12)
        fr.pack(fill="both", expand=True)
        ris = {}
        var = {}
        if info["kind"] == "spherex":
            d.title(self.t("spherex_title"))
            ttk.Label(fr, text=self.t("spherex_text")).pack(anchor="w")
            var["b"] = tk.IntVar(value=3)
            for k, (lo, hi) in enumerate(((0.75, 1.11), (1.11, 1.64), (1.64, 2.42), (2.42, 3.82), (3.82, 4.42),
                                          (4.42, 5.01)), start=1):
                self.Scelta(fr, text=f"D{k}  {lo}–{hi} µm", value=k, variable=var["b"]).pack(anchor="w")
            def ok():
                ris.update(spherex_band=var["b"].get())
                d.destroy()
        elif info["kind"] == "cube":
            d.title(self.t("cube_title"))
            n = info.get("planes", 1)
            ttk.Label(fr, text=self.t("cube_text", n=n)).grid(row=0, column=0, columnspan=4, sticky="w")
            var["m"] = tk.StringVar(value="plane")
            for i, (v, k) in enumerate((("plane", "cube_plane"), ("mean", "cube_mean"), ("sum", "cube_sum")), start=1):
                self.Scelta(fr, text=self.t(k), value=v, variable=var["m"]).grid(row=i, column=0, sticky="w")
            var["p"] = tk.IntVar(value=n // 2)
            ttk.Spinbox(fr, from_=0, to=n - 1, textvariable=var["p"], width=7).grid(row=1, column=1, sticky="w")
            var["a"] = tk.IntVar(value=0)
            var["b"] = tk.IntVar(value=n - 1)
            ttk.Label(fr, text=self.t("cube_from")).grid(row=2, column=1, sticky="e")
            ttk.Spinbox(fr, from_=0, to=n - 1, textvariable=var["a"], width=7).grid(row=2, column=2)
            ttk.Label(fr, text=self.t("cube_to")).grid(row=3, column=1, sticky="e")
            ttk.Spinbox(fr, from_=0, to=n - 1, textvariable=var["b"], width=7).grid(row=3, column=2)
            def ok():
                ris.update(cube_mode=var["m"].get(), cube_plane=var["p"].get(), cube_from=var["a"].get(),
                           cube_to=var["b"].get())
                d.destroy()
        else:
            d.title(self.t("hp_title"))
            ttk.Label(fr, text=self.t("hp_text", n=info.get("nside"))).grid(row=0, column=0, columnspan=3, sticky="w")
            var["m"] = tk.StringVar(value="allsky")
            self.Scelta(fr, text=self.t("hp_allsky"), value="allsky", variable=var["m"]).grid(row=1, column=0, sticky="w")
            self.Scelta(fr, text=self.t("hp_cutout"), value="cutout", variable=var["m"]).grid(row=2, column=0, sticky="w")
            righe = [("hp_column", "col", info.get("columns", [""])[0], info.get("columns", [])),
                     ("hp_frame", "frame", "G", ["G", "C", "E"]),
                     ("hp_width", "width", "2400", None), ("hp_name", "name", "", None),
                     ("hp_lon", "lon", "0", None), ("hp_lat", "lat", "0", None),
                     ("hp_size", "size", "10", None), ("hp_pix", "pix", "", None)]
            for i, (k, chiave, val, scelte) in enumerate(righe, start=3):
                ttk.Label(fr, text=self.t(k)).grid(row=i, column=0, sticky="w", pady=2)
                var[chiave] = tk.StringVar(value=val)
                if scelte:
                    w = ttk.Combobox(fr, textvariable=var[chiave], values=scelte, state="readonly", width=24)
                else:
                    w = ttk.Entry(fr, textvariable=var[chiave], width=26)
                w.grid(row=i, column=1, sticky="w", pady=2)
            ttk.Label(fr, text="G = " + self.t("frame_G") + ",  C = " + self.t("frame_C") + ",  E = " + self.t("frame_E"),
                      foreground="#555").grid(row=20, column=0, columnspan=3, sticky="w")
            def ok():
                try:
                    r = dict(projection=var["m"].get(), column=var["col"].get(), frame=var["frame"].get(),
                             width=int(float(var["width"].get() or 2400)))
                    if r["projection"] == "cutout":
                        nome = var["name"].get().strip()
                        if nome:
                            ra, de = C.risolvi_nome(nome)
                            lon, lat = ra, de
                            if r["frame"] != "C":
                                from astropy.coordinates import SkyCoord
                                import astropy.units as u
                                c = SkyCoord(ra * u.deg, de * u.deg).transform_to(
                                    "galactic" if r["frame"] == "G" else "barycentrictrueecliptic")
                                lon, lat = c.spherical.lon.deg, c.spherical.lat.deg
                        else:
                            lon, lat = float(var["lon"].get()), float(var["lat"].get())
                        r.update(lon=lon, lat=lat, size_deg=float(var["size"].get() or 10))
                        if var["pix"].get().strip():
                            r["pixel_arcmin"] = float(var["pix"].get())
                    ris.update(r)
                    d.destroy()
                except Exception as e:
                    self.messaggio("error", self.t("err"), str(e), padre=d)
        bt = ttk.Frame(fr)
        bt.pack(anchor="e", pady=(10, 0)) if info["kind"] == "spherex" else bt.grid(row=30, column=0, columnspan=4, sticky="e", pady=(10, 0))
        ttk.Button(bt, text=self.t("ok"), command=ok).pack(side="left", padx=4)
        ttk.Button(bt, text=self.t("cancel"), command=d.destroy).pack(side="left")
        self.wait_window(d)
        return ris if ris else None

    def fattore_scelto(self):
        r = str(self.imp.get("riduzione", "auto"))
        if r == "auto" or self.sorgente is None:
            return self.sorgente.fattore_auto(self.zona) if self.sorgente else 1
        return max(1, int(r))

    def carica_lavoro(self, dopo=None):
        """Legge l'immagine di lavoro (zona + riduzione) e rifà i passi."""
        s = self.sorgente
        f = self.fattore_scelto()
        mb = s.memoria_mb(self.zona) / f / f
        libera = ram_libera_gb()
        if mb > 1000 * libera * 0.35 and not getattr(self, "_autotest", False):
            if not self.messaggio("question", self.t("warn"), self.t("big_file", mb=mb, gb=libera)):
                return
        self.fattore = f
        ny, nx = s.shape
        z = self.zona or (0, ny, 0, nx)
        self.barra_pr.configure(value=0)
        def lavoro():
            a = s.leggi(z, f, progresso=self.progresso)
            q = C.qualita(a, s.meta(), O.Geometria(z[0], z[2], f, 1))
            return a, q
        def fatto(r):
            self.a0, self.q_orig = r
            self.geom0 = O.Geometria(z[0], z[2], f, 1)
            piano = self.a0 if self.a0.ndim == 2 else self.a0[..., 0]
            self.riquadro0 = O.riquadro_dati(piano)
            self.meta = s.meta()
            self.cache = []
            self.anteprima = None
            self.passo_mostrato = None
            self.mostra_originale()
            self.adatta()
            self.aggiorna_info()
            self.scrivi_qualita()
            self.title(f"{C.NOME} {C.VERSIONE}  —  {Path(getattr(s, 'originale', s.percorso)).name}  —  {C.COPYRIGHT}")
            if self.passi:
                self.ricalcola(0, dopo=dopo)
            else:
                self.mostra_risultato()
                self.stato(self.t("status_ready"))
                if dopo:
                    dopo()
            if self.op:
                self.scegli_op(self.op, self.par, self.modifica)
        self.in_thread(lavoro, fatto, testo=self.t("status_loading", f=s.percorso.name))

    def aggiorna_info(self):
        if not self.sorgente:
            self.lbl_info.configure(text="")
            return
        m = self.meta or self.sorgente.meta()
        ny, nx = self.sorgente.shape
        z = f"  [{self.zona[3] - self.zona[2]}×{self.zona[1] - self.zona[0]}]" if self.zona else ""
        self.lbl_info.configure(text=f"{m.get('mission')} · {m.get('filter')} · {m.get('wavelength')} · "
                                     f"{nx}×{ny}{z} · ÷{self.fattore}")

    def cambia_riduzione(self, v):
        self.imp["riduzione"] = v
        C.salva_impostazioni(self.imp)
        if self.sorgente:
            self.carica_lavoro()

    def chiedi_zona(self):
        if self.a0 is None:
            return
        self.bersaglio_rett = "zone"
        self.mouse.set("rect")
        self.stato(self.t("status_zone"))

    def intera(self):
        if self.sorgente and self.zona:
            self.zona = None
            self.carica_lavoro()

    def imposta_zona(self, x0, y0, x1, y1):
        ny, nx = self.sorgente.shape
        x0, x1 = sorted((int(max(0, x0)), int(min(nx, x1))))
        y0, y1 = sorted((int(max(0, y0)), int(min(ny, y1))))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return
        self.zona = (y0, y1, x0, x1)
        self.carica_lavoro()
        f = self.fattore_scelto()
        self.stato(self.t("zone_full_res", w=x1 - x0, h=y1 - y0, f=f))

    # ------------------------------------------------------------------ immagini per lo schermo
    def a_pil(self, vis):
        from PIL import Image
        a8 = (np.clip(np.nan_to_num(np.flipud(vis), nan=0.0), 0, 1) * 255 + 0.5).astype(np.uint8)
        return Image.fromarray(a8, "L" if a8.ndim == 2 else "RGB")

    def mostra_originale(self):
        if self.a0 is None:
            return
        self.v_sx.imposta(self.a_pil(C.per_visione(self.a0, True)), self.geom0)
        self.disegna()

    def risultato_corrente(self):
        """(array, geometria, lineare) da mostrare a destra."""
        if self.anteprima is not None:
            a, g, lin, avv, mod = self.anteprima[1]
            if self.mostra_modello.get() and mod is not None:
                return mod, g, True
            return a, g, lin
        if not self.cache:
            return self.a0, self.geom0, True
        i = self.passo_mostrato if self.passo_mostrato is not None else len(self.cache) - 1
        i = min(i, len(self.cache) - 1)
        if i < 0:
            return self.a0, self.geom0, True
        a, g, lin, avv, mod = self.cache[i]
        return a, g, lin

    def mostra_risultato(self):
        if self.a0 is None:
            return
        a, g, lin = self.risultato_corrente()
        self.v_dx.imposta(self.a_pil(C.per_visione(a, lin)), g)
        self.disegna()

    def adatta(self):
        if self.a0 is None:
            return
        w = max(50, self.cv_sx.winfo_width())
        h = max(50, self.cv_sx.winfo_height())
        ry0, ry1, rx0, rx1 = getattr(self, "riquadro0", None) or (0, self.a0.shape[0], 0, self.a0.shape[1])
        ny, nx = ry1 - ry0, rx1 - rx0
        s = self.geom0.s
        self.vista["zoom"] = min(w / (nx * s), h / (ny * s)) * 0.98
        x0, y0 = self.geom0.orig_da_arr(rx0 - 0.5, ry0 - 0.5)
        x1, y1 = self.geom0.orig_da_arr(rx1 - 0.5, ry1 - 0.5)
        self.vista["cx"], self.vista["cy"] = float(x0 + x1) / 2, float(y0 + y1) / 2
        self.disegna()

    def zoom_a(self, z, sx=None, sy=None, cv=None):
        cv = cv or self.cv_sx
        if sx is not None:
            xo, yo = self.schermo_a_orig(cv, sx, sy)
        self.vista["zoom"] = float(min(64.0, max(1e-4, z)))
        if sx is not None:
            w, h = cv.winfo_width(), cv.winfo_height()
            self.vista["cx"] = xo - (sx - w / 2) / self.vista["zoom"]
            self.vista["cy"] = yo + (sy - h / 2) / self.vista["zoom"]
        self.disegna()

    def schermo_a_orig(self, cv, sx, sy):
        w, h = cv.winfo_width(), cv.winfo_height()
        z = self.vista["zoom"]
        return self.vista["cx"] + (sx - w / 2) / z, self.vista["cy"] - (sy - h / 2) / z

    def orig_a_schermo(self, cv, xo, yo):
        w, h = cv.winfo_width(), cv.winfo_height()
        z = self.vista["zoom"]
        return w / 2 + (xo - self.vista["cx"]) * z, h / 2 - (yo - self.vista["cy"]) * z

    def disegna(self):
        for v in (self.v_sx, self.v_dx):
            self._disegna_vista(v)
        self._disegna_sovrapposizioni()

    def _disegna_vista(self, v):
        from PIL import Image, ImageTk
        cv = v.canvas
        try:
            w, h = cv.winfo_width(), cv.winfo_height()
        except tk.TclError:
            return
        cv.delete("img")
        cv.delete("vuoto")
        if v.immagine is None or w < 5 or h < 5:
            if self.a0 is None and v is self.v_sx:
                cv.create_text(w / 2, h / 2, text=self.t("no_file"), fill="#bbb", font=self.font(2), tags="vuoto",
                               width=max(100, w - 40), justify="center")
            return
        g = v.geom
        W, H = v.immagine.size
        z = self.vista["zoom"]
        # pixel dell'array per pixel dello schermo -> livello della piramide
        per_px = 1.0 / (z * g.s)
        k = 0
        while per_px / (2 ** (k + 1)) >= 1.0 and min(W, H) / (2 ** (k + 1)) >= 8:
            k += 1
        img = v.livello(k)
        f = 2 ** k
        xo0, yo1 = self.schermo_a_orig(cv, 0, 0)
        xo1, yo0 = self.schermo_a_orig(cv, w, h)
        xa0, ya1 = g.arr_da_orig(xo0, yo1)
        xa1, ya0 = g.arr_da_orig(xo1, yo0)
        left = (float(xa0) + 0.5) / f
        right = (float(xa1) + 0.5) / f
        upper = (H - 0.5 - float(ya1)) / f
        lower = (H - 0.5 - float(ya0)) / f
        R = getattr(Image, "Resampling", Image)
        metodo = R.NEAREST if z * g.s * f > 1.5 else R.BILINEAR
        estensione = getattr(getattr(Image, "Transform", Image), "EXTENT")
        try:
            vista = img.transform((w, h), estensione, (left, upper, right, lower), metodo)
        except Exception:
            return
        v.tk_img = ImageTk.PhotoImage(vista)
        cv.create_image(0, 0, anchor="nw", image=v.tk_img, tags="img")
        cv.tag_lower("img")

    def _disegna_sovrapposizioni(self):
        for cv in (self.cv_sx, self.cv_dx):
            cv.delete("over")
        cv = self.cv_sx
        f = self.font(-1)
        for o in self.oggetti:
            x, y = self.orig_a_schermo(cv, o["xo"], o["yo"])
            if -50 < x < cv.winfo_width() + 50 and -50 < y < cv.winfo_height() + 50:
                r = 7 * self.scala
                cv.create_oval(x - r, y - r, x + r, y + r, outline="#ffd24a", width=2, tags="over")
                cv.create_text(x + r + 3, y - r, text=o["nome"], anchor="sw", fill="#ffd24a", font=f, tags="over")
        # rettangolo del parametro e punti del fondo
        if self.par is not None and self.op:
            rect = self.par.get("rect")
            if self.op == "crop" and rect and rect[2] > rect[0] and self.par.get("mode") == "rect":
                for c in (self.cv_sx, self.cv_dx):
                    x0, y0 = self.orig_a_schermo(c, rect[0], rect[1])
                    x1, y1 = self.orig_a_schermo(c, rect[2], rect[3])
                    c.create_rectangle(x0, y0, x1, y1, outline="#4af", width=2, dash=(6, 3), tags="over")
            for p in self.par.get("points") or []:
                for c in (self.cv_sx,):
                    x, y = self.orig_a_schermo(c, p[0], p[1])
                    r = 5 * self.scala
                    c.create_oval(x - r, y - r, x + r, y + r, outline="#3f3", width=2, tags="over")
        if self.zona and self.a0 is not None:
            pass
        if self.trascina and self.trascina.get("rett"):
            x0, y0, x1, y1 = self.trascina["rett"]
            self.trascina["cv"].create_rectangle(x0, y0, x1, y1, outline="#fff", width=2, dash=(4, 2), tags="over")

    # ------------------------------------------------------------------ mouse
    def _cambia_mouse(self):
        if self.mouse.get() == "rect":
            if self.bersaglio_rett != "zone":
                self.bersaglio_rett = "param"
            self.stato(self.t("status_rect"))
        elif self.mouse.get() == "points":
            self.stato(self.t("status_points"))
        else:
            self.bersaglio_rett = None

    def _premi(self, e, forza_pan=False):
        modo = "pan" if forza_pan else self.mouse.get()
        if modo == "points":
            if self.op == "background" and self.par is not None:
                xo, yo = self.schermo_a_orig(e.widget, e.x, e.y)
                self.par.setdefault("points", []).append([round(float(xo), 1), round(float(yo), 1)])
                self._aggiorna_widget_punti()
                self.programma_anteprima()
                self.disegna()
            return
        self.trascina = dict(modo=modo, x=e.x, y=e.y, cx=self.vista["cx"], cy=self.vista["cy"], cv=e.widget)

    def _muovi(self, e):
        if not self.trascina:
            return
        if self.trascina["modo"] == "pan":
            z = self.vista["zoom"]
            self.vista["cx"] = self.trascina["cx"] - (e.x - self.trascina["x"]) / z
            self.vista["cy"] = self.trascina["cy"] + (e.y - self.trascina["y"]) / z
            self.disegna()
        else:
            self.trascina["rett"] = (self.trascina["x"], self.trascina["y"], e.x, e.y)
            self._disegna_sovrapposizioni()

    def _rilascia(self, e):
        t = self.trascina
        self.trascina = None
        if not t or t["modo"] != "rect" or not t.get("rett"):
            return
        x0, y0 = self.schermo_a_orig(t["cv"], t["x"], t["y"])
        x1, y1 = self.schermo_a_orig(t["cv"], e.x, e.y)
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        if self.bersaglio_rett == "zone":
            self.bersaglio_rett = None
            self.mouse.set("pan")
            self.imposta_zona(x0, y0, x1, y1)
            return
        if self.op != "crop":
            self.scegli_op("crop")
        self.par["mode"] = "rect"
        self.par["rect"] = [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]
        self.scegli_op("crop", self.par, self.modifica)
        self.programma_anteprima()
        self.disegna()

    def _destro(self, e):
        if self.mouse.get() == "points" and self.par and self.par.get("points"):
            xo, yo = self.schermo_a_orig(e.widget, e.x, e.y)
            pts = self.par["points"]
            d = [math.hypot(p[0] - xo, p[1] - yo) for p in pts]
            i = int(np.argmin(d))
            if d[i] * self.vista["zoom"] < 20:
                pts.pop(i)
                self._aggiorna_widget_punti()
                self.programma_anteprima()
                self.disegna()

    def _rotella(self, e, verso=None):
        if verso is None:
            verso = 1 if e.delta > 0 else -1
        self.zoom_a(self.vista["zoom"] * (1.25 if verso > 0 else 0.8), e.x, e.y, e.widget)

    def _posizione(self, e):
        if self.a0 is None:
            return
        xo, yo = self.schermo_a_orig(e.widget, e.x, e.y)
        testo = self.t("pix", x=xo, y=yo)
        v = self.v_sx if e.widget is self.cv_sx else self.v_dx
        a = self.a0 if v is self.v_sx else self.risultato_corrente()[0]
        if v.geom is not None and a is not None:
            xa, ya = v.geom.arr_da_orig(xo, yo)
            xi, yi = int(round(float(xa))), int(round(float(ya)))
            if 0 <= yi < a.shape[0] and 0 <= xi < a.shape[1]:
                val = a[yi, xi]
                testo += "   " + self.t("val", v=(f"{float(val):.5g}" if np.ndim(val) == 0 else
                                                 "/".join(f"{float(c):.3f}" for c in val)))
        w = self.sorgente.wcs if self.sorgente else None
        if w is not None:
            try:
                lon, lat = w.all_pix2world([[xo, yo]], 0)[0]
                tipo = str(w.wcs.ctype[0])
                if tipo.startswith("RA"):
                    from astropy.coordinates import Angle
                    import astropy.units as u
                    testo += "   RA " + Angle(lon * u.deg).to_string(unit=u.hour, sep=":", precision=1) + \
                             "  Dec " + Angle(lat * u.deg).to_string(unit=u.deg, sep=":", precision=0, alwayssign=True)
                else:
                    testo += f"   l {lon:.3f}°  b {lat:+.3f}°"
            except Exception:
                pass
        self.lbl_coord.configure(text=testo)

    # ------------------------------------------------------------------ operazioni
    def scegli_op(self, op_id, par=None, modifica=None):
        self.op = op_id
        op = O.OPERAZIONI[op_id]
        self.par = par if par is not None else op.predefiniti()
        if par is None and op_id == "crop" and self.a0 is not None:
            self.par["rect"] = [0, 0, 0, 0]
        self.modifica = modifica
        self.anteprima = None
        for w in self.fr_par.winfo_children():
            w.destroy()
        self.widget_par = {}
        ttk.Label(self.fr_par, text=self.t(f"op_{op_id}"), font=self.font(2, True)).pack(anchor="w")
        if modifica is not None:
            ttk.Label(self.fr_par, text=self.t("editing", n=modifica + 1), foreground="#a60").pack(anchor="w")
        self.adattabile(ttk.Label(self.fr_par, text=self.t(f"opd_{op_id}"), foreground="#444")).pack(anchor="w", fill="x", pady=(0, 6))
        if op_id in ("stretch", "curves"):
            self.cv_ist = tk.Canvas(self.fr_par, height=int(110 * self.scala), bg="#fafafa", highlightthickness=1,
                                    highlightbackground="#ccc")
            self.cv_ist.pack(fill="x", pady=(0, 6))
            self.cv_ist.bind("<Configure>", lambda e: self._disegna_istogramma())
        for p in op.parametri:
            if p.visibile_se and self.par.get(p.visibile_se[0]) not in p.visibile_se[1]:
                continue
            self._widget_parametro(p)
        if op_id == "background":
            self.Spunta(self.fr_par, text=self.t("show_model"), variable=self.mostra_modello,
                            command=self.mostra_risultato_o_anteprima).pack(anchor="w", pady=4)
        if op_id == "stretch":
            ttk.Button(self.fr_par, text=self.t("auto_btn"), command=self._stretch_auto).pack(anchor="w", pady=4)
        if getattr(op, "lento", False):
            st = S.STRUMENTI[op.strumento]
            v = st.versione_installata()
            self.adattabile(ttk.Label(self.fr_par, foreground="#246", text=(
                f"{st.nome} {v}" if v else self.t("tool_missing", n=st.nome)) + f" · {st.licenza}")).pack(
                anchor="w", fill="x", pady=(6, 2))
            riga = ttk.Frame(self.fr_par)
            riga.pack(anchor="w", pady=4)
            ttk.Button(riga, text=self.t("compute_preview"),
                       command=lambda: self.programma_anteprima(subito=True, forza=True)).pack(side="left")
            if not v:
                ttk.Button(riga, text=self.t("tool_install"),
                           command=lambda: self.installa_strumento(op.strumento)).pack(side="left", padx=4)
        self.btn_applica.configure(text=self.t("update_step") if modifica is not None else self.t("apply"))
        self.pannello.select(self.tab_op)
        self.after(80, self._mostra_parametri)
        self.after(30, self._disegna_istogramma)
        self.programma_anteprima(subito=True)
        self.disegna()

    def _ingresso_passo(self):
        """Array e stato in ingresso al passo in modifica (o in fondo alla ricetta)."""
        i = self.modifica if self.modifica is not None else len(self.passi)
        if i == 0 or not self.cache:
            return self.a0, self.geom0, True
        a, g, lin, avv, mod = self.cache[min(i, len(self.cache)) - 1]
        return a, g, lin

    def _widget_parametro(self, p):
        fr = ttk.Frame(self.fr_par)
        fr.pack(fill="x", pady=3)
        etichetta = self.t(f"p_{p.id}")
        if p.tipo in ("float", "int"):
            ttk.Label(fr, text=etichetta).pack(anchor="w")
            riga = ttk.Frame(fr)
            riga.pack(fill="x")
            var = tk.StringVar(value=self._fmt(p, self.par[p.id]))
            logaritmica = p.minimo and p.minimo > 0 and p.massimo / p.minimo >= 100
            def a_scala(v, p=p, lg=logaritmica):
                v = min(max(float(v), p.minimo), p.massimo)
                return (math.log(v / p.minimo) / math.log(p.massimo / p.minimo)) if lg else \
                    (v - p.minimo) / (p.massimo - p.minimo)
            def da_scala(x, p=p, lg=logaritmica):
                v = p.minimo * (p.massimo / p.minimo) ** x if lg else p.minimo + x * (p.massimo - p.minimo)
                if p.passo:
                    v = round(v / p.passo) * p.passo
                return int(round(v)) if p.tipo == "int" else float(round(v, 6))
            sc = ttk.Scale(riga, from_=0.0, to=1.0, orient="horizontal")
            sc.set(a_scala(self.par[p.id]))
            def mosso(x, p=p, var=var, da_scala=da_scala):
                v = da_scala(float(x))
                if v != self.par.get(p.id):
                    self.par[p.id] = v
                    var.set(self._fmt(p, v))
                    self.programma_anteprima()
            sc.configure(command=mosso)
            sc.pack(side="left", fill="x", expand=True)
            en = ttk.Entry(riga, textvariable=var, width=9)
            en.pack(side="left", padx=4)
            def scritto(e=None, p=p, var=var, sc=sc, a_scala=a_scala):
                try:
                    v = float(var.get().replace(",", "."))
                except ValueError:
                    return
                v = min(max(v, p.minimo), p.massimo)
                v = int(round(v)) if p.tipo == "int" else v
                self.par[p.id] = v
                sc.set(a_scala(v))
                self.programma_anteprima()
            en.bind("<Return>", scritto)
            en.bind("<FocusOut>", scritto)
            self.widget_par[p.id] = (var, sc, a_scala)
        elif p.tipo == "choice":
            ttk.Label(fr, text=etichetta).pack(anchor="w")
            testi = [T.scelta(self.imp["lingua"], c) for c in p.scelte]
            var = tk.StringVar(value=T.scelta(self.imp["lingua"], self.par[p.id]))
            cb = ttk.Combobox(fr, textvariable=var, values=testi, state="readonly")
            cb.pack(fill="x")
            def scelto(e=None, p=p, var=var, testi=testi):
                v = p.scelte[testi.index(var.get())]
                self.par[p.id] = v
                if p.id == "preset" and v in O.PRESET_CURVE:
                    self.par["curve"] = [list(x) for x in O.PRESET_CURVE[v]]
                self.scegli_op(self.op, self.par, self.modifica)
            cb.bind("<<ComboboxSelected>>", scelto)
        elif p.tipo == "bool":
            var = tk.BooleanVar(value=bool(self.par[p.id]))
            def spunta(p=p, var=var):
                self.par[p.id] = bool(var.get())
                self.scegli_op(self.op, self.par, self.modifica)
            self.Spunta(fr, text=etichetta, variable=var, command=spunta).pack(anchor="w")
        elif p.tipo == "rect":
            ttk.Label(fr, text=etichetta).pack(anchor="w")
            r = self.par[p.id]
            testo = self.t("rect_none") if not r or r[2] <= r[0] else \
                f"x {r[0]:.0f}–{r[2]:.0f}, y {r[1]:.0f}–{r[3]:.0f}  ({r[2] - r[0]:.0f}×{r[3] - r[1]:.0f})"
            ttk.Label(fr, text=testo, foreground="#246").pack(anchor="w")
            ttk.Button(fr, text=self.t("draw_rect"), command=lambda: (setattr(self, "bersaglio_rett", "param"),
                                                                     self.mouse.set("rect"), self._cambia_mouse())
                       ).pack(anchor="w")
        elif p.tipo == "points":
            ttk.Label(fr, text=etichetta).pack(anchor="w")
            self.lbl_punti = ttk.Label(fr, text=self.t("points_n", n=len(self.par.get(p.id) or [])), foreground="#246")
            self.lbl_punti.pack(anchor="w")
            riga = ttk.Frame(fr)
            riga.pack(anchor="w")
            ttk.Button(riga, text=self.t("add_points"), command=lambda: (self.mouse.set("points"), self._cambia_mouse())
                       ).pack(side="left")
            ttk.Button(riga, text=self.t("clear_points"), command=self._cancella_punti).pack(side="left", padx=4)
        elif p.tipo == "curve":
            lato = int(250 * self.scala)
            self.cv_curva = tk.Canvas(fr, width=lato, height=lato, bg="#fff", highlightthickness=1,
                                      highlightbackground="#999")
            self.cv_curva.pack(anchor="w")
            self.cv_curva.bind("<ButtonPress-1>", self._curva_premi)
            self.cv_curva.bind("<B1-Motion>", self._curva_muovi)
            self.cv_curva.bind("<ButtonRelease-1>", lambda e: self.programma_anteprima())
            self.cv_curva.bind("<ButtonPress-3>", self._curva_togli)
            self.after(20, self._disegna_curva)

    @staticmethod
    def _fmt(p, v):
        if p.tipo == "int":
            return str(int(v))
        if p.passo and p.passo < 0.01:
            return f"{float(v):.4f}"
        return f"{float(v):g}"

    def _aggiorna_widget_punti(self):
        try:
            self.lbl_punti.configure(text=self.t("points_n", n=len(self.par.get("points") or [])))
        except (AttributeError, tk.TclError):
            pass

    def _cancella_punti(self):
        if self.par is not None:
            self.par["points"] = []
            self._aggiorna_widget_punti()
            self.programma_anteprima()
            self.disegna()

    def _stretch_auto(self):
        a, g, lin = self._ingresso_passo()
        if a is None:
            return
        x = a.mean(axis=2) if a.ndim == 3 else a
        self.par.update(O.Stiramento.auto_valori(x, lin))
        if self.par["method"] == "auto":
            self.par["method"] = "ht"
        self.scegli_op("stretch", self.par, self.modifica)

    # --- istogramma e curva
    def _disegna_istogramma(self):
        cv = getattr(self, "cv_ist", None)
        if cv is None or not cv.winfo_exists() or self.a0 is None:
            return
        a, g, lin = self._ingresso_passo()
        x = a.mean(axis=2) if a.ndim == 3 else a
        if self.op == "curves" and lin:
            x, lin = O.stira_auto(x), False
        h = O.istogramma(x, lin).astype(float)
        cv.delete("all")
        w, hh = cv.winfo_width(), cv.winfo_height()
        if h.max() <= 0:
            return
        hl = np.log1p(h) / np.log1p(h.max())
        n = len(h)
        for i, v in enumerate(hl):
            x0 = i * w / n
            cv.create_rectangle(x0, hh, x0 + w / n + 1, hh - v * (hh - 4), fill="#88a", outline="")
        if self.op == "stretch" and self.par.get("method") in ("ht", "asinh", "log", "sqrt"):
            for k, col in (("black", "#000"), ("white", "#c00")):
                xx = float(self.par.get(k, 0)) * w
                cv.create_line(xx, 0, xx, hh, fill=col, width=2)
            if self.par.get("method") == "ht":
                b, wt = float(self.par["black"]), float(self.par["white"])
                xx = (b + float(self.par["mid"]) * (wt - b)) * w
                cv.create_line(xx, 0, xx, hh, fill="#080", width=2, dash=(3, 2))

    def _disegna_curva(self):
        cv = getattr(self, "cv_curva", None)
        if cv is None or not cv.winfo_exists():
            return
        cv.delete("all")
        L = int(cv.cget("width"))
        for i in range(1, 4):
            cv.create_line(i * L / 4, 0, i * L / 4, L, fill="#ddd")
            cv.create_line(0, i * L / 4, L, i * L / 4, fill="#ddd")
        cv.create_line(0, L, L, 0, fill="#ccc", dash=(2, 2))
        f = O.funzione_curva(self.par["curve"])
        xs = np.linspace(0, 1, 128)
        ys = f(xs)
        cv.create_line(*[c for x, y in zip(xs, ys) for c in (x * L, (1 - y) * L)], fill="#036", width=2, smooth=True)
        r = 5 * self.scala
        for x, y in self.par["curve"]:
            cv.create_oval(x * L - r, (1 - y) * L - r, x * L + r, (1 - y) * L + r, fill="#fff", outline="#036", width=2)

    def _curva_punto(self, e):
        L = int(self.cv_curva.cget("width"))
        return min(1, max(0, e.x / L)), min(1, max(0, 1 - e.y / L)), L

    def _curva_premi(self, e):
        x, y, L = self._curva_punto(e)
        pts = self.par["curve"]
        d = [math.hypot((p[0] - x) * L, (p[1] - y) * L) for p in pts]
        i = int(np.argmin(d))
        if d[i] < 12 * self.scala:
            self._curva_sel = i
        else:
            pts.append([round(x, 4), round(y, 4)])
            pts.sort()
            self._curva_sel = pts.index([round(x, 4), round(y, 4)])
        self.par["preset"] = "custom"
        self._disegna_curva()

    def _curva_muovi(self, e):
        i = getattr(self, "_curva_sel", None)
        if i is None:
            return
        x, y, L = self._curva_punto(e)
        pts = self.par["curve"]
        lo = pts[i - 1][0] + 0.005 if i > 0 else 0.0
        hi = pts[i + 1][0] - 0.005 if i < len(pts) - 1 else 1.0
        if i == 0:
            x = 0.0 if pts[0][0] == 0 else min(max(x, 0), hi)
        elif i == len(pts) - 1:
            x = 1.0 if pts[-1][0] == 1 else min(max(x, lo), 1)
        else:
            x = min(max(x, lo), hi)
        pts[i] = [round(x, 4), round(y, 4)]
        self._disegna_curva()

    def _curva_togli(self, e):
        x, y, L = self._curva_punto(e)
        pts = self.par["curve"]
        d = [math.hypot((p[0] - x) * L, (p[1] - y) * L) for p in pts]
        i = int(np.argmin(d))
        if d[i] < 12 * self.scala and 0 < i < len(pts) - 1:
            pts.pop(i)
            self._disegna_curva()
            self.programma_anteprima()

    # --- anteprima dal vivo
    def programma_anteprima(self, subito=False, forza=False):
        if self.a0 is None or self.op is None:
            return
        if getattr(O.OPERAZIONI[self.op], "lento", False) and not forza:
            try:
                self.lbl_prev.configure(text=self.t("slow_preview"))
            except tk.TclError:
                pass
            return
        if self._timer_anteprima:
            self.after_cancel(self._timer_anteprima)
        self._timer_anteprima = self.after(10 if subito else 350, self._calcola_anteprima)

    def _calcola_anteprima(self):
        self._timer_anteprima = None
        if self.op is None or self.par is None or self.a0 is None:
            return
        if self.op == "crop" and self.par.get("mode") == "rect" and not (self.par["rect"][2] > self.par["rect"][0]):
            self.anteprima = None
            self.mostra_risultato()
            return
        a, g, lin = self._ingresso_passo()
        i = self.modifica if self.modifica is not None else len(self.passi)
        passo = dict(op=self.op, on=True, params=json.loads(json.dumps(self.par)))
        chiave = json.dumps(passo, sort_keys=True)
        self.prev_id += 1
        mio = self.prev_id
        avv0 = self.cache[i - 1][3] if i > 0 and self.cache else set()
        def lavoro():
            ctx = O.Contesto(g.copia(), dict(self.meta), lin, anteprima=True)
            ctx.avvisi = set(avv0)
            r = O.applica_passo(a, passo, ctx)
            return (r, ctx.geom.copia(), ctx.lineare, set(ctx.avvisi), ctx.modello)
        def fatto(voce):
            if mio != self.prev_id or self.op is None:
                return
            self.anteprima = (chiave, voce)
            self.lbl_prev.configure(text="")
            self.mostra_risultato()
            self.stato(self.t("status_ready"))
        self.lbl_prev.configure(text=self.t("preview_wait"))
        self.in_thread(lavoro, fatto, self._errore_calcolo)

    def mostra_risultato_o_anteprima(self):
        self.mostra_risultato()

    def applica(self):
        if self.a0 is None or self.op is None:
            return
        op_obj = O.OPERAZIONI[self.op]
        if getattr(op_obj, "lento", False) and S.STRUMENTI[op_obj.strumento].eseguibile() is None:
            self.offri_installazione(op_obj.strumento)
            return
        passo = dict(op=self.op, on=True, params=json.loads(json.dumps(self.par)))
        chiave = json.dumps(passo, sort_keys=True)
        if self.modifica is not None and self.modifica < len(self.passi):
            i = self.modifica
            passo["on"] = self.passi[i].get("on", True)
            self.passi[i] = passo
        else:
            i = len(self.passi)
            self.passi.append(passo)
        riuso = self.anteprima if (self.anteprima and self.anteprima[0] == chiave and len(self.cache) >= i) else None
        op_fatta = self.op
        self.op, self.par, self.modifica, self.anteprima = None, None, None, None
        self.mostra_modello.set(False)
        for w in self.fr_par.winfo_children():
            w.destroy()
        ttk.Label(self.fr_par, text="✓ " + self.t(f"op_{op_fatta}"), font=self.font(1, True)).pack(anchor="w")
        if self.mouse.get() != "pan":
            self.mouse.set("pan")
        self.passo_mostrato = None
        if riuso and i == len(self.passi) - 1:
            self.cache = self.cache[:i] + [riuso[1]]
            self.aggiorna_lista()
            self.mostra_risultato()
        else:
            self.ricalcola(i)

    def annulla_op(self):
        self.op, self.par, self.modifica, self.anteprima = None, None, None, None
        self.mostra_modello.set(False)
        for w in self.fr_par.winfo_children():
            w.destroy()
        self.mouse.set("pan")
        self.mostra_risultato()

    def ricalcola(self, da=0, dopo=None):
        if self.a0 is None:
            return
        self.lavoro_id += 1
        mio = self.lavoro_id
        passi = json.loads(json.dumps(self.passi))
        cache = list(self.cache)
        self.barra_pr.configure(value=0)
        def lavoro():
            return C.esegui(self.a0, self.geom0, passi, self.meta, da=min(da, len(cache)), cache=cache,
                            progresso=self.progresso, anteprima=True)
        def fatto(c):
            if mio != self.lavoro_id or c is None:
                return
            self.cache = c
            self.aggiorna_lista()
            self.mostra_risultato()
            self.stato(self.t("status_ready"))
            if dopo:
                dopo()
        self.in_thread(lavoro, fatto, self._errore_calcolo, testo=self.t("status_computing"))

    def _errore_calcolo(self, e):
        try:
            self.lbl_prev.configure(text="")
        except tk.TclError:
            pass
        if isinstance(e, E.StrumentoMancante):
            self.offri_installazione(e.strumento)
        else:
            self._errore(e)

    # ------------------------------------------------------------------ passi
    def aggiorna_lista(self):
        try:
            self.lista.delete(0, "end")
        except (AttributeError, tk.TclError):
            return
        if not self.passi:
            self.lista.insert("end", self.t("steps_empty"))
            return
        ultimo_sci = None
        for i, p in enumerate(self.passi):
            if p.get("on", True) and O.OPERAZIONI[p["op"]].famiglia == "display":
                break
            ultimo_sci = i
        for i, p in enumerate(self.passi):
            op = O.OPERAZIONI[p["op"]]
            par = op.predefiniti()
            par.update(p.get("params", {}))
            segno = "✓" if p.get("on", True) else "○"
            dett = op.descrivi(par).split(":", 1)[-1].strip()
            self.lista.insert("end", f"{segno} {i + 1}. {self.t('op_' + p['op'])} — {dett}")
            if not p.get("on", True):
                self.lista.itemconfigure("end", foreground="#999")
            if i == ultimo_sci and i < len(self.passi) - 1:
                self.lista.insert("end", "     " + self.t("science_line"))
                self.lista.itemconfigure("end", foreground="#1a7f1a")

    def _passo_sel(self):
        sel = self.lista.curselection()
        if not sel or not self.passi:
            return None
        riga = sel[0]
        # le righe di separazione non sono passi
        testo = self.lista.get(riga)
        if not testo.lstrip().startswith(("✓", "○")):
            return None
        return int(testo.split(".", 1)[0].split()[-1]) - 1

    def _seleziona_passo(self):
        i = self._passo_sel()
        if i is None:
            return
        self.passo_mostrato = i
        self.anteprima = None
        self.mostra_risultato()

    def modifica_passo(self):
        i = self._passo_sel()
        if i is None:
            return
        p = self.passi[i]
        op = O.OPERAZIONI[p["op"]]
        par = op.predefiniti()
        par.update(json.loads(json.dumps(p.get("params", {}))))
        self.scegli_op(p["op"], par, i)

    def accendi_passo(self):
        i = self._passo_sel()
        if i is None:
            return
        self.passi[i]["on"] = not self.passi[i].get("on", True)
        self.ricalcola(i)

    def sposta_passo(self, d):
        i = self._passo_sel()
        if i is None or not (0 <= i + d < len(self.passi)):
            return
        self.passi[i], self.passi[i + d] = self.passi[i + d], self.passi[i]
        self.ricalcola(min(i, i + d))

    def elimina_passo(self, i):
        if i is None or not (0 <= i < len(self.passi)):
            return
        self.passi.pop(i)
        self.passo_mostrato = None
        self.ricalcola(i)

    def elimina_tutti(self):
        self.passi = []
        self.cache = []
        self.passo_mostrato = None
        self.aggiorna_lista()
        self.mostra_risultato()

    # ------------------------------------------------------------------ qualità
    def scrivi_qualita(self):
        try:
            t = self.txt_q
        except AttributeError:
            return
        t.configure(state="normal")
        t.delete("1.0", "end")
        if not self.q_orig:
            t.configure(state="disabled")
            return
        ris = {v["key"]: v for v in self.q_ris}
        t.insert("end", f"{'':<20}{self.t('q_original'):<22}{self.t('q_result') if ris else ''}\n", "head")
        for v in self.q_orig:
            nome = self.t("q_" + v["key"])
            val = v["text"]
            if v["key"] == "wcs":
                val = self.t("yes") if v["value"] else self.t("no")
            t.insert("end", f"{nome[:19]:<20}")
            t.insert("end", f"{val[:21]:<22}", v["verdict"])
            r = ris.get(v["key"])
            if r:
                t.insert("end", f"{r['text'][:21]}", r["verdict"])
            t.insert("end", "\n")
        t.configure(state="disabled")

    def misura_risultato(self):
        if self.a0 is None:
            return
        a, g, lin, avv = C.stato_scienza(self.a0, self.geom0, self.passi, self.cache)
        def lavoro():
            return C.qualita(a, self.meta, g, lin)
        def fatto(q):
            self.q_ris = q
            self.scrivi_qualita()
            self.stato(self.t("status_ready"))
        self.in_thread(lavoro, fatto, testo=self.t("status_computing"))

    # ------------------------------------------------------------------ oggetti
    def cerca_oggetti(self):
        if self.a0 is None:
            return
        w = C.wcs_per_geometria(self.sorgente.wcs, self.geom0, self.a0.shape[:2])
        if w is None:
            self.messaggio("info", self.t("info"), self.t("no_wcs"))
            return
        shape = self.a0.shape[:2]
        geom = self.geom0
        def lavoro():
            return C.oggetti_nel_campo(w, shape, timeout=60)
        def fatto(r):
            ogg, errori = r
            self.oggetti = []
            for o in ogg:
                xo, yo = geom.orig_da_arr(o["x"], o["y"])
                self.oggetti.append(dict(xo=float(xo), yo=float(yo), nome=o.get("nome", "?"), tipo=o.get("tipo", "")))
            self.disegna()
            self.stato(self.t("status_objects_n", n=len(self.oggetti)))
            if errori and not self.oggetti:
                self.messaggio("warning", self.t("warn"), self.t("objects_err", e="\n".join(errori)))
        self.in_thread(lavoro, fatto, testo=self.t("status_objects"))

    # ------------------------------------------------------------------ strumenti esterni
    def controllo_automatico(self):
        """Una volta al giorno: cerca le versioni nuove e aggiorna da solo gli
        strumenti già installati (se l'opzione è attiva)."""
        if getattr(self, "_autotest", False) or not S.controllo_dovuto():
            return
        if not any(st.versione_installata() for st in S.STRUMENTI.values()):
            return
        self.controlla_strumenti(mostra=False, aggiorna=bool(self.imp.get("aggiorna_strumenti", True)))

    def controlla_strumenti(self, mostra=True, aggiorna=False, dopo=None):
        def fatto(r):
            nuovi = [S.STRUMENTI[k] for k, v in r.items() if v["update"]]
            if aggiorna and nuovi:
                for st in nuovi:
                    self.installa_strumento(st.id, chiedi=False, info=r[st.id]["info"])
            elif nuovi:
                self.stato(self.t("tools_updates", n=", ".join(st.nome for st in nuovi)))
            if mostra:
                righe = []
                for k, v in r.items():
                    st = S.STRUMENTI[k]
                    righe.append(f"{st.nome}: {self.t('tools_installed')} {v['installed'] or '—'}, "
                                 f"{self.t('tools_latest')} {v['latest'] or '?'}"
                                 + (f"  ({v['error']})" if v["error"] else ""))
                self.messaggio("info", self.t("menu_tools"), "\n".join(righe))
            if dopo:
                dopo(r)
        self.in_thread(S.controlla_tutti, fatto, lambda e: self.stato(f"{self.t('menu_tools')}: {e}"),
                       testo=self.t("tools_checking"))

    def offri_installazione(self, sid):
        st = S.STRUMENTI[sid]
        if self.messaggio("question", self.t("menu_tools"), self.t("tool_ask", n=st.nome, l=st.licenza, s=st.sito)):
            self.installa_strumento(sid, chiedi=False)

    def installa_strumento(self, sid, chiedi=True, info=None, dopo=None):
        st = S.STRUMENTI[sid]
        if chiedi and not self.messaggio("question", self.t("menu_tools"),
                                         self.t("tool_ask", n=st.nome, l=st.licenza, s=st.sito)):
            return
        self.barra_pr.configure(value=0)
        def lavoro():
            return st.installa(info, progresso=self.progresso, log=lambda t: self.coda.put(lambda: self.stato(t)))
        def fatto(exe):
            self.stato(self.t("tool_done", n=st.nome, v=st.versione_installata()))
            if self.op and getattr(O.OPERAZIONI[self.op], "strumento", None) == sid:
                self.scegli_op(self.op, self.par, self.modifica)
            if getattr(self, "_fin_strumenti", None) is not None:
                self._aggiorna_finestra_strumenti()
            if dopo:
                dopo(exe)
        self.in_thread(lavoro, fatto, lambda e: self.messaggio("error", self.t("err"),
                                                                 self.t("tool_failed", n=st.nome, e=e)),
                       testo=self.t("tool_installing", n=st.nome))

    def finestra_strumenti(self):
        if getattr(self, "_fin_strumenti", None) is not None:
            try:
                self._fin_strumenti.lift()
                return
            except tk.TclError:
                pass
        d = tk.Toplevel(self)
        d.title(self.t("tools_window"))
        d.transient(self)
        d.geometry(f"{int(900 * self.scala)}x{int(430 * self.scala)}")
        self._fin_strumenti = d
        d.protocol("WM_DELETE_WINDOW", lambda: (setattr(self, "_fin_strumenti", None), d.destroy()))
        fr = ttk.Frame(d, padding=12)
        fr.pack(fill="both", expand=True)
        self.adattabile(ttk.Label(fr, text=self.t("tools_intro", d=S.cartella_base()))).pack(fill="x", pady=(0, 8))
        self._righe_strumenti = ttk.Frame(fr)
        self._righe_strumenti.pack(fill="both", expand=True)
        bt = ttk.Frame(fr)
        bt.pack(fill="x", pady=(8, 0))
        ttk.Button(bt, text=self.t("tools_check"), command=lambda: self.controlla_strumenti(
            mostra=False, dopo=lambda r: self._aggiorna_finestra_strumenti(r))).pack(side="left")
        self._aggiorna_finestra_strumenti()

    def _aggiorna_finestra_strumenti(self, risultati=None):
        if risultati is not None:
            self._ultimo_controllo = risultati
        r = getattr(self, "_ultimo_controllo", {}) or {}
        cont = getattr(self, "_righe_strumenti", None)
        if cont is None:
            return
        try:
            for w in cont.winfo_children():
                w.destroy()
        except tk.TclError:
            return
        for i, (sid, st) in enumerate(S.STRUMENTI.items()):
            v = st.versione_installata()
            ult = (r.get(sid) or {}).get("latest")
            ttk.Label(cont, text=st.nome, font=self.font(0, True)).grid(row=2 * i, column=0, sticky="w", pady=(6, 0))
            ttk.Label(cont, text=f"{self.t('tools_installed')}: {v or '—'}   {self.t('tools_latest')}: {ult or '?'}"
                      ).grid(row=2 * i, column=1, sticky="w", padx=12, pady=(6, 0))
            ttk.Label(cont, text=f"{self.t('tool_desc_' + sid)} · {st.licenza}", foreground="#555"
                      ).grid(row=2 * i + 1, column=0, columnspan=2, sticky="w")
            testo = self.t("tool_update") if (v and ult and S._confronta_versioni(ult, v) > 0) else \
                (self.t("tool_reinstall") if v else self.t("tool_install"))
            ttk.Button(cont, text=testo, command=lambda s_=sid: self.installa_strumento(s_)).grid(
                row=2 * i, column=2, padx=4, pady=(6, 0))
            if v:
                ttk.Button(cont, text=self.t("tool_remove"), command=lambda s_=st: (
                    s_.rimuovi(), self._aggiorna_finestra_strumenti())).grid(row=2 * i, column=3, pady=(6, 0))
        cont.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------ ricette
    def ricetta(self):
        r = C.ricetta_vuota(self.sorgente, self.zona, self.fattore)
        r["steps"] = json.loads(json.dumps(self.passi))
        return r

    def salva_ricetta(self):
        if not self.passi:
            return
        base = Path(getattr(self.sorgente, "originale", self.sorgente.percorso)).stem if self.sorgente else "fitslab"
        p = chiedi_salvataggio(self, self.t("save_recipe"), self.imp.get("ultima_cartella", Path.home()),
                               f"{base}_recipe.json")
        if p:
            Path(p).write_text(json.dumps(self.ricetta(), indent=1, ensure_ascii=False), encoding="utf-8")

    def carica_ricetta(self):
        p = chiedi_file(self, self.t("open_recipe"), self.imp.get("ultima_cartella", Path.home()),
                        [("JSON", [".json"])])
        if not p:
            return
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        self.passi = [s for s in r.get("steps", []) if s.get("op") in O.OPERAZIONI]
        if self.sorgente and r.get("zone") and Path(r.get("source", "")).name == self.sorgente.percorso.name:
            self.zona = tuple(r["zone"])
            self.carica_lavoro()
        else:
            self.cache = []
            self.ricalcola(0)
        self.stato(self.t("recipe_loaded", n=len(self.passi)))

    # ------------------------------------------------------------------ salvataggio
    def finestra_salva(self):
        if self.a0 is None:
            return
        d = tk.Toplevel(self)
        d.title(self.t("save_title"))
        d.transient(self)
        fr = ttk.Frame(d, padding=12)
        fr.pack(fill="both", expand=True)
        s = self.sorgente
        orig = Path(getattr(s, "originale", s.percorso))
        var_c = tk.StringVar(value=str(orig.parent / (orig.name.split(".")[0] + "_fitslab")))
        ttk.Label(fr, text=self.t("save_folder")).grid(row=0, column=0, sticky="w")
        ttk.Entry(fr, textvariable=var_c, width=60).grid(row=1, column=0, sticky="ew")
        ttk.Button(fr, text=self.t("change"), command=lambda: var_c.set(
            chiedi_cartella(d, self.t("save_folder"), var_c.get()) or var_c.get())).grid(row=1, column=1, padx=4)
        ttk.Label(fr, text=self.t("save_quality"), font=self.font(0, True)).grid(row=2, column=0, sticky="w", pady=(10, 0))
        var_q = tk.StringVar(value="full" if self.fattore > 1 else "work")
        self.Scelta(fr, text=self.t("q_full"), value="full", variable=var_q).grid(row=3, column=0, sticky="w")
        self.Scelta(fr, text=self.t("q_work"), value="work", variable=var_q).grid(row=4, column=0, sticky="w")
        ttk.Label(fr, text=self.t("formats"), font=self.font(0, True)).grid(row=5, column=0, sticky="w", pady=(10, 0))
        scelti = set(self.imp.get("formati", []))
        var_f = {}
        for i, k in enumerate(C.FORMATI):
            var_f[k] = tk.BooleanVar(value=k in scelti)
            self.Spunta(fr, text=self.t(f"f_{k}"), variable=var_f[k]).grid(row=6 + i, column=0, sticky="w")
        var_firma = tk.BooleanVar(value=bool(self.imp.get("firma", True)))
        self.Spunta(fr, text=self.t("signature"), variable=var_firma).grid(row=20, column=0, sticky="w", pady=(8, 0))
        def salva():
            formati = [k for k, v in var_f.items() if v.get()]
            self.imp["formati"] = formati
            self.imp["firma"] = var_firma.get()
            C.salva_impostazioni(self.imp)
            d.destroy()
            self.esegui_salvataggio(var_c.get(), formati, var_q.get() == "full", var_firma.get())
        ttk.Button(fr, text=self.t("save_now"), command=salva).grid(row=30, column=0, sticky="e", pady=(12, 0))

    def esegui_salvataggio(self, cartella, formati, piena, firma, dopo=None):
        ricetta = self.ricetta()
        s = self.sorgente
        self.barra_pr.configure(value=0)
        a0, g0, passi, cache = self.a0, self.geom0, json.loads(json.dumps(self.passi)), list(self.cache)
        def lavoro():
            if piena and self.fattore > 1:
                return C.elabora_e_salva(s, ricetta, cartella, formati, firma, 1, progresso=self.progresso)
            return C.salva_uscite(s, ricetta, a0, g0, passi, cache, cartella, formati, firma, self.progresso)
        def fatto(scritti):
            self.stato(self.t("status_saved", n=len(scritti), d=cartella))
            if dopo:
                dopo(scritti)
            elif not getattr(self, "_autotest", False):
                self.messaggio("info", self.t("saved_title"), self.t("saved_text", f="\n".join(str(p) for p in scritti)))
        self.in_thread(lavoro, fatto, testo=self.t("saving"))

    def esci(self):
        C.salva_impostazioni(self.imp)
        self.destroy()


# =============================================================================
# AUTOTEST E DIAGNOSTICA
# =============================================================================
def _fits_prova(cartella, sip=True, forma=(700, 900)):
    """FITS sintetico: stelle, gradiente, pixel caldi, buco, bordo vuoto, WCS TAN (+SIP)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    rng = np.random.default_rng(3)
    ny, nx = forma
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = 200 + 0.05 * xx + 0.02 * yy
    stelle = []
    for _ in range(150):
        x, y, f = rng.uniform(15, nx - 15), rng.uniform(15, ny - 15), rng.uniform(200, 5000)
        img = img + f * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.6 ** 2))
        stelle.append((x, y))
    img = img + rng.normal(0, 6, img.shape)
    img.ravel()[rng.integers(0, nx * ny, 150)] += 3000
    img[300:306, 400:408] = np.nan
    img[:, :25] = 0
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"] if sip else ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [202.4696, 47.1952]            # M 51
    w.wcs.crpix = [nx / 2 + 0.3, ny / 2 - 0.7]
    w.wcs.cd = np.array([[-1.1e-4, 1.5e-5], [1.4e-5, 1.1e-4]])
    h = w.to_header()
    if sip:
        h["CTYPE1"], h["CTYPE2"] = "RA---TAN-SIP", "DEC--TAN-SIP"
        h["A_ORDER"] = 2
        h["B_ORDER"] = 2
        h["A_2_0"], h["A_0_2"], h["A_1_1"] = 2e-6, -1e-6, 5e-7
        h["B_2_0"], h["B_0_2"], h["B_1_1"] = -1e-6, 3e-6, 1e-6
    h["TELESCOP"] = "JWST"
    h["INSTRUME"] = "NIRCAM"
    h["FILTER"] = "F200W"
    h["BUNIT"] = "MJy/sr"
    h["EXPTIME"] = 1234.5
    h["DATE-OBS"] = "2026-01-02"
    h["OBJECT"] = "TEST"
    p = Path(cartella) / "prova_jwst_f200w.fits"
    fits.PrimaryHDU(img.astype(np.float32), header=h).writeto(p, overwrite=True)
    return p, stelle


def autotest():
    import tempfile
    risultati = []
    def v(ok, testo):
        risultati.append(bool(ok))
        print(f"[{'ok' if ok else 'ERRORE'}] {testo}", flush=True)
    print(f"{C.NOME} {C.VERSIONE} - autotest  ({C.COPYRIGHT_ASCII})")
    print(f"Python {sys.version.split()[0]}, numpy {np.__version__}", flush=True)
    tmp = Path(tempfile.mkdtemp(prefix="fitslab_test_"))
    C.FILE_IMPOSTAZIONI = tmp / "impostazioni_prova.json"      # non tocca quelle vere
    try:
        from astropy.io import fits
        from astropy.wcs import WCS
        import astropy
        print(f"astropy {astropy.__version__}")
        # --- immagine con SIP
        p, stelle = _fits_prova(tmp)
        s = C.Sorgente(p)
        m = s.meta()
        v(m["mission"] == "JWST" and m["filter"] == "F200W" and abs(m["wavelength_um"] - 2.0) < 1e-6,
          f"missione/filtro/lunghezza: {m['mission']} {m['filter']} {m['wavelength']}")
        v(s.wcs is not None and s.wcs.sip is not None, "WCS con distorsione SIP letto")
        a = s.leggi()
        v(np.isnan(a[:, :25]).all() and np.isfinite(a[:, 30:]).mean() > 0.99, "bordo a zero -> NaN")
        # WCS dopo zona + riduzione + ritaglio + binning: stesse coordinate in cielo
        z = (40, 640, 60, 860)
        a2 = s.leggi(z, 2)
        g = O.Geometria(z[0], z[2], 2, 1)
        passi = [dict(op="crop", params=dict(mode="rect", rect=[101, 77, 700, 560])),
                 dict(op="bin", params=dict(factor=3))]
        cache = C.esegui(a2, g, passi, m)
        gf = cache[-1][1]
        wf = C.wcs_per_geometria(s.wcs, gf, cache[-1][0].shape)
        errori = []
        for xa, ya in ((0, 0), (10.3, 20.7), (cache[-1][0].shape[1] - 1, cache[-1][0].shape[0] - 1)):
            xo, yo = gf.orig_da_arr(xa, ya)
            c1 = s.wcs.pixel_to_world(xo, yo)
            c2 = wf.pixel_to_world(xa, ya)
            errori.append(c1.separation(c2).arcsec)
        scala = C.scala_arcsec(s.wcs)
        v(max(errori) < 0.001 * scala, f"WCS dopo zona/riduzione/ritaglio/binning (con SIP): errore massimo "
                                       f"{max(errori) / scala:.2e} pixel")
        print(f"     forma finale {cache[-1][0].shape}, geometria {gf.as_dict()}")
        # --- ricetta completa e uscite
        passi = [dict(op="pixels", params=dict(cosmic=True)),
                 dict(op="background", params=dict(method="poly", degree=2, points=[[100, 100], [800, 600]])),
                 dict(op="denoise", params=dict(algorithm="multiscale")),
                 dict(op="crop", params=dict(mode="data")),
                 dict(op="stretch", params=dict(method="auto")),
                 dict(op="curves", params=dict(curve=O.PRESET_CURVE["s_soft"], contrast=10)),
                 dict(op="palette", params=dict(palette="gold"))]
        ric = C.ricetta_vuota(s)
        ric["steps"] = passi
        t0 = time.time()
        scritti = C.elabora_e_salva(s, ric, tmp / "uscita", list(C.FORMATI), True, 1, log=lambda *x: None)
        v(len(scritti) == 8, f"uscite scritte ({len(scritti)}) in {time.time() - t0:.1f} s: "
                             + ", ".join(Path(x).name for x in scritti))
        sci = tmp / "uscita" / "science" / "prova_jwst_f200w_science.fits"
        with fits.open(sci) as h:
            hd = h[0].header
            ws = WCS(hd)
            d = h[0].data
            v(hd.get("BUNIT") == "MJy/sr" and hd.get("FILTER") == "F200W" and hd.get("FLABDNS") is True,
              "FITS scientifico: unità, filtro e avviso di riduzione del rumore conservati")
            v(any("background" in str(x) for x in hd["HISTORY"]), "HISTORY con i passi (in inglese)")
            x0 = 25          # il ritaglio «data» ha tolto le 25 colonne vuote
            e = max(s.wcs.pixel_to_world(xx + x0, yy).separation(ws.pixel_to_world(xx, yy)).arcsec
                    for xx, yy in ((0, 0), (300.5, 200.2), (d.shape[1] - 1, d.shape[0] - 1)))
            v(e < 1e-4, f"coordinate del FITS scientifico ritagliato: errore {e:.2e}\"")
        with fits.open(tmp / "uscita" / "aesthetic" / "prova_jwst_f200w_aesthetic.fits") as h:
            v(h[0].data.shape[0] == 3 and h[0].header.get("FLABTYPE") == "aesthetic", "FITS estetico RGB con WCS")
        from PIL import Image
        im = Image.open(tmp / "uscita" / "aesthetic" / "prova_jwst_f200w_aesthetic.png")
        v(im.mode == "RGB" and im.size == (875, 700), f"PNG {im.mode} {im.size}")
        im = Image.open(tmp / "uscita" / "aesthetic" / "prova_jwst_f200w_aesthetic_forum.jpg")
        v((tmp / "uscita" / "aesthetic" / "prova_jwst_f200w_aesthetic_forum.jpg").stat().st_size < 1.05e6,
          "JPEG per forum sotto 1 MB")
        q = C.qualita(a, m)
        chiavi = {x["key"]: x for x in q}
        v("fwhm" in chiavi and abs(chiavi["fwhm"]["value"] - 3.77) < 0.6,
          f"referto: FWHM {chiavi.get('fwhm', {}).get('text')} (atteso ~3,8 px), stelle {chiavi.get('stars', {}).get('text')}")
        v(chiavi["hot_pixels"]["value"] > 0 and chiavi["holes"]["value"] > 0, "referto: pixel caldi e buco trovati")
        # ogni operazione da sola, con tutte le scelte
        for op in O.OPERAZIONI.values():
            if getattr(op, "lento", False):
                continue          # programmi esterni: provati più avanti con programmi finti
            for prm in op.parametri:
                for scelta in prm.scelte or [None]:
                    par = op.predefiniti()
                    if scelta is not None:
                        par[prm.id] = scelta
                    if op.id == "denoise" and par.get("algorithm") == "nlmeans":
                        continue
                    try:
                        ctx = O.Contesto(O.Geometria(), dict(m))
                        r = op.applica(a, par, ctx)
                        assert r.shape[0] > 1
                    except Exception as e:
                        v(False, f"operazione {op.id} {prm.id}={scelta}: {e}")
        v(True, "tutte le operazioni con tutte le scelte")
        # deconvoluzione: stelle più strette, flusso e fondo conservati
        g1 = O.Geometria()
        dc = O.OPERAZIONI["deconv"].applica(a, O.OPERAZIONI["deconv"].predefiniti(), O.Contesto(g1))
        f0, f1 = C.misura_stelle(a, O.rumore_pixel(a)), C.misura_stelle(dc, O.rumore_pixel(a))
        m0 = O.statistiche(a)[0]
        fl = np.nansum(dc - m0) / np.nansum(a - m0)
        v(f1.get("n", 0) > 0 and f1["fwhm_px"] < 0.9 * f0["fwhm_px"] and abs(fl - 1) < 0.03,
          f"deconvoluzione: FWHM {f0.get('fwhm_px', 0):.2f} -> {f1.get('fwhm_px', 0):.2f} px, flusso x{fl:.3f}")
        v("saspro" not in sys.modules["fitslab_strumenti"].STRUMENTI, "solo strumenti gratuiti (niente Seti Astro Suite Pro)")
        # --- radio 4D e cubo
        h4 = fits.Header()
        w4 = WCS(naxis=4)
        w4.wcs.ctype = ["RA---SIN", "DEC--SIN", "FREQ", "STOKES"]
        w4.wcs.crval = [83.8, -5.4, 1.4e9, 1]
        w4.wcs.cdelt = [-4e-4, 4e-4, 1e6, 1]
        w4.wcs.crpix = [100, 100, 1, 1]
        h4.extend(w4.to_header())
        h4["TELESCOP"] = "VLA"
        h4["BUNIT"] = "JY/BEAM"
        h4["BMAJ"] = 45 / 3600
        dati = np.random.default_rng(1).normal(0, 1, (1, 1, 200, 200)).astype(np.float32)
        fits.PrimaryHDU(dati, header=h4).writeto(tmp / "nvss.fits")
        s4 = C.Sorgente(tmp / "nvss.fits")
        m4 = s4.meta()
        v(s4.shape == (200, 200) and s4.wcs is not None and m4["mission"] == "VLA" and "GHz" in m4["wavelength"]
          and O.modo_binning_auto(m4["unit"]) == "mean",
          f"radio 4D: {m4['mission']} {m4['wavelength']} {m4['unit']} (binning: media)")
        cubo = np.random.default_rng(2).normal(5, 1, (6, 50, 60)).astype(np.float32)
        hc = WCS(naxis=3)
        hc.wcs.ctype = ["RA---TAN", "DEC--TAN", "VRAD"]
        hc.wcs.crval = [10, 20, 0]
        hc.wcs.cdelt = [-1e-3, 1e-3, 1000]
        hc.wcs.crpix = [30, 25, 1]
        fits.PrimaryHDU(cubo, header=hc.to_header()).writeto(tmp / "cubo.fits")
        info = C.ispeziona(tmp / "cubo.fits")
        sc = C.Sorgente(tmp / "cubo.fits", dict(cube_mode="mean", cube_from=1, cube_to=4))
        v(info["kind"] == "cube" and info["planes"] == 6 and np.allclose(sc.leggi(), cubo[1:5].mean(0), atol=1e-5),
          "cubo: media dei piani 1-4")
        # --- Planck / HEALPix
        import fitslab_healpix as HP
        for nested in (False, True):
            ns = 64
            th, ph = HP.pix2ang_ring(ns, np.arange(12 * ns * ns))
            valori = np.cos(th).astype(np.float32)           # = sin(b)
            if nested:
                ordine = HP.ang2pix(ns, th, ph, nested=True)
                vn = np.empty_like(valori)
                vn[ordine] = valori
                valori = vn
            col = fits.Column(name="I_STOKES", format="E", unit="K_CMB", array=valori)
            hdu = fits.BinTableHDU.from_columns([col])
            hdu.header.update(PIXTYPE="HEALPIX", ORDERING="NESTED" if nested else "RING", NSIDE=ns,
                              COORDSYS="GALACTIC", INDXSCHM="IMPLICIT")
            pp = tmp / f"HFI_SkyMap_143_prova_{'nest' if nested else 'ring'}.fits"
            h0 = fits.Header()
            h0["TELESCOP"] = "PLANCK"
            fits.HDUList([fits.PrimaryHDU(header=h0), hdu]).writeto(pp)
            sh = C.Sorgente(pp, dict(projection="cutout", lon=30, lat=20, size_deg=6, frame="G"))
            a_h = sh.leggi()
            centro = a_h[a_h.shape[0] // 2, a_h.shape[1] // 2]
            v(abs(centro - math.sin(math.radians(20))) < 0.03 and sh.meta()["mission"] == "Planck",
              f"Planck {'NESTED' if nested else 'RING'}: valore al centro {centro:.3f} (atteso 0,342), "
              f"{sh.meta()['filter']}, {sh.meta()['wavelength']}")
            sc_eq = C.Sorgente(pp, dict(projection="cutout", lon=229.0, lat=11.0, size_deg=4, frame="C"))
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            b = SkyCoord(229 * u.deg, 11 * u.deg).galactic.b.deg
            a_e = sc_eq.leggi()
            v(abs(a_e[a_e.shape[0] // 2, a_e.shape[1] // 2] - math.sin(math.radians(b))) < 0.03,
              f"Planck in coordinate equatoriali (b = {b:.1f}°)")
        sa = C.Sorgente(pp, dict(projection="allsky", width=800))
        v(sa.leggi().shape == (400, 800) and np.isfinite(sa.leggi()).mean() > 0.7, "Planck: tutto il cielo (Mollweide)")
        try:
            import astropy_healpix as AH
            rng = np.random.default_rng(5)
            lon = rng.uniform(0, 360, 20000)
            lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 20000)))
            for ordine in ("ring", "nested"):
                rif = AH.lonlat_to_healpix(lon * u.deg, lat * u.deg, 2048, order=ordine)
                mio = HP.ang2pix(2048, np.radians(90 - lat), np.radians(lon), nested=ordine == "nested")
                v((rif == mio).mean() > 0.9999, f"HEALPix {ordine} uguale ad astropy-healpix: {(rif == mio).mean() * 100:.3f}%")
        except ImportError:
            print("(astropy-healpix non installato: confronto saltato)")
        # --- riconoscimento delle missioni
        for h, nome, atteso in (({}, "EUC_MER_BGSUB-MOSAIC-NIR-H_TILE102018211.fits", "Euclid"),
                                ({"TELESCOP": "ROMAN"}, "x.fits", "Roman"), ({}, "hlsp_x_hst_acs_f814w_drz.fits", "Hubble"),
                                ({"TELESCOP": "PLANCK"}, "x.fits", "Planck"), ({"TELESCOP": "CHANDRA"}, "x.fits", "Chandra"),
                                ({}, "tess2024-s0075-1-1-0283-s_ffic.fits", "TESS"), ({"TELESCOP": "2MASS"}, "x", "2MASS")):
            r = C.riconosci_missione(h, nome)
            v(r == atteso, f"missione da {nome or h}: {r}")
        v(C.filtro_euclid("EUC_MER_BGSUB-MOSAIC-NIR-H_TILE1.fits", {}) == "NIR_H", "filtro Euclid NIR_H dal nome")
        # --- strumenti esterni (1.1): versioni, scelta del file, estrazione, esecuzione con programmi finti
        import fitslab_strumenti as S
        import fitslab_esterni as E
        import zipfile
        v(S._confronta_versioni("2.6.2-0242", "2.6.1-0232") > 0 and S._confronta_versioni("3.1.0rc2", "3.1.0") < 0
          and S._confronta_versioni("3.0.2", "3.0.2") == 0, "confronto delle versioni")
        nomi = [{"name": n} for n in ("GraXpert-win64.exe", "GraXpert-linux.zip", "GraXpert-macos-arm64.dmg",
                                       "graxpert_3.0.2_amd64.deb")]
        scelto = S.GraXpert()._scegli_asset(nomi)["name"]
        v(scelto == {"windows-x64": "GraXpert-win64.exe", "linux-x64": "GraXpert-linux.zip"}.get(S.piattaforma(), scelto),
          f"GraXpert: file scelto per {S.piattaforma()}: {scelto}")
        zp = tmp / "prova.zip"
        with zipfile.ZipFile(zp, "w") as z:
            zi = zipfile.ZipInfo("pacco/starnet2")
            zi.external_attr = 0o755 << 16
            z.writestr(zi, "#!/bin/sh\n")
        S.estrai(zp, tmp / "estratto")
        v(os.access(tmp / "estratto" / "pacco" / "starnet2", os.X_OK) or sys.platform.startswith("win"),
          "estrazione dello zip con il permesso di esecuzione")
        with zipfile.ZipFile(tmp / "cattivo.zip", "w") as z:
            z.writestr("../fuori.txt", "x")
        try:
            S.estrai(tmp / "cattivo.zip", tmp / "estratto2")
            v(False, "zip con percorsi pericolosi rifiutato")
        except ValueError:
            v(True, "zip con percorsi pericolosi rifiutato")
        if not sys.platform.startswith("win"):
            os.environ["FITSLAB_STRUMENTI"] = str(tmp / "strumenti")
            finti = {"starnet2": ('#!/bin/sh\nin=""; out=""; uns=""\n'
                                  'while [ $# -gt 0 ]; do case "$1" in --input) in="$2"; shift;; --output) out="$2"; shift;; '
                                  '--unscreen) uns="$2"; shift;; esac; shift; done\n'
                                  'cp "$in" "$out"; [ -n "$uns" ] && cp "$in" "$uns"\necho \'{"progress": 1.0}\'\n'),
                     "graxpert": ('#!/bin/sh\nin="$1"; shift; out=""\n'
                                  'while [ $# -gt 0 ]; do case "$1" in -output) out="$2"; shift;; esac; shift; done\n'
                                  'cp "$in" "$out.fits"\n')}
            for sid, testo in finti.items():
                st = S.STRUMENTI[sid]
                d = st.cartella / "0.0-test"
                d.mkdir(parents=True, exist_ok=True)
                exe = d / sid
                exe.write_text(testo)
                S.rendi_eseguibile(exe)
                st._salva_stato(dict(version="0.0-test", executable=str(exe)))
            ctx = O.Contesto(O.Geometria(), dict(m))
            for op_id, par in (("starnet", {}), ("starnet", {"result": "stars", "upsample": True}),
                               ("graxpert_bg", {}), ("graxpert_denoise", {"strength": 0.3})):
                ctx = O.Contesto(O.Geometria(), dict(m))
                t0 = time.time()
                r = O.applica_passo(a, dict(op=op_id, params=par), ctx)
                uguale = np.allclose(np.nan_to_num(r), np.nan_to_num(a), rtol=1e-5, atol=1e-3)
                v(r.shape == a.shape and uguale and np.isnan(r[:, :25]).all() and ctx.lineare,
                  f"{op_id} {par}: programma eseguito, risultato riletto ({time.time() - t0:.2f} s)")
            t0 = time.time()
            ctx = O.Contesto(O.Geometria(), dict(m))
            O.applica_passo(a, dict(op="graxpert_denoise", params={"strength": 0.3}), ctx)
            v(time.time() - t0 < 0.5 and "denoised" in ctx.avvisi, "risultato in memoria al secondo calcolo, avviso rumore")
            rgb = np.stack([a, a * 0.5, a * 0.2], axis=2)
            r = O.applica_passo(rgb, dict(op="starnet", params={}), O.Contesto(O.Geometria(), dict(m)))
            v(r.shape == rgb.shape and np.allclose(np.nan_to_num(r), np.nan_to_num(rgb), rtol=1e-5, atol=1e-3),
              "StarNet2 su immagine a colori (cubo 3 x H x W)")
            gx = np.where(np.isfinite(a), a, np.nanmedian(a)).astype(np.float32)
            fondo_tolto = gx - 0.02 * np.arange(gx.shape[1])[None, :]
            norm = (fondo_tolto - fondo_tolto.min()) / (fondo_tolto.max() - fondo_tolto.min())
            r = E._stessa_scala(norm.astype(np.float32), gx)
            v(abs(np.percentile(r, 99.9) - np.percentile(fondo_tolto, 99.9)) < 0.01 * np.percentile(fondo_tolto, 99.9),
              "scala di GraXpert (uscita normalizzata 0-1) riportata ai valori originali")
            ric2 = C.ricetta_vuota(s)
            ric2["steps"] = [dict(op="graxpert_bg", params={}), dict(op="stretch", params={})]
            scr = C.elabora_e_salva(s, ric2, tmp / "uscita_esterni", ["science_fits", "png"], True, 1,
                                    log=lambda *x: None)
            with fits.open(scr[0]) as h:
                v(any("GraXpert" in str(x) for x in h[0].header["HISTORY"]), "ricetta con GraXpert salvata (HISTORY)")
            S.STRUMENTI["starnet2"].rimuovi()
            try:
                O.applica_passo(a, dict(op="starnet", params={"upsample": False, "stride": 64}), O.Contesto(O.Geometria(), dict(m)))
                v(False, "strumento mancante segnalato")
            except E.StrumentoMancante:
                v(True, "strumento mancante segnalato (la finestra propone di installarlo)")
        # --- finestra
        app = FitsLab()
        app._autotest = True
        app.update()
        app.apri(p, {})
        _aspetta(app, lambda: app.a0 is not None, 60)
        v(app.a0 is not None and app.v_sx.immagine is not None, "finestra: file aperto e mostrato")
        app.update()
        app.update_idletasks()
        larghe = app.cv_sx.winfo_width() + app.cv_dx.winfo_width()
        v(larghe > 0.5 * app.winfo_width(), f"finestra: le immagini occupano {larghe} px su {app.winfo_width()} "
                                            f"(pannello {app.cont_pannello.winfo_width()} px)")
        app.scegli_op("stretch")
        _aspetta(app, lambda: app.anteprima is not None, 30)
        v(app.anteprima is not None, "finestra: anteprima dal vivo dello stiramento")
        app.applica()
        _aspetta(app, lambda: len(app.cache) == 1, 30)
        app.scegli_op("curves")
        app.par["curve"] = [[0, 0], [0.3, 0.4], [1, 1]]
        app.applica()
        _aspetta(app, lambda: len(app.cache) == 2, 30)
        v(len(app.passi) == 2 and app.lista.size() >= 2, "finestra: due passi nella ricetta")
        app.imp["lingua"] = "en"
        app.var_lingua.set("en")
        app.cambia_lingua()
        _aspetta(app, lambda: app.v_sx.immagine is not None, 10)
        v(app.t("apply") == "Apply" and app.v_sx.immagine is not None, "finestra: lingua inglese")
        app.imp["lingua"] = "it"
        app.var_lingua.set("it")
        app.cambia_lingua()
        app.applica_testo(175)
        app.update()
        app.zoom_a(2.0, 100, 100)
        app._seleziona_passo()
        app.misura_risultato()
        _aspetta(app, lambda: bool(app.q_ris), 30)
        v(bool(app.q_ris), "finestra: referto del risultato")
        fatto = []
        app.esegui_salvataggio(str(tmp / "uscita_finestra"), ["science_fits", "png"], False, True, dopo=fatto.append)
        _aspetta(app, lambda: bool(fatto), 60)
        v(fatto and len(fatto[0]) >= 3, "finestra: salvataggio")
        if not sys.platform.startswith("win"):
            app.scegli_op("graxpert_bg")
            app.update()
            v(app.anteprima is None, "finestra: niente anteprima automatica per i programmi lenti")
            app.programma_anteprima(subito=True, forza=True)
            _aspetta(app, lambda: app.anteprima is not None, 30)
            v(app.anteprima is not None, "finestra: anteprima di GraXpert a richiesta")
            app.annulla_op()
        app.finestra_strumenti()
        app.update()
        v(app._fin_strumenti is not None, "finestra: Strumenti esterni")
        app.applica_testo(150)
        app.imp["testo"] = C.carica_impostazioni().get("testo", 150)
        app.destroy()
    except Exception:
        traceback.print_exc()
        risultati.append(False)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(risultati)
    print(f"\n{sum(risultati)}/{len(risultati)} prove superate -> {'TUTTO OK' if ok else 'CI SONO ERRORI'}")
    return 0 if ok else 1


def _aspetta(app, condizione, secondi):
    t = time.time()
    while time.time() - t < secondi:
        app.update()
        if condizione():
            return True
        time.sleep(0.02)
    return False


def diagnostica():
    import tempfile
    print(f"{C.NOME} {C.VERSIONE} - diagnostica di rete")
    tmp = Path(tempfile.mkdtemp(prefix="fitslab_diag_"))
    try:
        p, _ = _fits_prova(tmp, sip=False)
        s = C.Sorgente(p)
        t = time.time()
        ogg, errori = C.oggetti_nel_campo(s.wcs, s.shape, timeout=60)
        print(f"[{'ok' if ogg else 'ERRORE'}] oggetti nel campo di M 51: {len(ogg)} in {time.time() - t:.1f} s "
              f"({', '.join(o['nome'] for o in ogg[:5])})")
        for e in errori:
            print("   ", e)
        nomi = [o["nome"] for o in ogg]
        print(f"[{'ok' if 'M 51' in nomi[:3] else 'ERRORE'}] M 51 tra i primi tre nomi: {', '.join(nomi[:3])}")
        try:
            ra, de = C.risolvi_nome("M 51")
            print(f"[ok] NED ObjectLookup: M 51 = {ra:.4f} {de:+.4f}")
        except Exception as e:
            print(f"[ERRORE] NED ObjectLookup: {e}")
        import fitslab_strumenti as S
        for sid, st in S.STRUMENTI.items():
            try:
                t = time.time()
                info = st.ultima()
                print(f"[ok] {st.nome}: ultima versione {info['version']} ({time.time() - t:.1f} s) -> "
                      f"{info.get('file_name')} {('SHA-256 ' + info['sha256'][:12] + '…') if info.get('sha256') else ''}")
            except Exception as e:
                print(f"[ERRORE] {st.nome}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if ("--autotest" in sys.argv or "--diagnostica" in sys.argv) and sys.stdout is None:
        # eseguibile Windows senza console: i risultati vanno in un file accanto
        nome = "autotest_risultato.txt" if "--autotest" in sys.argv else "diagnostica_risultato.txt"
        sys.stdout = sys.stderr = open(nome, "w", encoding="utf-8")
    if "--autotest" in sys.argv:
        sys.exit(autotest())
    if "--diagnostica" in sys.argv:
        diagnostica()
        sys.exit(0)
    file_iniziale = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    FitsLab(file_iniziale).mainloop()
