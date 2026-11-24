# -*- coding: utf-8 -*-
"""
FITS LAB - STRUMENTI ESTERNI GRATUITI
=====================================
Scarica, installa, aggiorna ed esegue programmi gratuiti di altri autori, sempre
all'ultima versione e dai siti ufficiali (FITS Lab non li ridistribuisce):

  - StarNet2 (Mikita Misiura, starnetastro.com): rimozione delle stelle.
    Ultima versione dal feed JSON ufficiale, con controllo SHA-256.
  - GraXpert (github.com/Steffenhir/GraXpert, GPL-3.0): fondo e rumore con AI.
    Ultima versione dalle pubblicazioni di GitHub.

Solo programmi gratuiti (licenza libera o freeware): niente a pagamento e niente
modelli con licenza proprietaria ristretta (per questo Cosmic Clarity è escluso).

Tutto va in una cartella dell'utente (niente sudo):
  Linux:   ~/.local/share/fitslab/strumenti/<strumento>/<versione>/
  Windows: %LOCALAPPDATA%\\FITSLab\\strumenti\\...
La variabile d'ambiente FITSLAB_STRUMENTI cambia la cartella (usata dall'autotest).

© 2026 Edoardo Battistel - FITS Lab
"""

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from datetime import date
from pathlib import Path

USER_AGENT = "FITSLab/1.7 (Python)"
TIMEOUT = 60

URL_STARNET = "https://starnetastro.com/cli-tools/latest.json"
URL_GRAXPERT = "https://api.github.com/repos/Steffenhir/GraXpert/releases"


def cartella_base():
    if os.environ.get("FITSLAB_STRUMENTI"):
        return Path(os.environ["FITSLAB_STRUMENTI"])
    if sys.platform.startswith("win"):
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "FITSLab" / "strumenti"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "fitslab" / "strumenti"


def piattaforma():
    s = sys.platform
    m = platform.machine().lower()
    if s.startswith("win"):
        return "windows-x64"
    if s == "darwin":
        return "macos-arm64" if m in ("arm64", "aarch64") else "macos-x64"
    return "linux-x64"


def ambiente_di_sistema():
    """Ambiente per lanciare programmi esterni dall'eseguibile PyInstaller:
    si ripristinano le variabili originali, altrimenti il programma esterno
    caricherebbe le librerie incluse in FITS Lab."""
    env = dict(os.environ)
    for var in ("LD_LIBRARY_PATH", "GDK_PIXBUF_MODULE_FILE", "GTK_PATH", "GIO_MODULE_DIR", "PYTHONPATH",
                "PYTHONHOME", "TCL_LIBRARY", "TK_LIBRARY", "SSL_CERT_FILE", "QT_PLUGIN_PATH"):
        orig = env.pop(var + "_ORIG", None)
        if orig is not None:
            env[var] = orig
        elif getattr(sys, "frozen", False):
            env.pop(var, None)
    return env


def _sessione():
    import requests
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _confronta_versioni(a, b):
    """-1, 0, 1 confrontando versioni come 2.6.2-0242, 3.0.2 o 3.1.0rc2
    (una versione di prova «rc», «b», «a», «dev» viene prima della definitiva)."""
    def chiave(v):
        v = str(v).strip().lower().lstrip("v")
        m = re.match(r"(\d+(?:\.\d+)*)(.*)$", v)
        if not m:
            return ((), (0, v))
        numeri = tuple(int(x) for x in m.group(1).split("."))
        numeri = numeri + (0,) * (6 - len(numeri))
        resto = m.group(2)
        mp = re.match(r"[-_.]?(dev|a|alpha|b|beta|rc|pre)\.?(\d*)", resto)
        if mp:
            ordine = {"dev": 0, "a": 1, "alpha": 1, "b": 2, "beta": 2, "pre": 3, "rc": 3}[mp.group(1)]
            return (numeri, (0, ordine, int(mp.group(2) or 0)))
        mb = re.search(r"(\d+)", resto)
        return (numeri, (1, int(mb.group(1)) if mb else 0, 0))
    ka, kb = chiave(a), chiave(b)
    return (ka > kb) - (ka < kb)


def scarica(url, destinazione, sha256=None, progresso=None, sessione=None):
    """Scarica a pezzi con avanzamento; controlla lo SHA-256 se dato."""
    s = sessione or _sessione()
    h = hashlib.sha256()
    with s.get(url, stream=True, timeout=TIMEOUT, allow_redirects=True) as r:
        r.raise_for_status()
        totale = int(r.headers.get("Content-Length") or 0)
        fatto = 0
        with open(destinazione, "wb") as f:
            for blocco in r.iter_content(1024 * 1024):
                if not blocco:
                    continue
                f.write(blocco)
                h.update(blocco)
                fatto += len(blocco)
                if progresso and totale:
                    progresso(fatto / totale)
    if sha256 and h.hexdigest().lower() != sha256.lower():
        Path(destinazione).unlink(missing_ok=True)
        raise ValueError(f"SHA-256 check failed for {url}: the download is damaged or was altered")
    return destinazione


def estrai(archivio, cartella):
    archivio = str(archivio)
    cartella = Path(cartella)
    cartella.mkdir(parents=True, exist_ok=True)
    base = cartella.resolve()
    if archivio.lower().endswith(".zip"):
        with zipfile.ZipFile(archivio) as z:
            for m in z.infolist():
                p = (cartella / m.filename).resolve()
                if not str(p).startswith(str(base)):
                    raise ValueError(f"unsafe path in archive: {m.filename}")
            z.extractall(cartella)
            # zipfile non conserva i permessi di esecuzione: si rimettono
            for m in z.infolist():
                modo = (m.external_attr >> 16) & 0o777
                if modo & 0o111:
                    p = cartella / m.filename
                    if p.is_file():
                        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    elif re.search(r"\.(tar\.gz|tgz|tar\.xz|tar)$", archivio.lower()):
        with tarfile.open(archivio) as t:
            t.extractall(cartella, filter="data")
    else:
        dest = cartella / Path(archivio).name
        shutil.copy2(archivio, dest)
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    return cartella


def rendi_eseguibile(p):
    p = Path(p)
    if p.exists() and not sys.platform.startswith("win"):
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# =============================================================================
# STRUMENTI
# =============================================================================
class Strumento:
    id = ""
    nome = ""
    autore = ""
    licenza = ""
    sito = ""

    @property
    def cartella(self):
        return cartella_base() / self.id

    def _stato_file(self):
        return self.cartella / "installato.json"

    def stato(self):
        try:
            return json.loads(self._stato_file().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _salva_stato(self, d):
        self.cartella.mkdir(parents=True, exist_ok=True)
        self._stato_file().write_text(json.dumps(d, indent=1), encoding="utf-8")

    def versione_installata(self):
        st = self.stato()
        if st.get("version") and self.eseguibile() is not None:
            return st["version"]
        return None

    def eseguibile(self):
        st = self.stato()
        p = st.get("executable")
        if p and Path(p).exists():
            return Path(p)
        return None

    def ultima(self, sessione=None):
        """{'version', 'url', 'sha256', 'file_name', 'size'}"""
        raise NotImplementedError

    def installa(self, info=None, progresso=None, log=print):
        raise NotImplementedError

    def rimuovi(self):
        if self.cartella.exists():
            shutil.rmtree(self.cartella, ignore_errors=True)

    def _pulisci_vecchie(self, tieni):
        for d in self.cartella.iterdir() if self.cartella.exists() else []:
            if d.is_dir() and d.name != tieni:
                shutil.rmtree(d, ignore_errors=True)

    def esegui(self, argomenti, timeout=3600, riga=None, cwd=None):
        """Esegue il programma; riga(testo) riceve l'uscita riga per riga."""
        exe = self.eseguibile()
        if exe is None:
            raise FileNotFoundError(f"{self.nome} is not installed")
        cmd = [str(exe)] + [str(a) for a in argomenti]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env=ambiente_di_sistema(), cwd=str(cwd or exe.parent), bufsize=1,
                                errors="replace")
        uscita = []
        t0 = time.time()
        for r in proc.stdout:
            uscita.append(r.rstrip())
            if riga:
                riga(r.rstrip())
            if time.time() - t0 > timeout:
                proc.kill()
                raise TimeoutError(f"{self.nome} took more than {timeout} s")
        proc.wait()
        if proc.returncode != 0:
            coda = "\n".join(uscita[-25:])
            raise RuntimeError(f"{self.nome} ended with error {proc.returncode}:\n{coda}")
        return "\n".join(uscita)


class StarNet(Strumento):
    id = "starnet2"
    nome = "StarNet2"
    autore = "Mikita Misiura"
    licenza = "free download (all rights reserved by the author; not redistributed by FITS Lab)"
    sito = "https://starnetastro.com/cli-tools/starnet/"

    def ultima(self, sessione=None):
        s = sessione or _sessione()
        r = s.get(URL_STARNET, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        voce = d["tools"]["starnet2"]["latest"][piattaforma()]
        pacco = voce["packages"]["archive"]
        versione = voce["version"] + (f"-{voce['build']}" if voce.get("build") else "")
        return dict(version=versione, url=pacco["url"], sha256=pacco.get("sha256"),
                    file_name=pacco.get("file_name") or Path(pacco["url"]).name, size=pacco.get("size_bytes"),
                    date=voce.get("release_date"))

    def installa(self, info=None, progresso=None, log=print):
        info = info or self.ultima()
        dest = self.cartella / info["version"]
        with tempfile.TemporaryDirectory(prefix="fitslab_dl_") as tmp:
            arch = Path(tmp) / info["file_name"]
            log(f"download {info['url']}")
            scarica(info["url"], arch, info.get("sha256"), progresso)
            if dest.exists():
                shutil.rmtree(dest)
            estrai(arch, dest)
        nome = "starnet2.exe" if sys.platform.startswith("win") else "starnet2"
        trovati = [p for p in dest.rglob(nome) if p.is_file()] or \
                  [p for p in dest.rglob("starnet*") if p.is_file() and (p.suffix in ("", ".exe"))]
        if not trovati:
            raise FileNotFoundError(f"starnet2 executable not found in {dest}")
        exe = sorted(trovati, key=lambda p: len(p.parts))[0]
        rendi_eseguibile(exe)
        self._salva_stato(dict(version=info["version"], executable=str(exe), installed=str(date.today()),
                               source=info["url"]))
        self._pulisci_vecchie(info["version"])
        return exe


class GraXpert(Strumento):
    id = "graxpert"
    nome = "GraXpert"
    autore = "Steffen Hirtle and contributors"
    licenza = "GPL-3.0 (open source)"
    sito = "https://github.com/Steffenhir/GraXpert"

    def __init__(self, prerelease=False):
        self.prerelease = prerelease

    def _scegli_asset(self, assets):
        pf = piattaforma()
        def punteggio(a):
            n = a["name"].lower()
            if pf == "windows-x64":
                ok = "win" in n
                pref = n.endswith(".exe") * 2 + n.endswith(".zip")
            elif pf.startswith("macos"):
                ok = "mac" in n or "darwin" in n
                pref = n.endswith(".zip") * 2 + n.endswith(".dmg")
            else:
                ok = "linux" in n and not n.endswith((".deb", ".rpm", ".sha256"))
                pref = n.endswith(".zip") * 2 + (("." not in n.split("linux")[-1]) * 3) + n.endswith((".tar.gz", ".tgz"))
            return (ok, pref, -len(n))
        candidati = sorted(assets, key=punteggio, reverse=True)
        if not candidati or not punteggio(candidati[0])[0]:
            raise FileNotFoundError("no GraXpert download for this system in the latest release: "
                                    + ", ".join(a["name"] for a in assets))
        return candidati[0]

    def ultima(self, sessione=None):
        s = sessione or _sessione()
        r = s.get(URL_GRAXPERT, timeout=TIMEOUT, params={"per_page": 15},
                  headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        uscite = [u for u in r.json() if not u.get("draft") and (self.prerelease or not u.get("prerelease"))]
        if not uscite:
            raise ValueError("no GraXpert release found")
        u = uscite[0]
        a = self._scegli_asset(u.get("assets", []))
        dig = str(a.get("digest") or "")
        return dict(version=u["tag_name"].lstrip("v"), url=a["browser_download_url"], file_name=a["name"],
                    sha256=dig.split(":", 1)[1] if dig.startswith("sha256:") else None, size=a.get("size"),
                    date=str(u.get("published_at", ""))[:10], prerelease=bool(u.get("prerelease")))

    def installa(self, info=None, progresso=None, log=print):
        info = info or self.ultima()
        dest = self.cartella / info["version"]
        with tempfile.TemporaryDirectory(prefix="fitslab_dl_") as tmp:
            arch = Path(tmp) / info["file_name"]
            log(f"download {info['url']}")
            scarica(info["url"], arch, info.get("sha256"), progresso)
            if dest.exists():
                shutil.rmtree(dest)
            estrai(arch, dest)
        candidati = [p for p in dest.rglob("*") if p.is_file() and "graxpert" in p.name.lower()
                     and (p.suffix.lower() in ("", ".exe", ".appimage") or p.name.lower().startswith("graxpert-linux"))]
        if not candidati:
            raise FileNotFoundError(f"GraXpert executable not found in {dest}")
        exe = sorted(candidati, key=lambda p: (len(p.parts), len(p.name)))[0]
        rendi_eseguibile(exe)
        self._salva_stato(dict(version=info["version"], executable=str(exe), installed=str(date.today()),
                               source=info["url"]))
        self._pulisci_vecchie(info["version"])
        return exe


STRUMENTI = {s.id: s for s in (StarNet(), GraXpert())}


# =============================================================================
# CONTROLLO DEGLI AGGIORNAMENTI
# =============================================================================
def file_controllo():
    return cartella_base() / "ultimo_controllo.json"


def controlla_tutti(sessione=None):
    """{id: {'installed', 'latest', 'update', 'error'}} e memorizza la data."""
    s = sessione or _sessione()
    out = {}
    for sid, st in STRUMENTI.items():
        voce = dict(installed=st.versione_installata(), latest=None, update=False, error=None, info=None)
        try:
            info = st.ultima(s)
            voce["latest"] = info["version"]
            voce["info"] = info
            voce["update"] = bool(voce["installed"]) and _confronta_versioni(info["version"], voce["installed"]) > 0
        except Exception as e:
            voce["error"] = str(e)
        out[sid] = voce
    try:
        cartella_base().mkdir(parents=True, exist_ok=True)
        file_controllo().write_text(json.dumps(dict(date=str(date.today()),
                                                    results={k: {x: v[x] for x in ("installed", "latest", "update",
                                                                                   "error")}
                                                             for k, v in out.items()}), indent=1), encoding="utf-8")
    except OSError:
        pass
    return out


def controllo_dovuto():
    try:
        d = json.loads(file_controllo().read_text(encoding="utf-8"))
        return d.get("date") != str(date.today())
    except (OSError, ValueError):
        return True
