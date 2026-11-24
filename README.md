<p align="center"><img src="fitslab.png" width="128" alt="FITS Lab icon"></p>

# FITS Lab

*A non-destructive FITS editing lab: science-grade and aesthetic outputs from the same recipe*

FITS Lab shows the original on the left and the result on the right, and builds a **recipe** of non-destructive steps (stretch, background, noise, colour, crop, binning…). It writes two kinds of output: `science/` (linear FITS that keep a valid WCS even after cropping and binning) and `aesthetic/` (images for showing). It reads all missions including Euclid, Roman (ASDF), radio data, cubes and Planck HEALPix maps, and shows SIMBAD/NED objects on a full-resolution zoom. It can download and update free external tools on its own (StarNet2 with SHA-256 check, GraXpert, Seti Astro Suite / Cosmic Clarity).

## Features

* Italian/English interface, outputs in English
* Recipe of reversible steps, saved and re-applied
* Optional external tools installed in your home folder without admin rights
* Shares its analysis engine with AutoAstroPipe (`motore_aap.py`, `rifinitura.py`, `roman_asdf.py`)

> **Language note.** The interface is in **Italian** (large, adjustable text); plots, PDF/PNG cards and scientific output are in **English**. Code comments are in Italian. The original Italian guide is [`LEGGIMI.txt`](LEGGIMI.txt).

**Version:** 1.7 (2026-11-15). **Status:** 1.0 built and installed (self-test 37/37); later versions add external tools and features.

## Quick start

Requirements: Python 3.10+ (tested with Python 3.14 on Ubuntu) with `tkinter` (Ubuntu: `sudo apt install python3-tk`).

```bash
git clone https://github.com/EDOKILLER/fits-lab.git
cd fits-lab
python3 -m pip install -r requirements.txt

python3 fitslab.py --autotest
python3 fitslab.py --diagnostica
python3 fitslab.py        # opens the window
```

### Standalone executable

`bash crea_eseguibile_ubuntu.sh` builds a single-file executable with PyInstaller in a temporary virtual environment, runs the self-test, and adds the program with its icon to the application menu and the desktop (first time on GNOME: right-click → *Allow launching*). On Windows 11 install Python from python.org (tick *Add python.exe to PATH*) and double-click `crea_eseguibile_windows.bat`.

The build scripts were written for the author's machine: check the paths at the top before running them.

## Data sources and acknowledgements

Data are downloaded at run time from public services. If you publish results, please follow their acknowledgement policies:

* [Mikulski Archive for Space Telescopes (STScI): Hubble, JWST, Kepler, TESS, GALEX, Pan-STARRS](https://archive.stsci.edu/)
* [NASA/IPAC Infrared Science Archive: 2MASS, WISE, Spitzer, Euclid, Planck](https://irsa.ipac.caltech.edu/)
* [SIMBAD database, CDS, Strasbourg](https://simbad.cds.unistra.fr/)
* [NASA/IPAC Extragalactic Database](https://ned.ipac.caltech.edu/)

The file `openngc_compatto.csv` is a compact copy of [OpenNGC](https://github.com/mattiaverga/OpenNGC) by Mattia Verga, licensed under CC BY-SA 4.0 (see `OpenNGC_licenza_CC-BY-SA-4.0.txt`); it keeps that licence, not the BSD licence of the code.

This project is independent and not endorsed by any of these institutions.

## License

Code: [BSD 3-Clause](LICENSE) © 2026 Edoardo Battistel. You can use, modify and redistribute it, also commercially, as long as you keep the copyright notice; the author's name may not be used to promote derived works without permission. If you use it in research, see [`CITATION.cff`](CITATION.cff).

---
Part of a series of astronomy programs by Edoardo Battistel — see also [X-rays from Black Holes](https://github.com/EDOKILLER/x-rays-from-black-holes) and the other repositories on [this profile](https://github.com/EDOKILLER?tab=repositories).
