#!/usr/bin/env bash
# =============================================================================
# Crea l'eseguibile "FITSLab" per Ubuntu, lo prova (autotest) e lo installa
# con l'icona nel menu e sulla Scrivania.
#
# Uso:  cd ~/Claude/Astropy/programma_fitslab/versione_1.7
#       bash crea_eseguibile_ubuntu.sh
#
# © 2026 Edoardo Battistel - FITS Lab
# =============================================================================
set -e
cd "$(dirname "$(readlink -f "$0")")"

for f in fitslab.py fitslab_core.py fitslab_operazioni.py fitslab_healpix.py fitslab_testi.py fitslab_strumenti.py fitslab_esterni.py \
         motore_aap.py roman_asdf.py rifinitura.py openngc_compatto.csv fitslab.png; do
    [ -f "$f" ] || { echo "ERRORE: manca $f in questa cartella ($(pwd))."; exit 1; }
done

echo "=== 1/4  Strumenti di sistema (chiede la password) ==="
sudo apt update
sudo apt install -y python3-venv python3-tk zenity

echo "=== 2/4  Ambiente Python per la compilazione ==="
[ -d .venv-build ] || python3 -m venv .venv-build
source .venv-build/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib astropy astropy-healpix scikit-image PyWavelets pyvo requests \
    pillow tifffile psutil asdf asdf-astropy gwcs lz4 pyinstaller

echo "=== Prova del programma prima della compilazione ==="
python fitslab.py --autotest

echo "=== 3/4  Creazione dell'eseguibile (qualche minuto) ==="
pyinstaller --noconfirm --clean --onefile --windowed \
    --name FITSLab \
    --collect-all astropy \
    --collect-all astropy_healpix \
    --collect-all skimage \
    --collect-all pywt \
    --collect-all pyvo \
    --collect-all erfa \
    --collect-all asdf \
    --collect-all asdf_astropy \
    --collect-all asdf_standard \
    --collect-all asdf_transform_schemas \
    --collect-all asdf_coordinates_schemas \
    --collect-all asdf_wcs_schemas \
    --collect-all gwcs \
    --collect-all lz4 \
    --collect-submodules scipy \
    --collect-binaries scipy \
    --collect-submodules matplotlib \
    --collect-data matplotlib \
    --collect-data astropy_iers_data \
    --collect-submodules tifffile \
    --copy-metadata scipy --copy-metadata numpy --copy-metadata astropy --copy-metadata astropy-healpix \
    --copy-metadata scikit-image --copy-metadata PyWavelets --copy-metadata pyvo --copy-metadata matplotlib \
    --copy-metadata pyerfa --copy-metadata astropy-iers-data --copy-metadata packaging --copy-metadata pyyaml \
    --copy-metadata psutil --copy-metadata pillow --copy-metadata tifffile --copy-metadata lazy_loader \
    --copy-metadata imageio --copy-metadata networkx --copy-metadata requests --copy-metadata asdf \
    --copy-metadata asdf-astropy --copy-metadata asdf-standard --copy-metadata asdf-transform-schemas \
    --copy-metadata asdf-coordinates-schemas --copy-metadata asdf-wcs-schemas --copy-metadata gwcs --copy-metadata lz4 \
    --hidden-import fitslab_core --hidden-import fitslab_operazioni --hidden-import fitslab_healpix \
    --hidden-import fitslab_testi --hidden-import fitslab_strumenti --hidden-import fitslab_esterni --hidden-import motore_aap --hidden-import roman_asdf --hidden-import rifinitura \
    --hidden-import PIL._tkinter_finder --hidden-import PIL.ImageTk \
    --add-data "openngc_compatto.csv:." \
    --add-data "fitslab.png:." \
    fitslab.py
deactivate

echo "=== Verifica dell'eseguibile (autotest, apre anche la finestra) ==="
if ! ./dist/FITSLab --autotest > autotest.txt 2>&1; then
    cat autotest.txt
    echo
    echo "L'autotest è FALLITO: i dettagli sono in $(pwd)/autotest.txt"
    echo "L'eseguibile non è stato installato. Invia autotest.txt per la correzione."
    rm -rf build FITSLab.spec
    exit 1
fi
cat autotest.txt

echo "=== Prova di SIMBAD e NED (diagnostica, serve Internet) ==="
./dist/FITSLab --diagnostica > diagnostica.txt 2>&1 || true
grep -E "^\[(ok|ERRORE)\]" diagnostica.txt || true

echo "=== 4/4  Installazione nella home e icona nel menu ==="
DEST="$HOME/.local/bin"
mkdir -p "$DEST" "$HOME/.local/share/applications" "$HOME/.local/share/icons/hicolor/512x512/apps"
cp fitslab.png "$HOME/.local/share/icons/hicolor/512x512/apps/fitslab.png"
gtk-update-icon-cache -q "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
cp dist/FITSLab "$DEST/FITSLab"
chmod +x "$DEST/FITSLab"

# Tk dichiara la classe della finestra come «Fitslab» (solo la prima lettera maiuscola):
# StartupWMClass deve essere identico, e il file si chiama come la classe in minuscolo.
cat > "$HOME/.local/share/applications/fitslab.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=FITS Lab
Comment=Laboratorio interattivo per i FITS di tutti i telescopi (scienza ed estetica)
Exec=$DEST/FITSLab %f
Icon=$HOME/.local/share/icons/hicolor/512x512/apps/fitslab.png
StartupWMClass=Fitslab
Terminal=false
Categories=Science;Astronomy;Education;Graphics;
MimeType=image/fits;application/fits;
EOF
chmod +x "$HOME/.local/share/applications/fitslab.desktop"
update-desktop-database -q "$HOME/.local/share/applications" 2>/dev/null || true

SCRIVANIA="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Scrivania")"
if [ -d "$SCRIVANIA" ]; then
    cp "$HOME/.local/share/applications/fitslab.desktop" "$SCRIVANIA/"
    chmod +x "$SCRIVANIA/fitslab.desktop"
    gio set "$SCRIVANIA/fitslab.desktop" metadata::trusted true 2>/dev/null || true
fi

rm -rf build FITSLab.spec

echo
echo "FATTO."
echo "  Eseguibile:  $DEST/FITSLab   (copia anche in $(pwd)/dist/)"
echo "  Avvio:       cerca 'FITS Lab' nel menu, oppure doppio clic sull'icona sulla Scrivania"
echo "               (la prima volta: tasto destro -> 'Consenti avvio')."
echo "  Controllo dell'icona nella barra:  xwininfo -root -tree | grep -i fitslab"
