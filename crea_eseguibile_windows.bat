@echo off
REM ===========================================================================
REM Crea FITSLab.exe per Windows 11 (da fare UNA sola volta).
REM Serve Python per la sola compilazione (python.org, spunta "Add python.exe to PATH").
REM Uso: doppio clic su questo file, nella cartella con fitslab.py.
REM (C) 2026 Edoardo Battistel - FITS Lab
REM ===========================================================================
chcp 65001 >nul
cd /d "%~dp0"

for %%f in (fitslab.py fitslab_core.py fitslab_operazioni.py fitslab_healpix.py fitslab_testi.py fitslab_strumenti.py fitslab_esterni.py motore_aap.py roman_asdf.py rifinitura.py openngc_compatto.csv fitslab.png fitslab.ico) do (
    if not exist %%f (
        echo ERRORE: manca %%f in questa cartella: %cd%
        pause
        exit /b 1
    )
)

where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)
%PY% --version >nul 2>nul
if errorlevel 1 (
    echo ERRORE: Python non trovato. Installalo da https://www.python.org
    pause
    exit /b 1
)

echo === 1/3  Ambiente Python per la compilazione ===
if not exist .venv-build (%PY% -m venv .venv-build || goto errore)
call .venv-build\Scripts\activate.bat
python -m pip install --upgrade pip || goto errore
python -m pip install numpy scipy matplotlib astropy astropy-healpix scikit-image PyWavelets pyvo requests pillow tifffile psutil asdf asdf-astropy gwcs lz4 pyinstaller || goto errore

echo === 2/3  Creazione dell'eseguibile ===
pyinstaller --noconfirm --clean --onefile --windowed --name FITSLab --icon fitslab.ico ^
    --collect-all astropy --collect-all astropy_healpix --collect-all skimage --collect-all pywt --collect-all pyvo ^
    --collect-all erfa --collect-all asdf --collect-all asdf_astropy --collect-all asdf_standard ^
    --collect-all asdf_transform_schemas --collect-all asdf_coordinates_schemas --collect-all asdf_wcs_schemas ^
    --collect-all gwcs --collect-all lz4 --collect-submodules scipy --collect-binaries scipy ^
    --collect-submodules matplotlib --collect-data matplotlib --collect-data astropy_iers_data --collect-submodules tifffile ^
    --copy-metadata scipy --copy-metadata numpy --copy-metadata astropy --copy-metadata astropy-healpix ^
    --copy-metadata scikit-image --copy-metadata PyWavelets --copy-metadata pyvo --copy-metadata matplotlib ^
    --copy-metadata pyerfa --copy-metadata astropy-iers-data --copy-metadata packaging --copy-metadata pyyaml ^
    --copy-metadata psutil --copy-metadata pillow --copy-metadata tifffile --copy-metadata lazy_loader ^
    --copy-metadata imageio --copy-metadata networkx --copy-metadata requests --copy-metadata asdf ^
    --copy-metadata asdf-astropy --copy-metadata asdf-standard --copy-metadata asdf-transform-schemas ^
    --copy-metadata asdf-coordinates-schemas --copy-metadata asdf-wcs-schemas --copy-metadata gwcs --copy-metadata lz4 ^
    --hidden-import fitslab_core --hidden-import fitslab_operazioni --hidden-import fitslab_healpix ^
    --hidden-import fitslab_testi --hidden-import fitslab_strumenti --hidden-import fitslab_esterni --hidden-import motore_aap --hidden-import roman_asdf --hidden-import rifinitura ^
    --hidden-import PIL._tkinter_finder --hidden-import PIL.ImageTk ^
    --add-data "openngc_compatto.csv;." --add-data "fitslab.png;." ^
    fitslab.py || goto errore
call deactivate

echo === Verifica dell'eseguibile (autotest) ===
del /q autotest_risultato.txt 2>nul
dist\FITSLab.exe --autotest > autotest.txt 2>&1
if exist autotest_risultato.txt type autotest_risultato.txt >> autotest.txt
type autotest.txt
findstr /c:"TUTTO OK" autotest.txt >nul || (
    echo.
    echo L'autotest e' FALLITO: i dettagli sono in %cd%\autotest.txt
    goto errore
)

echo === 3/3  Collegamento sul Desktop ===
copy /y dist\FITSLab.exe "%cd%\FITSLab.exe" >nul
powershell -NoProfile -Command ^
  "$d=[Environment]::GetFolderPath('Desktop'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut(\"$d\FITS Lab.lnk\"); $s.TargetPath='%cd%\FITSLab.exe'; $s.WorkingDirectory='%cd%'; $s.IconLocation='%cd%\fitslab.ico'; $s.Save()"

rmdir /s /q .venv-build build 2>nul
del /q FITSLab.spec 2>nul
echo.
echo FATTO. Sul Desktop trovi il collegamento "FITS Lab".
pause
exit /b 0

:errore
echo.
echo Si e' verificato un errore: copia i messaggi qui sopra e inviameli.
pause
exit /b 1
