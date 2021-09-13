PyHoLo is a software intended for translating electron holograms acquired in TEM into phase shift of electron wave. It consequently allows for determination of electric and magnetic fields in a sample from a set of "upside" and "downside" holograms.

Copyright (C) 2020  Krzysztof Morawiec

---------------------------------------------------------------------------------------------

I. Downloading PyHoLo

Current version of PyHoLo can be downloaded from the following github repository:

https://github.com/morawatur/PyHoLo

This repository contains source files needed for running PyHoLo. To run PyHoLo from source you will need to install Python interpreter. For instructions on how to do that see sections III and IV(a).

Alternatively, I can send you precompiled package with ready-to-use executable, which would not require you to install any additional tools or libraries. If you are interested, please e-mail me at morawk@ifpan.edu.pl
For instruction on how to run PyHoLo from precompiled package see section IV(b).

---------------------------------------------------------------------------------------------

II. List of source files:

Constants.py -- constants needed in some of the calculations
Dm3Reader.py -- methods for reading DM3 files storing image data from TEM
GUI.py -- GUI design and callback methods for handling all events called by user
Holo.py -- procedures dedicated to the problem of restoring phase shift from electron hologram
ImageSupport.py -- definitions of Image and ImageList objects, procedures for most of the image processing
PyHoLo.py -- main PyHoLo script with a call to starting GUI
Transform.py -- methods for image alignment: shift, rotation, magnification, warping

Supplementary scripts:

Phase3d.py -- script for plotting Phase(x,y) in three-dimensional space
RenameSerDm3.py -- script for renaming series of dm3 files

---------------------------------------------------------------------------------------------

III. Installation prerequisities

In order to run PyHoLo you will require:

- Windows Vista, 7, 8 or 10 (≥ Vista),
- Python language interpreter version 3.6 or later (e.g. Anaconda3),
- (optionally) external Python libraries, such as PyQt5.

1. Download the Anaconda environment from https://www.anaconda.com/distribution/. Choose a version compatible with Python 3.6 or later.

2. Install the Anaconda environment by running the installation file and following the instructions on the screen. We recommend you to install Anaconda in the default location suggested by the instalator. In Windows it will be 'C:\Users\user_name\Anaconda3'.

3. After installation open Anaconda Prompt. In Windows 7 it can be found under Start menu, in All Programs -> Anaconda3 -> Anaconda Prompt.

4 (May not be necessary). New version of Anaconda (3.7) comes with a number of libraries which are required for PyHoLo to work. However, older Anaconda versions may be missing newer distributions of those libraries, e.g. PyQt5, scikit-image. To install those packages (i.e. if you stumble upon some error during running PyHoLo), type the following command in the Anaconda Prompt:

conda install <package_name>
(e.g. conda install pyqt)

You will be asked for confirmation of the package name which you want to install. You can confirm it by typing 'y' into the Anaconda Prompt and pressing Enter. If the installation of library is complete, you can close the Anaconda Prompt.

5. PyHoLo should be ready to use!

---------------------------------------------------------------------------------------------

IV.(a) Running PyHoLo from Python interpreter (source)

1. Extract files from downloaded zip archive to a directory on local disk (e.g. C:\python_programs\PyHoLo).

2. Open the Anaconda Prompt. In Windows 7 it can be found under Start menu, in All Programs -> Anaconda3 -> Anaconda Prompt.

3. From the Anaconda Prompt level go to the directory with extracted files. You can do it by entering the following command:

cd path_to_directory
(e.g. cd C:\python_programs\PyHoLo)

4. Run the PyHoLo.py script by entering the following command:

python PyHoLo.py

Main window should appear. Click on "File -> Open dm3 or npy" or "File -> Open dm3 series" to open single dm3/npy image or series of dm3 images, respectively. In the latter case, when the Open dialog appears, mark the first image in the series and press Open.

Note 1: Input images must be stored in DM3 format (internal file format of Gatan Microscopy Suite software) or NPY format (file format used in numpy package for storing arrays).

Note 2: Image names in the dm3 image series must have a common set of characters which denote series name and they must be numbered according to the following convention:

ser_name_01.dm3, ser_name_02.dm3, ser_name_03.dm3 etc.
where in this case 'ser_name_' is a name of hologram series (digits smaller than 10 must be preceeded with '0', eg. 'ser_name_03.dm3').

---------------------------------------------------------------------------------------------

IV.(b) Running PyHoLo from EXE

1. Extract files from downloaded zip archive to a directory on local disk (e.g. C:\programs\PyHoLo).

2. Find PyHoLo.exe in the following directory: extraction_path\PyHoLo\dist\PyHoLo

3. Run PyHoLo.exe (you can create a shortcut to it on your desktop).

---------------------------------------------------------------------------------------------

See the manual for more detailed informations about using PyHoLo.