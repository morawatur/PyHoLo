PyHoLo is a software intended for translating electron holograms acquired in TEM into phase shift of electron wave. It consequently allows for determination of electric and magnetic fields in a sample from a set of "upside" and "downside" holograms.

Copyright (C) 2020  Krzysztof Morawiec

---------------------------------------------------------------------------------------------

I. Downloading PyHoLo

Current version of PyHoLo can be downloaded from the following github repository:

https://github.com/morawatur/PyHoLo

This repository contains source files needed for running PyHoLo.
To run PyHoLo from source you will need to install Python interpreter and PyQt5 library.
For instructions on how to do that see sections III and IV(a).

Alternatively, I can send you precompiled package with ready-to-use executable, which would not require you
to install any additional tools or libraries. If you are interested, please e-mail me at morawk@ifpan.edu.pl
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

Supplementary scripts (not fully implemented):

GradientArrows.py -- methods intended for presenting color phase maps with imposed grid of arrows, which show local direction of magnetic field
Phase3d.py -- script for plotting Phase(x,y) in three-dimensional space

---------------------------------------------------------------------------------------------

III. Installation prerequisities

In order to run PyHoLo you will require:

- Windows Vista, 7, 8 or 10 (≥ Vista),
- Python language interpreter version 3.6 or later (e.g. Anaconda),
- PyQt5 library supporting the graphical user interface.

1. Download the Anaconda environment from  https://www.continuum.io/downloads. Choose a version compatible with Python 3.6 or later.

2. Install the Anaconda environment by running the installation file and following the instructions on the screen. We recommend you to install Anaconda in the default location suggested by the instalator. In Windows it will be 'C:\Users\user_name\Anaconda3'.

3. After installation open Anaconda Prompt. In Windows 7 it can be found under Start menu, in All Programs -> Anaconda -> Anaconda Prompt.

4. Type the following command in the Anaconda Prompt in order to install PyQt5 library:

conda install pyqt

You will be asked for confirmation of the library name which you want to install. You can confirm it by typing 'y' into the Anaconda Prompt and pressing Enter. If the installation of library is complete, you can close the Anaconda Prompt.

PyHoLo should be ready to use!

---------------------------------------------------------------------------------------------

IV.(a) Running PyHoLo from Python interpreter (source)

1. Extract files from downloaded zip archive to a directory on local disk (e.g. C:\python_programs\PyHoLo).

2. Open the Python command interpreter (e.g. IPython). If you have Anaconda, then in Windows 7 it can be found under Start menu, in All Programs -> Anaconda -> IPython.

3. From the command interpreter level go to the directory with extracted files. You can do it by typing in the interpreter the following command:

cd path_to_directory
(e.g. cd C:\python_programs\PyHoLo)

4. Run the PyHoLo.py script by entering the following command:

%run PyHoLo.py

A dialog window should appear, which allows you to point to a directory with series of experimental images stored in DM3 file format.
Mark the first image from series and press Open.

Note 1: Input images must be stored in DM3 format (internal file format of Gatan Microscopy Suite software).

Note 2: Image names must have a common set of characters which denote series name and they must be numbered according to the following convention:

ser_name_01.dm3, holo02.dm3, holo03.dm3 etc.
where in this case 'ser_name_' is a name of series (digits smaller than 10 must be preceeded with '0', eg. 'holo03.dm3').

---------------------------------------------------------------------------------------------

IV.(b) Running PyHoLo from EXE

1. Extract files from downloaded zip archive to a directory on local disk (e.g. C:\programs\PyHoLo).

2. Find PyHoLo.exe in the following directory: extraction_path\PyHoLo\dist\PyHoLo

3. Run PyHoLo.exe or create a shortcut to it on desktop.

---------------------------------------------------------------------------------------------

V. Unit test

1. Run PyHoLo.

2. Load set of holograms from the "input" directory, provided with the source files.

3. Navigate to 3rd image ("ref_down") by clicking Next (Navigation tab), and flip the image by clicking Flip button (Navigation tab).

4. Do the same for 4th image ("obj_down").