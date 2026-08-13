MINIMAX H3 CHECKPOINT CONVERTER
================================

Turns a large MiniMax H3 model file (about 66 GB) into a small one (about
12.5 GB) that ComfyUI can load.


HOW TO USE IT
-------------

  1. Double-click              start_windows.bat

     The first run takes a while. It downloads everything it needs into this
     folder, including its own copy of Python. You do not need to install
     anything yourself.

  2. Click                     Browse...

  3. Pick your model file      (the big .safetensors file)

  4. Click                     A   or   B

  5. Wait.

  6. Done. The new file is saved next to your original.


WHICH BUTTON?
-------------

  A - Pruned W4A8 ConvRot
      Matches the known 12 GB MiniMax H3 W4A8 checkpoints. Needs a ComfyUI
      build with W4A8 support.

  B - Pruned NVFP4
      Uses ComfyUI's standard NVFP4 support. Needs a Blackwell GPU
      (RTX 50-series).

If you are not sure, pick A.


WHAT IT WILL NOT DO
-------------------

  - It never changes or deletes your original file.
  - It never leaves a broken file behind pretending to be finished. The new
    file only gets its real name after it has been checked.
  - It will not start if there is not enough free disk space, and it will tell
    you how much you need.


WHAT YOU NEED
-------------

  Windows 10 or 11, 64-bit
  NVIDIA RTX 5090 (or another tested Blackwell card)
  32 GB video memory
  96 GB system memory
  Enough free disk space for the new file - roughly 25 GB free for A,
  35 GB for B
  A current NVIDIA driver

The converter will never install or change your graphics driver. If your
driver is too old it stops and tells you.


IF SOMETHING GOES WRONG
-----------------------

Every run writes a log file into the  logs  folder, named with the date and
time. If the converter reports a problem, that file has the details.

If the program closes by itself without an error message, that log file will
still contain the reason - the crash details are written to it directly. Send
the whole file when reporting the problem.

Try  update_windows.bat  first - it repairs the installed components.


UPDATING
--------

Double-click                   update_windows.bat


MORE DETAIL
-----------

README.md has the technical description, and docs\COMPATIBILITY.md lists the
exact tested versions.
