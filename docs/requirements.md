# Requirements

## Python

Python **3.11** or newer.

Linux, macOS and Windows are all supported, and each release is tested on all
three. The base package uses NumPy, Polars, and Matplotlib.

## edf2asc (only for EyeLink data)

EyeLink EDF files are parsed by first converting them to ASCII with **`edf2asc`**, a tool from SR Research distributed inside the **EyeLink Developers Kit**.

1. Create a free account and download the kit from the [SR Research support forum](https://www.sr-research.com/support/).
2. Install the kit (the installer places `edf2asc` in a system location on macOS/Windows and provides a `.deb` for Debian/Ubuntu Linux).
3. Make sure `edf2asc` is on your `PATH`. Verify with:

```bash
edf2asc -h
```

You only need `edf2asc` for EyeLink EDF input. EyeLink ASC files are read
directly, and the Tobii, GazePoint, and WebGazer pipelines do not require it.

## Base Python dependencies

The default installation declares only direct dependencies shared across the package:

- `numpy>=1.24`
- `polars>=1.26.0`
- `matplotlib>=3.9.2`

Raw conversion, BIDS TSV/JSON I/O, preprocessing, and analysis all use Polars
tables.

PsychoPy text logs are parsed directly, without PsychoPy installed.

## Optional feature dependencies

Install feature-specific packages only when needed:

```bash
pip install "pyxations[remodnav]"   # REMoDNaV detection
pip install "pyxations[multimatch]" # MultiMatch scanpath comparison
pip install "pyxations[video]"      # OpenCV-backed gaze/video animation
pip install "pyxations[all]"        # all optional features
```

The minimum optional versions are:

- `remodnav>=1.1.2`
- `multimatch-gaze>=0.1.3`
- `opencv-python-headless>=4.9.0.80`

Optional modules are imported lazily. Importing the base package does not load these libraries; requesting an unavailable feature raises an error containing the relevant installation command.

Saving an animation as MP4 also requires an `ffmpeg` executable available on
`PATH`.

## Version policy

Runtime dependencies declare a minimum supported version, so Pyxations can
coexist with newer releases of the scientific stack. The committed `uv.lock`
pins a cross-platform development environment; it does not constrain installs
of the library.

Development and documentation tools are installed through the `dev` and `docs` optional groups.
