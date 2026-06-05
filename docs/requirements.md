# Requirements

## Python

Python **3.10** or newer.

Linux, macOS and Windows are all supported. The package is pure Python and depends on the standard scientific stack.

## edf2asc (only for EyeLink data)

EyeLink EDF files are parsed by first converting them to ASCII with **`edf2asc`**, a tool from SR Research distributed inside the **EyeLink Developers Kit**.

1. Create a free account and download the kit from the [SR Research support forum](https://www.sr-research.com/support/).
2. Install the kit (the installer places `edf2asc` in a system location on macOS/Windows and provides a `.deb` for Debian/Ubuntu Linux).
3. Make sure `edf2asc` is on your `PATH`. Verify with:

```bash
edf2asc -h
```

If you see a usage message, you're set. If not, add the install directory to `PATH` (or symlink the binary into `/usr/local/bin`).

You only need `edf2asc` when `dataset_format="eyelink"`. Tobii, Gazepoint and WebGazer pipelines don't require it.

## Python dependencies

Pyxations relies on the standard scientific Python stack:

- `numpy`, `pandas`, `pyarrow`, `polars`
- `scipy`, `statsmodels`
- `matplotlib`, `seaborn`, `opencv-python`
- `remodnav`, `multimatch-gaze` (eye-movement algorithms)
- `tqdm`

The pinned full list lives in `pyproject.toml`. `pip install pyxations` pulls them all in.
