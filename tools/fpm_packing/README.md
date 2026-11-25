# Packing LED-array FPM acquisitions for the Velos platform

A raw 225-LED scan of the CS2100M camera is ~0.9 GB of 16-bit TIFFs. These scripts keep **all frames** but only a
square region of the sensor, in compressed parts of 15–25 MB, so scans can be copied, uploaded and used for training.
Originals are only read. Put the scripts next to the folder `Real Samples` (acquisition folders with `positions.csv`,
`run_info.json` and `led_###_x##_y##.tif`) and run one of:

| Script | What it packs | Output (`_velos_packed/`) |
|---|---|---|
| `PACK_REAL_SAMPLES_FOR_VELOS.bat` (Windows, finds Python/Anaconda) → `pack_fpm_for_velos.py 512` | the central 512 × 512 pixels of every acquisition | `<name>_c512_meta.npz` + `_partKof5.npz` |
| `pack_fpm_for_velos.m` (MATLAB alternative) | the same | `<name>_c512_meta.mat` + `_partKof5.mat` |
| `pack_fpm_side_regions.m` (MATLAB) | two 512 × 512 regions left and right of the centre of one blood and one leaf scan (test regions never used for training) | `<name>_L512_*.mat`, `<name>_R512_*.mat` |

The packed files are read directly by the platform: `python -m velos_recon.fpm run <..._meta.mat>`,
`python -m velos_recon.fpm convert <..._meta.mat> scan.npz` (single-file API format) and the M4 pipeline
`PACK=_velos_packed ml/scripts/m4_fpm_pipeline.sh` (see `docs/fpm.md`, `docs/handover.md`).
