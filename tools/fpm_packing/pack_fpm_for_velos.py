"""
Pack the FPM raw data in "Real Samples" into small files for the Velos platform (M4).

For every acquisition folder that has positions.csv, this script keeps ALL frames but only
the central N x N pixels (default 512, the same patch as the MATLAB quick check), and writes
compressed .npz files of about 15-25 MB to the folder "_velos_packed" next to this script.
The original files are only read, never changed.

    double-click PACK_REAL_SAMPLES_FOR_VELOS.bat      (or)    python pack_fpm_for_velos.py 512
"""
import csv
import glob
import os
import sys
import time

import numpy as np

CHUNK = 45  # frames per part file


def read_tif(path):
    try:
        import tifffile
        return np.asarray(tifffile.imread(path))
    except Exception:
        from PIL import Image
        with Image.open(path) as im:
            return np.array(im)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 512
    include_colour = "--all" in sys.argv
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(here, "Real Samples")
    out = os.path.join(here, "_velos_packed")
    os.makedirs(out, exist_ok=True)
    sets = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "positions.csv"), recursive=True)})
    print(f"{len(sets)} acquisition folder(s) under {root}; crop {N} x {N}; output {out}\n")
    t_all = time.time()
    for d in sets:
        name = os.path.relpath(d, root).replace(os.sep, "__").replace(" ", "_")
        meta_path = os.path.join(out, f"{name}_c{N}_meta.npz")
        if os.path.exists(meta_path):
            print("already packed:", name)
            continue
        with open(os.path.join(d, "positions.csv"), encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        files = [r["file"] for r in rows]
        missing = [f for f in files if not os.path.exists(os.path.join(d, f))]
        if missing:
            print(f"SKIP {name}: {len(missing)} image(s) missing")
            continue
        first = read_tif(os.path.join(d, files[0]))
        colour = first.ndim == 3
        if colour and not include_colour:
            print(f"SKIP {name}: colour camera frames (run with --all to include)")
            continue
        H, W = first.shape[:2]
        n = min(N, H, W) // 2 * 2
        r0, c0 = (H - n) // 2, (W - n) // 2
        ch = None
        if colour:
            ch = int(np.argmax([first[..., k].mean() for k in range(first.shape[2])]))
        run_info = ""
        if os.path.exists(os.path.join(d, "run_info.json")):
            with open(os.path.join(d, "run_info.json"), encoding="utf-8") as f:
                run_info = f.read()
        with open(os.path.join(d, "positions.csv"), encoding="utf-8-sig") as f:
            positions = f.read()
        t0 = time.time()
        stack = np.empty((len(files), n, n), dtype=first.dtype if not colour else first.dtype)
        centre_full, centre_file = None, None
        try:
            import json
            ci = int(json.loads(run_info).get("center_led_index", 0)) if run_info else 0
        except Exception:
            ci = 0
        for m, fn in enumerate(files):
            a = read_tif(os.path.join(d, fn))
            if colour:
                a = a[..., ch]
            stack[m] = a[r0:r0 + n, c0:c0 + n]
            if ci and m == ci - 1:
                centre_full, centre_file = a, fn
            if (m + 1) % 25 == 0 or m + 1 == len(files):
                print(f"  {name}: {m + 1}/{len(files)} frames read")
        parts = (len(files) + CHUNK - 1) // CHUNK
        for k in range(parts):
            np.savez_compressed(os.path.join(out, f"{name}_c{N}_part{k + 1}of{parts}.npz"),
                                frames=stack[k * CHUNK:(k + 1) * CHUNK], first_index=k * CHUNK)
        np.savez_compressed(meta_path, files=np.array(files), positions_csv=positions, run_info_json=run_info,
                            crop_row0=r0, crop_col0=c0, frame_height=H, frame_width=W, crop=n,
                            colour_channel=-1 if ch is None else ch, n_frames=len(files), parts=parts,
                            centre_full=centre_full if centre_full is not None else np.zeros((0, 0), np.uint16),
                            centre_file=centre_file or "")
        size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(out, f"{name}_c{N}_*.npz")))
        print(f"  -> {parts + 1} files, {size / 1e6:.1f} MB, {time.time() - t0:.0f} s\n")
    print(f"Done in {time.time() - t_all:.0f} s. Please tell Claude that packing has finished.")


if __name__ == "__main__":
    main()
