#!/usr/bin/env python3
"""Fully local, classical (non-AI) image denoiser for PNG/JPEG, single files or batch directories."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
METHODS = ("nlm", "bilateral", "median", "gaussian")


def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return np.rint(img.astype(np.float32) / 257.0).astype(np.uint8)
    if img.dtype in (np.float32, np.float64):
        return np.rint(np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def kernel_size(strength):
    k = 2 * int(strength // 5) + 1
    k = max(3, min(15, k))
    return k if k % 2 == 1 else k + 1


def denoise_core(img, method, strength):
    if method == "nlm":
        h = float(strength)
        if img.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)
        return cv2.fastNlMeansDenoising(img, None, h, 7, 21)
    if method == "bilateral":
        s = float(strength) * 7.5
        return cv2.bilateralFilter(img, 9, s, s)
    if method == "median":
        return cv2.medianBlur(img, kernel_size(strength))
    if method == "gaussian":
        k = kernel_size(strength)
        return cv2.GaussianBlur(img, (k, k), 0)
    raise ValueError(f"unknown method: {method}")


def denoise_image(img, method, strength):
    img = to_uint8(img)
    if img.ndim == 3 and img.shape[2] == 4:
        out = denoise_core(img[:, :, :3], method, strength)
        return np.dstack((out, img[:, :, 3]))
    return denoise_core(img, method, strength)


def make_preview(before, after):
    def prep(im):
        im = to_uint8(im)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        h = 480
        if im.shape[0] > h:
            im = cv2.resize(im, (max(1, int(im.shape[1] * h / im.shape[0])), h))
        return im

    a, b = prep(before), prep(after)
    h = min(a.shape[0], b.shape[0])
    a, b = a[:h], b[:h]
    gap = np.full((h, 8, 3), 255, np.uint8)
    return cv2.hconcat([a, gap, b])


def process_file(src, dst, method, strength, fmt, quality, preview):
    src, dst = Path(src), Path(dst)
    raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError("unreadable or unsupported image")
    out = denoise_image(raw, method, strength)
    ext = ("." + fmt) if fmt else src.suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        ext = ".png"
    target = dst.with_suffix(ext)
    target.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        if out.ndim == 3 and out.shape[2] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
        elif out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if not cv2.imwrite(str(target), out, params):
        raise ValueError("failed to write output")
    if preview:
        prev = make_preview(raw, out)
        cv2.imwrite(str(target.with_name(target.stem + "_compare.png")), prev, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return target


def _worker(job):
    src, dst, method, strength, fmt, quality, preview = job
    t0 = time.perf_counter()
    try:
        target = process_file(src, dst, method, strength, fmt, quality, preview)
        return str(src), str(target), time.perf_counter() - t0, None
    except Exception as exc:
        return str(src), None, time.perf_counter() - t0, str(exc)


def collect_inputs(inp, out_dir):
    if inp.is_file():
        return [inp]
    out_resolved = out_dir.resolve()
    return sorted(
        f for f in inp.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS and out_resolved not in f.resolve().parents
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="denoise.py",
        description="Denoise PNG/JPEG images fully locally with classical filters (no AI, no network).",
        epilog="examples:\n"
               "  python denoise.py photo.png\n"
               "  python denoise.py photos/ -o clean/ -m nlm -s 12 -j 4\n"
               "  python denoise.py noisy.png -m median -s 10 --format png --preview\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="image file or directory (searched recursively for .png/.jpg/.jpeg)")
    ap.add_argument("-o", "--output", help="output directory (default: <input>_denoised next to the input)")
    ap.add_argument("-m", "--method", choices=METHODS, default="nlm",
                    help="nlm=non-local means (best for photo noise, default), bilateral=edge-preserving smooth, "
                         "median=best for salt-and-pepper, gaussian=simple blur")
    ap.add_argument("-s", "--strength", type=float, default=10.0,
                    help="strength knob (default 10): nlm uses it as h directly, bilateral as sigma/7.5, "
                         "median/gaussian derive the kernel size from it")
    ap.add_argument("--format", choices=["png", "jpg", "jpeg"], default=None,
                    help="force output format (default: keep each input's format)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default 95)")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="parallel workers for batch mode (default 1)")
    ap.add_argument("--preview", action="store_true", help="also write a side-by-side <name>_compare.png")
    args = ap.parse_args(argv)

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"error: no such file or directory: {inp}")
    if inp.is_file():
        if inp.suffix.lower() not in IMAGE_EXTS:
            sys.exit(f"error: unsupported extension {inp.suffix!r} (use .png/.jpg/.jpeg)")
        default_out = inp.parent / (inp.stem + "_denoised")
    else:
        default_out = inp.parent / (inp.name + "_denoised")
    out_dir = Path(args.output) if args.output else default_out

    files = collect_inputs(inp, out_dir)
    if not files:
        sys.exit("error: no PNG/JPEG images found")

    jobs = []
    for f in files:
        rel = f.relative_to(inp) if inp.is_dir() else Path(f.name)
        jobs.append((f, out_dir / rel, args.method, args.strength, args.format, args.quality, args.preview))

    total = len(jobs)
    print(f"denoising {total} image(s) | method={args.method} strength={args.strength} -> {out_dir}")
    if args.jobs > 1 and total > 1:
        import multiprocessing as mp
        with mp.Pool(min(args.jobs, total)) as pool:
            results = pool.map(_worker, jobs)
    else:
        results = [_worker(j) for j in jobs]

    ok = 0
    for i, (src, dst, dt, err) in enumerate(results, 1):
        if err:
            print(f"[{i}/{total}] FAIL {src}: {err}")
        else:
            ok += 1
            print(f"[{i}/{total}] {src} -> {dst} ({dt:.2f}s)")
    failed = total - ok
    print(f"done: {ok}/{total} succeeded" + (f", {failed} failed" if failed else ""))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
