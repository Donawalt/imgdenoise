#!/usr/bin/env python3
"""Fully local, classical (non-AI) image denoiser for photos and 3D renders, single files or batch directories."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".exr"}
METHODS = ("nlm", "bilateral", "median", "gaussian", "wavelet")
PRESETS = {
    "photo": dict(method="nlm", strength=12.0, fireflies=0.0, edge_keep=0.6, sharpen=0.25),
    "render": dict(method="wavelet", strength=12.0, fireflies=50.0, edge_keep=0.8, sharpen=0.35),
}


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


def remove_fireflies(img, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gf = gray.astype(np.float32)
    mean = cv2.boxFilter(gf, -1, (5, 5), normalize=True)
    sq = cv2.boxFilter(gf * gf, -1, (5, 5), normalize=True)
    local_std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    med = cv2.medianBlur(img, 5)
    diff = np.abs(img.astype(np.int16) - med.astype(np.int16))
    d = diff.max(axis=2) if img.ndim == 3 else diff
    mask = (d > thresh) & (d > 6.0 * local_std)
    out = img.copy()
    out[mask] = med[mask]
    return out


def _wavelet_channel(ch, strength):
    import pywt
    f = ch.astype(np.float32) / 255.0
    max_level = pywt.dwtn_max_level(f.shape, "db4")
    if max_level < 1:
        return ch.copy()
    level = min(4, max_level)
    coeffs = pywt.wavedec2(f, "db4", level=level)
    sigma = float(np.median(np.abs(coeffs[1][0]))) / 0.6745
    t = (strength / 10.0) * sigma * np.sqrt(2.0 * np.log(max(f.size, 2)))
    details = [tuple(pywt.threshold(c, t, mode="soft") for c in d) for d in coeffs[1:]]
    rec = pywt.waverec2([coeffs[0]] + details, "db4")[: f.shape[0], : f.shape[1]]
    return np.rint(np.clip(rec, 0.0, 1.0) * 255.0).astype(np.uint8)


def wavelet_denoise(img, strength):
    if img.ndim == 3:
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = _wavelet_channel(ycc[:, :, 0], strength)
        cb = _wavelet_channel(ycc[:, :, 1], strength * 0.5)
        cr = _wavelet_channel(ycc[:, :, 2], strength * 0.5)
        return cv2.cvtColor(cv2.merge((y, cb, cr)), cv2.COLOR_YCrCb2BGR)
    return _wavelet_channel(img, strength)


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
    if method == "wavelet":
        return wavelet_denoise(img, float(strength))
    raise ValueError(f"unknown method: {method}")


def denoise_image(img, method, strength, fireflies=0.0):
    img = to_uint8(img)
    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        if fireflies > 0:
            bgr = remove_fireflies(bgr, fireflies)
        out = denoise_core(bgr, method, strength)
        return np.dstack((out, img[:, :, 3]))
    if fireflies > 0:
        img = remove_fireflies(img, fireflies)
    return denoise_core(img, method, strength)


def _sharpen_gray(g, gain, w=None):
    gf = g.astype(np.float32)
    detail = gf - cv2.GaussianBlur(gf, (0, 0), 1.2)
    if w is not None:
        detail = detail * w
    return np.rint(np.clip(gf + gain * detail, 0.0, 255.0)).astype(np.uint8)


def _sharpen_bgr(bgr, gain):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    w = edge_weight(lab[:, :, 0])
    lab[:, :, 0] = _sharpen_gray(lab[:, :, 0], gain, w)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_sharpen(img, amount):
    gain = min(3.0, max(0.0, amount) * 3.0)
    if img.ndim == 3 and img.shape[2] == 4:
        return np.dstack((_sharpen_bgr(img[:, :, :3], gain), img[:, :, 3]))
    if img.ndim == 3:
        return _sharpen_bgr(img, gain)
    return _sharpen_gray(img, gain)


def edge_weight(gray, thresh=0.60):
    smooth = cv2.GaussianBlur(gray, (9, 9), 0)
    gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    scale = float(np.percentile(mag, 98))
    if scale <= 0:
        scale = 1.0
    g = np.clip(mag / scale, 0.0, 1.0)
    w = (g > thresh).astype(np.float32)
    return cv2.GaussianBlur(w, (3, 3), 0)


def apply_edge_keep(orig, den, amount):
    orig = to_uint8(orig)
    if orig.ndim == 3 and orig.shape[2] == 4:
        w = edge_weight(cv2.cvtColor(orig[:, :, :3], cv2.COLOR_BGR2GRAY))[:, :, None]
        blended = np.rint(den[:, :, :3].astype(np.float32) * (1.0 - amount * w) + orig[:, :, :3].astype(np.float32) * (amount * w)).astype(np.uint8)
        return np.dstack((blended, orig[:, :, 3]))
    if orig.ndim == 3:
        w = edge_weight(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY))[:, :, None]
    else:
        w = edge_weight(orig)
    return np.rint(den.astype(np.float32) * (1.0 - amount * w) + orig.astype(np.float32) * (amount * w)).astype(np.uint8)


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


def process_file(src, dst, method, strength, fmt, quality, preview, edge_keep, fireflies, sharpen):
    src, dst = Path(src), Path(dst)
    raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError("unreadable or unsupported image")
    out = denoise_image(raw, method, strength, fireflies)
    if edge_keep > 0:
        out = apply_edge_keep(raw, out, min(1.0, max(0.0, edge_keep)))
    if sharpen > 0:
        out = apply_sharpen(out, sharpen)
    ext = ("." + fmt) if fmt else src.suffix.lower()
    if ext not in IMAGE_EXTS:
        ext = ".png"
    target = dst.with_suffix(ext)
    target.parent.mkdir(parents=True, exist_ok=True)
    params = []
    if ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        if out.ndim == 3 and out.shape[2] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)
        elif out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if raw.dtype == np.uint16 and ext not in (".jpg", ".jpeg"):
        out = np.clip(np.rint(out.astype(np.float32) * 257.0), 0, 65535).astype(np.uint16)
    if not cv2.imwrite(str(target), out, params):
        raise ValueError("failed to write output")
    if preview:
        prev = make_preview(raw, out)
        cv2.imwrite(str(target.with_name(target.stem + "_compare.png")), prev, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return target


def _worker(job):
    src, dst, method, strength, fmt, quality, preview, edge_keep, fireflies, sharpen = job
    t0 = time.perf_counter()
    try:
        target = process_file(src, dst, method, strength, fmt, quality, preview, edge_keep, fireflies, sharpen)
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
        description="Denoise photos and 3D renders fully locally with classical filters (no AI, no network).",
        epilog="examples:\n"
               "  python denoise.py photo.png\n"
               "  python denoise.py photos/ -o clean/ -m nlm -s 12 -j 4\n"
               "  python denoise.py noisy.png -m median -s 10 --format png --preview\n"
               "  python denoise.py photo.png -m nlm -s 12 --edge-keep 0.8\n"
               "  python denoise.py render.png --preset render          # 3D render: fireflies + wavelet + sharp edges\n"
               "  python denoise.py photos/ -o clean/ --preset photo -j 4\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", help="image file or directory (searched recursively for .png/.jpg/.jpeg/.tif/.tiff/.webp/.exr)")
    ap.add_argument("-o", "--output", help="output directory (default: <input>_denoised next to the input)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="workflow preset: photo=high-ISO photos (nlm + mild edge-keep + light sharpen), "
                         "render=3D renders (firefly removal + wavelet + strong edge-keep + sharpen); "
                         "explicit flags override the preset values")
    ap.add_argument("-m", "--method", choices=METHODS, default=None,
                    help="nlm=non-local means (best for photo noise), bilateral=edge-preserving smooth, "
                         "median=best for salt-and-pepper, gaussian=simple blur, wavelet=multi-scale thresholding "
                         "with separate luminance/chroma strength (darktable-style; default nlm, or preset)")
    ap.add_argument("-s", "--strength", type=float, default=None,
                    help="strength knob (default 10, or preset): nlm uses it as h directly, bilateral as sigma/7.5, "
                         "median/gaussian derive the kernel size from it, wavelet scales the wavelet threshold")
    ap.add_argument("--format", choices=["png", "jpg", "jpeg", "tif", "tiff", "webp", "exr"], default=None,
                    help="force output format (default: keep each input's format)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100 (default 95)")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="parallel workers for batch mode (default 1)")
    ap.add_argument("--preview", action="store_true", help="also write a side-by-side <name>_compare.png")
    ap.add_argument("--edge-keep", type=float, default=None, metavar="0-1",
                    help="blend edges detected in the original image back into the denoised result so edges stay "
                         "sharp (0=off, 1=keep original pixels at strong edges; try 0.6-0.9)")
    ap.add_argument("--fireflies", type=float, default=None, metavar="0-255",
                    help="pre-pass that replaces fireflies/hot pixels (pixels deviating more than T from the local "
                         "5x5 median) with the local median, before denoising; ideal for 3D renders (try 40-80), 0=off")
    ap.add_argument("--sharpen", type=float, default=None, metavar="0-1",
                    help="final output sharpening: luminance-only unsharp mask applied only at edges, so flat areas "
                         "and chroma noise are not amplified (0=off, try 0.2-0.4; applied last, after denoise/edge-keep)")
    args = ap.parse_args(argv)

    base = dict(method="nlm", strength=10.0, fireflies=0.0, edge_keep=0.0, sharpen=0.0)
    if args.preset:
        base.update(PRESETS[args.preset])
    method = args.method if args.method is not None else base["method"]
    strength = args.strength if args.strength is not None else base["strength"]
    fireflies = args.fireflies if args.fireflies is not None else base["fireflies"]
    edge_keep = args.edge_keep if args.edge_keep is not None else base["edge_keep"]
    sharpen = args.sharpen if args.sharpen is not None else base["sharpen"]

    inp = Path(args.input)
    if not inp.exists():
        sys.exit(f"error: no such file or directory: {inp}")
    if inp.is_file():
        if inp.suffix.lower() not in IMAGE_EXTS:
            sys.exit(f"error: unsupported extension {inp.suffix!r} (use {', '.join(sorted(IMAGE_EXTS))})")
        default_out = inp.parent / (inp.stem + "_denoised")
    else:
        default_out = inp.parent / (inp.name + "_denoised")
    out_dir = Path(args.output) if args.output else default_out

    files = collect_inputs(inp, out_dir)
    if not files:
        sys.exit("error: no images found")

    jobs = []
    for f in files:
        rel = f.relative_to(inp) if inp.is_dir() else Path(f.name)
        jobs.append((f, out_dir / rel, method, strength, args.format, args.quality, args.preview, edge_keep, fireflies, sharpen))

    total = len(jobs)
    print(f"denoising {total} image(s) | method={method} strength={strength} edge-keep={edge_keep} "
          f"fireflies={fireflies} sharpen={sharpen} -> {out_dir}")
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
