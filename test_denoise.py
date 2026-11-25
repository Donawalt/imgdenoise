#!/usr/bin/env python3
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
CLI = HERE / "denoise.py"


def make_clean():
    img = np.zeros((240, 320, 3), np.uint8)
    for x in range(320):
        img[:, x] = (x * 255 // 319, 128, 255 - x * 255 // 319)
    cv2.circle(img, (90, 120), 50, (0, 200, 80), -1)
    cv2.rectangle(img, (180, 60), (290, 190), (40, 90, 220), -1)
    cv2.putText(img, "DENOISE", (88, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def run(*args):
    r = subprocess.run([sys.executable, str(CLI), *map(str, args)], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        raise SystemExit(f"CLI failed: {args}")
    return r.stdout


def main():
    rng = np.random.default_rng(42)
    clean = make_clean()
    noisy_g = np.clip(clean.astype(np.float32) + rng.normal(0, 25, clean.shape), 0, 255).astype(np.uint8)
    noisy_sp = clean.copy()
    m1 = rng.random(clean.shape[:2]) < 0.03
    m2 = rng.random(clean.shape[:2]) < 0.03
    noisy_sp[m1] = 255
    noisy_sp[m2] = 0

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        batch = td / "batch"
        batch.mkdir()
        cv2.imwrite(str(batch / "a.png"), noisy_g)
        cv2.imwrite(str(batch / "b.jpg"), noisy_g, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(td / "noisy.png"), noisy_g)
        cv2.imwrite(str(td / "noisy_sp.png"), noisy_sp)
        rgba = cv2.cvtColor(clean, cv2.COLOR_BGR2BGRA)
        alpha = np.full(rgba.shape[:2], 128, np.uint8)
        alpha[60:180, 80:240] = 255
        rgba[:, :, 3] = alpha
        cv2.imwrite(str(batch / "rgba.png"), rgba)

        out_single = td / "out_single"
        run(td / "noisy.png", "-o", out_single, "-m", "nlm", "-s", "14")
        den = cv2.imread(str(out_single / "noisy.png"), cv2.IMREAD_UNCHANGED)
        p_before, p_after = psnr(noisy_g, clean), psnr(den, clean)
        print(f"nlm single: PSNR noisy={p_before:.2f} dB -> denoised={p_after:.2f} dB")
        assert den is not None and p_after > p_before + 2, "NLM did not improve PSNR"

        out_sp = td / "out_sp"
        run(td / "noisy_sp.png", "-o", out_sp, "-m", "median", "-s", "10", "--preview")
        den_sp = cv2.imread(str(out_sp / "noisy_sp.png"), cv2.IMREAD_UNCHANGED)
        p_before, p_after = psnr(noisy_sp, clean), psnr(den_sp, clean)
        print(f"median salt&pepper: PSNR noisy={p_before:.2f} dB -> denoised={p_after:.2f} dB")
        assert p_after > p_before + 2, "median did not improve PSNR"
        assert (out_sp / "noisy_sp_compare.png").exists(), "preview missing"

        out_batch = td / "out_batch"
        run(batch, "-o", out_batch, "-m", "nlm", "-s", "12", "-j", "2", "--edge-keep", "0.8")
        assert (out_batch / "a.png").exists()
        assert (out_batch / "b.jpg").exists()
        den_a = cv2.imread(str(out_batch / "a.png"), cv2.IMREAD_UNCHANGED)
        print(f"batch nlm: PSNR noisy={psnr(noisy_g, clean):.2f} dB -> denoised={psnr(den_a, clean):.2f} dB")
        assert psnr(den_a, clean) > psnr(noisy_g, clean) + 2, "batch NLM did not improve PSNR"

        den_rgba = cv2.imread(str(out_batch / "rgba.png"), cv2.IMREAD_UNCHANGED)
        assert den_rgba.shape[2] == 4, "alpha channel lost"
        assert np.array_equal(den_rgba[:, :, 3], alpha), "alpha channel modified"
        print("alpha channel preserved exactly")

        out_bil = td / "out_bil"
        run(td / "noisy.png", "-o", out_bil, "-m", "bilateral", "-s", "10")
        out_gau = td / "out_gau"
        run(td / "noisy.png", "-o", out_gau, "-m", "gaussian", "-s", "10")
        assert (out_bil / "noisy.png").exists() and (out_gau / "noisy.png").exists()
        print(f"bilateral: {psnr(noisy_g, clean):.2f} -> {psnr(cv2.imread(str(out_bil / 'noisy.png')), clean):.2f} dB")
        print(f"gaussian:  {psnr(noisy_g, clean):.2f} -> {psnr(cv2.imread(str(out_gau / 'noisy.png')), clean):.2f} dB")

        out_ek = td / "out_ek"
        run(td / "noisy.png", "-o", out_ek, "-m", "gaussian", "-s", "10", "--edge-keep", "0.9")
        den_g = cv2.imread(str(out_gau / "noisy.png"), cv2.IMREAD_UNCHANGED)
        den_ek = cv2.imread(str(out_ek / "noisy.png"), cv2.IMREAD_UNCHANGED)

        def grad_mag(img):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))

        grad_clean = grad_mag(clean)
        edge_band = grad_clean > np.percentile(grad_clean, 90)
        flat = ~cv2.dilate(edge_band.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        err_g = float(np.mean(np.abs(grad_mag(den_g) - grad_clean)[edge_band]))
        err_ek = float(np.mean(np.abs(grad_mag(den_ek) - grad_clean)[edge_band]))
        mse_g = float(np.mean((den_g.astype(np.float64) - clean.astype(np.float64))[flat] ** 2))
        mse_ek = float(np.mean((den_ek.astype(np.float64) - clean.astype(np.float64))[flat] ** 2))
        print(f"edge-keep: edge gradient error gaussian={err_g:.1f} -> edge-keep={err_ek:.1f}; flat MSE {mse_g:.1f} -> {mse_ek:.1f}")
        assert err_ek < err_g, "edge-keep did not keep edges sharper"
        assert mse_ek <= mse_g * 1.05 + 1e-6, "edge-keep degraded flat areas"

        out_jpg = td / "out_jpg"
        run(td / "noisy.png", "-o", out_jpg, "--format", "jpg", "--quality", "90")
        j = cv2.imread(str(out_jpg / "noisy.jpg"), cv2.IMREAD_UNCHANGED)
        assert j is not None and j.ndim == 3 and j.shape[2] == 3, "jpg conversion failed"
        print("forced jpg output ok (3-channel)")

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
