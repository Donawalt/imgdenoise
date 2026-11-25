# imgdenoise

Small, powerful image denoiser for **photography and 3D-render post-processing** that runs **fully locally with classical image processing — no AI, no models, no network**.

Supports **PNG, JPEG, TIFF, WebP and EXR** input, a **single image** or a **whole directory (batch)**, with parallel workers. Built for two workflows via `--preset`:

- **`photo`** — high-ISO photos: Gaussian/chroma noise → `nlm` + mild `--edge-keep` + light `--sharpen`
- **`render`** — 3D renders: `--fireflies` pre-pass (Monte-Carlo fireflies/hot pixels) + `wavelet` multi-scale denoise + strong `--edge-keep` + `--sharpen`, keeping hard edges crisp

## Install

```bash
pip install -r requirements.txt   # opencv-python-headless + numpy + PyWavelets, nothing else
```

## Usage

```bash
# single image -> ./photo_denoised/photo.png
python denoise.py photo.png

# batch a folder, 4 workers, stronger denoise
python denoise.py photos/ -o clean/ -m nlm -s 14 -j 4

# salt-and-pepper noise -> median filter, plus a side-by-side preview
python denoise.py noisy.png -m median -s 10 --preview

# keep edges razor-sharp: blend edges from the original back into the denoised image
python denoise.py photo.png -m nlm -s 12 --edge-keep 0.8

# one-command workflows (explicit flags override preset values)
python denoise.py photo.png --preset photo --preview
python denoise.py render.png --preset render
python denoise.py renders/ -o clean/ --preset render -j 4

# force JPEG output at quality 90
python denoise.py scan.png -o out/ --format jpg --quality 90
```

Run `python denoise.py -h` for the full option list.

## Methods (`-m`)

| method | best for | `-s` (default 10) means |
|---|---|---|
| `nlm` (default) | Gaussian/photo noise, keeps edges & textures | NLM `h` parameter directly |
| `bilateral` | smooth noise while preserving edges | `sigma = s * 7.5` |
| `median` | salt-and-pepper / impulse noise | kernel size derived from `s` (10 -> 5x5) |
| `gaussian` | mild noise, quick smoothing | kernel size derived from `s` |
| `wavelet` | render grain / multi-scale noise; luminance denoised 2x stronger than chroma (darktable-style) | scales the per-scale soft threshold (universal threshold at `s`=10) |

Higher `-s` = stronger denoising (more blur). Start at 10–15 for typical photos.

## Extra steps (pro workflow order: fireflies -> denoise -> edge-keep -> sharpen last)

- `--fireflies 0-255` — pre-pass for 3D renders: pixels deviating more than `T` from the local 5x5 median
  (fireflies, hot pixels) are replaced by that median **before** denoising. Try `40-80` for renders, `0` (off) for photos.
- `--edge-keep 0-1` — see Notes below.
- `--sharpen 0-1` — **output sharpening**, applied last: luminance-only unsharp mask, masked to edges only,
  so flat areas and chroma noise are never amplified and no color fringing is introduced. Try `0.2-0.4`.

## Notes

- Inputs are processed at 8-bit depth (float/16-bit EXR/TIFF are converted; 16-bit inputs are written back as 16-bit,
  JPEG output is always 8-bit). For HDR EXR pipelines, denoise a 16-bit TIFF instead.
- Input formats: `.png .jpg .jpeg .tif .tiff .webp .exr`.
- PNG alpha channels are preserved untouched.
- Output format defaults to each input's format; `--format` forces one format for all.
- Default output directory is `<input>_denoised` next to the input; `-o` overrides it. Directory structure is preserved.
- `--preview` writes `<name>_compare.png` (before | after) next to each output.
- `--edge-keep 0-1` detects edges in the original image (Sobel gradient, adaptive threshold) and blends the
  original pixels back in at strong edges, so edges stay sharp/"net" while flat areas are fully denoised.
  `0` = off (default), `1` = keep original pixels at edges. Try `0.6-0.9`; works with every method and keeps alpha intact.
- `--preset photo|render` sets sensible defaults for method/strength/fireflies/edge-keep/sharpen; any explicit
  flag (e.g. `-m nlm -s 15`) overrides the preset value.

## Test

```bash
python test_denoise.py   # synthesizes noisy images, runs the CLI, checks PSNR improves
```
