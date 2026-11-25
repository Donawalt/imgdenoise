# imgdenoise

Small, powerful image denoiser that runs **fully locally with classical image processing — no AI, no models, no network**.

Supports **PNG and JPEG**, a **single image** or a **whole directory (batch)**, with parallel workers.

## Install

```bash
pip install -r requirements.txt   # opencv-python-headless + numpy, nothing else
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

Higher `-s` = stronger denoising (more blur). Start at 10–15 for typical photos.

## Notes

- Inputs are processed at 8-bit depth; outputs are 8-bit PNG/JPEG (JPEG is always 8-bit).
- PNG alpha channels are preserved untouched.
- Output format defaults to each input's format; `--format` forces one format for all.
- Default output directory is `<input>_denoised` next to the input; `-o` overrides it. Directory structure is preserved.
- `--preview` writes `<name>_compare.png` (before | after) next to each output.
- `--edge-keep 0-1` detects edges in the original image (Sobel gradient, adaptive threshold) and blends the
  original pixels back in at strong edges, so edges stay sharp/"net" while flat areas are fully denoised.
  `0` = off (default), `1` = keep original pixels at edges. Try `0.6-0.9`; works with every method and keeps alpha intact.

## Test

```bash
python test_denoise.py   # synthesizes noisy images, runs the CLI, checks PSNR improves
```
