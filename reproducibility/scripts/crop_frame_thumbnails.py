"""Cut the frame thumbnails used in the two frame figures of the manuscript.

The figure with MLVU frames and their BLIP-base captions (frames/mlvu_00.png to
mlvu_11.png) and the figure with frames from the CMS.595 lectures
(frames/lecture_00.png to lecture_08.png) are assembled in LaTeX from single
thumbnails. The thumbnails were cut from the two composite figures of the first
submitted version of the manuscript: a 3 x 4 grid of MLVU frames and a 3 x 3
grid of lecture frames. The pixel boxes below are the cells of those grids; a
one-pixel border is removed from each cell.

Usage:
    python crop_frame_thumbnails.py mlvu_grid.png lecture_grid.png output_folder

Requirements: Pillow.
"""
import os
import sys

from PIL import Image

# Rows (top, bottom) and, for each row, the columns (left, right) of the MLVU grid.
MLVU_ROWS = [(26, 162), (220, 353), (403, 541)]
MLVU_COLS = [[(17, 241), (260, 489), (506, 734), (751, 977)],
             [(17, 240), (259, 486), (506, 732), (750, 975)],
             [(15, 239), (259, 485), (507, 732), (750, 975)]]
# Rows and columns of the lecture grid (the same columns in every row).
LECTURE_ROWS = [(9, 361), (499, 851), (991, 1343)]
LECTURE_COLS = [(13, 365), (377, 730), (743, 1095)]


def crop_grid(path, rows, cols_per_row, prefix, out):
    image = Image.open(path).convert("RGB")
    k = 0
    for (y0, y1), cols in zip(rows, cols_per_row):
        for x0, x1 in cols:
            image.crop((x0 + 1, y0 + 1, x1 - 1, y1 - 1)).save(os.path.join(out, f"{prefix}_{k:02d}.png"))
            k += 1
    return k


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    mlvu_grid, lecture_grid, out = sys.argv[1:]
    os.makedirs(out, exist_ok=True)
    n1 = crop_grid(mlvu_grid, MLVU_ROWS, MLVU_COLS, "mlvu", out)
    n2 = crop_grid(lecture_grid, LECTURE_ROWS, [LECTURE_COLS] * len(LECTURE_ROWS), "lecture", out)
    print(f"{n1} MLVU and {n2} lecture thumbnails written to {out}")


if __name__ == "__main__":
    main()
