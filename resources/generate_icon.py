"""Generate the LingSu Protein app icon (macOS .icns + Windows .ico +
multi-resolution .png). The icon is a stylized α-helix wrapped around
a vertical bar with a colored gradient — chosen to read clearly at
16×16 (dock micro icon) and at 1024×1024 (App Store / Retina).

Run with `python resources/generate_icon.py` from the project root.

Output files:
  resources/icon_16.png ... icon_1024.png   (multi-resolution PNGs)
  resources/icon.iconset/                    (macOS iconset dir)
  resources/icon.icns                        (macOS, via iconutil)
  resources/icon.ico                         (Windows)

On macOS this script will automatically invoke `iconutil` to assemble
.icns. On other platforms the .icns file is omitted but the PNGs are
still produced for QIcon to use.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


# Icon size ladder. 1024 is the source master; we downsample with
# LANCZOS so the helix stays crisp at every DPI.
SIZES = [16, 24, 32, 48, 64, 128, 256, 512, 1024]
OUTPUT_DIR = Path(__file__).parent
ICONSET_DIR = OUTPUT_DIR / "icon.iconset"


def _helix_gradient(t: float) -> tuple[int, int, int]:
    """A teal→cyan→lime gradient that reads at small sizes.

    The visual brief is "soft teal with a hint of green" — closer to a
    modern life-sciences UI than the usual saturated primary blue. At
    t=0 we're a deep teal (28, 110, 130); at t=1 we go to a fresh
    cyan-green (90, 220, 180).
    """
    r0, g0, b0 = (28, 110, 130)
    r1, g1, b1 = (90, 220, 180)
    return (
        int(r0 + (r1 - r0) * t),
        int(g0 + (g1 - g0) * t),
        int(b0 + (b1 - b0) * t),
    )


def _render_master(size: int) -> Image.Image:
    """Render the master icon at the given square ``size``.

    The icon is a rounded square with a soft inner glow and a stylized
    α-helix in the foreground. At small sizes the helix reads as a
    single ribbon; at large sizes the spiral structure becomes
    visible.
    """
    # Render at 4x and downsample so anti-aliasing is generous.
    scale = 4
    big = size * scale
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1) Rounded-square background with vertical gradient.
    radius = int(big * 0.22)
    for y in range(big):
        t = y / max(1, big - 1)
        r, g, b = _helix_gradient(t)
        draw.line([(0, y), (big, y)], fill=(r, g, b, 255))

    # Mask to rounded rect
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, big - 1, big - 1), radius=radius, fill=255
    )
    bg = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    bg.paste(img, (0, 0), mask)
    img = bg
    draw = ImageDraw.Draw(img)

    # 2) Soft inner glow ring (lighter teal)
    glow = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.rounded_rectangle(
        (int(big * 0.06), int(big * 0.06), int(big * 0.94), int(big * 0.94)),
        radius=int(big * 0.18), outline=(255, 255, 255, 70), width=int(big * 0.015),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=big * 0.012))
    img = Image.alpha_composite(img, glow)
    draw = ImageDraw.Draw(img)

    # 3) Stylized α-helix: two intertwined sine ribbons drawn with
    # thick polylines so the helix reads as continuous twists at any
    # size. No horizontal "rungs" — they make the icon look like a
    # gear at small sizes and obscure the spiral.
    cx = big / 2
    top = big * 0.16
    bottom = big * 0.84
    height = bottom - top
    n_turns = 2.6
    samples = 380
    amp = big * 0.28

    def strand_points(phase: float) -> list[tuple[float, float]]:
        pts = []
        for i in range(samples + 1):
            u = i / samples
            y = top + height * u
            x = cx + amp * math.sin(2 * math.pi * n_turns * u + phase)
            pts.append((x, y))
        return pts

    # Back strand: very subtle, near-white but slightly cooler — sits
    # behind, visible only at the crossover points.
    pts_back = strand_points(phase=0.0)
    draw.line(pts_back, fill=(220, 235, 240, 210),
              width=int(big * 0.085), joint="curve")

    # Front strand: bright white, phase-shifted by π so the two
    # strands interleave as a true double helix.
    pts_front = strand_points(phase=math.pi)
    draw.line(pts_front, fill=(255, 255, 255, 250),
              width=int(big * 0.085), joint="curve")

    # 5) Outer rounded-square mask (for the inner contents too, in
    # case any stroke leaks past the radius)
    final_mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(final_mask).rounded_rectangle(
        (0, 0, big - 1, big - 1), radius=radius, fill=255
    )
    img.putalpha(final_mask)

    # Downsample
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _write_pngs() -> list[Path]:
    """Write one PNG per size and return the list of paths."""
    paths = []
    for s in SIZES:
        out = OUTPUT_DIR / f"icon_{s}.png"
        master = _render_master(s)
        master.save(out, format="PNG", optimize=True)
        paths.append(out)
        print(f"  wrote {out.name} ({s}x{s})")
    return paths


def _build_iconset(png_paths: list[Path]) -> None:
    """Lay out the .iconset directory with the names iconutil expects."""
    if ICONSET_DIR.exists():
        shutil.rmtree(ICONSET_DIR)
    ICONSET_DIR.mkdir(parents=True)
    mapping = {
        16:   "icon_16x16.png",
        32:   "icon_16x16@2x.png",   # 32 == 16@2x
        32:   "icon_32x32.png",
        64:   "icon_32x32@2x.png",
        128:  "icon_128x128.png",
        256:  "icon_128x128@2x.png",
        256:  "icon_256x256.png",
        512:  "icon_256x256@2x.png",
        512:  "icon_512x512.png",
        1024: "icon_512x512@2x.png",
    }
    # Avoid the dict-collapse: write each size to its @2x counterpart
    # for the half-size ladder (Apple wants both base and @2x).
    pairs = [
        (16, "icon_16x16.png"),
        (32, "icon_16x16@2x.png"),
        (32, "icon_32x32.png"),
        (64, "icon_32x32@2x.png"),
        (128, "icon_128x128.png"),
        (256, "icon_128x128@2x.png"),
        (256, "icon_256x256.png"),
        (512, "icon_256x256@2x.png"),
        (512, "icon_512x512.png"),
        (1024, "icon_512x512@2x.png"),
    ]
    by_size = {p.stem.split("_")[1]: p for p in png_paths}
    for s, name in pairs:
        src = by_size[f"{s}"]
        shutil.copy(src, ICONSET_DIR / name)
    print(f"  wrote iconset at {ICONSET_DIR}")


def _build_icns() -> None:
    """Run macOS ``iconutil`` to assemble .icns from the iconset."""
    icns_path = OUTPUT_DIR / "icon.icns"
    if icns_path.exists():
        icns_path.unlink()
    try:
        subprocess.run(
            ["iconutil", "-c", "icns", str(ICONSET_DIR), "-o", str(icns_path)],
            check=True, capture_output=True,
        )
        print(f"  wrote {icns_path.name}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"  iconutil failed ({e}); .icns not built (non-macOS host?).")


def _build_ico(png_paths: list[Path]) -> None:
    """Assemble a multi-resolution .ico for Windows."""
    ico_path = OUTPUT_DIR / "icon.ico"
    # PIL wants the largest first; we pick the ones that match Windows
    # icon conventions.
    win_sizes = [16, 24, 32, 48, 64, 128, 256]
    images = []
    for s in win_sizes:
        p = OUTPUT_DIR / f"icon_{s}.png"
        if p.exists():
            images.append(Image.open(p))
    if images:
        images[0].save(
            ico_path, format="ICO", sizes=[(s, s) for s in win_sizes],
            append_images=images[1:],
        )
        print(f"  wrote {ico_path.name}")


def main() -> int:
    print(f"Generating LingSu Protein icon -> {OUTPUT_DIR}")
    png_paths = _write_pngs()
    _build_iconset(png_paths)
    _build_icns()
    _build_ico(png_paths)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())