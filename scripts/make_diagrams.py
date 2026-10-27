"""Generate the Velos diagrams as SVG in the website's dark palette.

    python scripts/make_diagrams.py website/public/images

Produces architecture.svg, spark-workflow.svg, validation-roadmap.svg. The same files are
rasterised to PNG for reports by scripts/render_diagrams.js (Playwright).
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

BG, CARD, TXT, SUB = "#0b1020", "#111a33", "#ffffff", "#9fb0c8"
CYAN, SKY, VIOLET, EMERALD, AMBER, ROSE = "#22d3ee", "#38bdf8", "#a78bfa", "#34d399", "#fbbf24", "#fb7185"
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def defs(extra: str = "", w: int = 1600, h: int = 800) -> str:
    return f"""<defs>
  <linearGradient id="g-accent" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{CYAN}"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient>
  <linearGradient id="g-line" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="{w}" y2="0"><stop offset="0" stop-color="{CYAN}"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient>
  <linearGradient id="g-vline" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="0" y2="{h}"><stop offset="0" stop-color="{CYAN}"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient>
  <radialGradient id="glow-c" cx="0.3" cy="0.1" r="0.7"><stop offset="0" stop-color="{CYAN}" stop-opacity="0.16"/><stop offset="1" stop-color="{CYAN}" stop-opacity="0"/></radialGradient>
  <radialGradient id="glow-v" cx="0.85" cy="0.9" r="0.7"><stop offset="0" stop-color="#8b5cf6" stop-opacity="0.18"/><stop offset="1" stop-color="#8b5cf6" stop-opacity="0"/></radialGradient>
  <filter id="shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="6" stdDeviation="10" flood-color="#000" flood-opacity="0.45"/></filter>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M0 0.5 L9.5 5 L0 9.5 z" fill="#8b5cf6"/></marker>
  <marker id="arrow-c" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse"><path d="M0 0.5 L9.5 5 L0 9.5 z" fill="{CYAN}"/></marker>
  {extra}
</defs>"""


def canvas(w: int, h: int, title: str, subtitle: str, body: str, extra_defs: str = "") -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="{FONT}">
{defs(extra_defs, w, h)}
<rect width="{w}" height="{h}" fill="{BG}"/>
<rect width="{w}" height="{h}" fill="url(#glow-c)"/>
<rect width="{w}" height="{h}" fill="url(#glow-v)"/>
<text x="{w/2}" y="64" text-anchor="middle" fill="{TXT}" font-size="36" font-weight="700" letter-spacing="-0.5">{esc(title)}</text>
<text x="{w/2}" y="98" text-anchor="middle" fill="{SUB}" font-size="18">{esc(subtitle)}</text>
{body}
</svg>"""


def wrapped(x: float, y: float, text: str, width_chars: int, size: int, fill: str, anchor: str = "middle",
            weight: int = 400, line_h: float | None = None) -> str:
    lines = textwrap.wrap(text, width_chars)
    lh = line_h or size * 1.35
    tspans = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else lh}">{esc(l)}</tspan>' for i, l in enumerate(lines))
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{fill}" font-size="{size}" font-weight="{weight}">{tspans}</text>'


def card(x, y, w, h, accent, title, body, title_size=24, body_size=16, body_chars=24, badge=None) -> str:
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{CARD}" fill-opacity="0.92" stroke="{accent}" stroke-opacity="0.75" stroke-width="1.5" filter="url(#shadow)"/>',
           f'<rect x="{x+18}" y="{y+1}" width="{w-36}" height="5" rx="2.5" fill="{accent}" fill-opacity="0.95"/>']
    title_chars = max(8, int((w - 28) / (title_size * 0.56)))
    n_title = len(textwrap.wrap(title, title_chars))
    ty = y + 50
    if badge:
        out.append(f'<circle cx="{x+34}" cy="{y+44}" r="18" fill="url(#g-accent)"/>'
                   f'<text x="{x+34}" y="{y+50}" text-anchor="middle" fill="#fff" font-size="17" font-weight="700">{esc(badge)}</text>')
        out.append(wrapped(x + 62, ty, title, title_chars, title_size, TXT, anchor="start", weight=600, line_h=title_size * 1.15))
    else:
        out.append(wrapped(x + w / 2, ty, title, title_chars, title_size, TXT, weight=600, line_h=title_size * 1.15))
    body_y = ty + title_size * 1.15 * (n_title - 1) + 34
    out.append(wrapped(x + w / 2, body_y, body, body_chars, body_size, SUB))
    return "\n".join(out)


def harrow(x1, y, x2, color="url(#g-line)", marker="arrow", width=3, dash=None) -> str:
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="{width}" marker-end="url(#{marker})"{d}/>'


# ---------------------------------------------------------------------------------- 1. architecture
def architecture() -> str:
    W, H = 1600, 760
    row1 = [("630 nm source", "Diode laser / LED + spatial filter", CYAN),
            ("Probe optics", "Beam shaping and aperture", CYAN),
            ("Live-cell sample", "Culture dish on XY stage; optional incubator enclosure", EMERALD),
            ("Diffraction field", "Lensless or relay-assisted propagation", SKY),
            ("CMOS camera", "Raw intensity frames", SKY),
            ("Acquisition computer", "Stage–camera synchronisation; metadata and quality control", VIOLET)]
    row2 = [("Calibration & QC", "Dark/flat correction, position check, background and saturation control", AMBER),
            ("SPARK reconstruction", "Physics-based, sparse-scanning and AI-assisted reconstruction", VIOLET),
            ("Quantitative outputs", "Amplitude, phase, dry-mass proxies, time-series features", CYAN),
            ("Biological analysis", "Morphology, growth, migration, drug response and organoid metrics", EMERALD)]
    b = []
    cw, ch, gap, y1 = 222, 186, 32, 140
    x0 = (W - (6 * cw + 5 * gap)) / 2
    xs1 = [x0 + i * (cw + gap) for i in range(6)]
    for i, (t, d, c) in enumerate(row1):
        b.append(card(xs1[i], y1, cw, ch, c, t, d, title_size=22, body_size=15, body_chars=22))
        if i < 5:
            b.append(harrow(xs1[i] + cw + 4, y1 + ch / 2, xs1[i + 1] - 6))
    cw2, gap2, y2 = 322, 50, 500
    x02 = (W - (4 * cw2 + 3 * gap2)) / 2
    xs2 = [x02 + i * (cw2 + gap2) for i in range(4)]
    for i, (t, d, c) in enumerate(row2):
        b.append(card(xs2[i], y2, cw2, 168, c, t, d, title_size=23, body_size=15, body_chars=34))
        if i < 3:
            b.append(harrow(xs2[i] + cw2 + 4, y2 + 84, xs2[i + 1] - 6))
    # control loop: acquisition computer → sample stage (dashed cyan)
    xc6 = xs1[5] + cw / 2; xc3 = xs1[2] + cw / 2
    yc = y1 + ch + 62
    b.append(f'<path d="M{xc6 - 40} {y1 + ch} V{yc} H{xc3} V{y1 + ch + 10}" fill="none" stroke="{CYAN}" stroke-width="2.5" stroke-dasharray="8 7" marker-end="url(#arrow-c)"/>')
    b.append(f'<text x="{(xc6 + xc3) / 2}" y="{yc - 10}" text-anchor="middle" fill="{CYAN}" font-size="15" font-weight="600">Scan control: controlled overlapping scan positions and triggers</text>')
    # data flow: acquisition computer → calibration (gradient)
    yd = y2 - 48; xd1 = xs2[0] + cw2 / 2
    b.append(f'<path d="M{xc6 + 40} {y1 + ch} V{yd} H{xd1} V{y2 - 8}" fill="none" stroke="url(#g-line)" stroke-width="3" marker-end="url(#arrow)"/>')
    b.append(f'<text x="{(xc6 + xd1) / 2}" y="{yd + 24}" text-anchor="middle" fill="{SUB}" font-size="15">Raw frames + scan coordinates + acquisition metadata</text>')
    # row labels
    b.append(f'<text x="{x0}" y="{y1 - 14}" fill="{CYAN}" font-size="13" font-weight="700" letter-spacing="2.5">ACQUISITION (VELOS QPI HARDWARE)</text>')
    b.append(f'<text x="{x02}" y="{y2 - 70}" fill="{VIOLET}" font-size="13" font-weight="700" letter-spacing="2.5">RECONSTRUCTION AND ANALYSIS (SPARK SOFTWARE)</text>')
    return canvas(W, H, "Velos QPI: acquisition and reconstruction architecture",
                  "From a 630 nm probe scanning a living sample to quantitative, biologically meaningful outputs", "\n".join(b))


# ---------------------------------------------------------------------------------- 2. SPARK workflow
def spark_workflow() -> str:
    W, H = 1600, 1000
    steps = [("Import", "Intensity frames, scan coordinates, wavelength, pixel size", CYAN),
             ("Quality control", "Saturation, drift, missing frames, position consistency", SKY),
             ("Initialization", "Probe/object estimate, support, calibration parameters", SKY),
             ("Reconstruction", None, VIOLET),
             ("Quantification", "Phase, amplitude, optical path delay, cell features", EMERALD),
             ("Export & report", "Images, videos, masks, CSV metrics, reproducible settings", EMERALD)]
    b = []
    x, w, h, gap, y0 = 330, 1210, 108, 28, 140
    spine_x = 250
    ys = [y0 + i * (h + gap) for i in range(6)]
    b.append(f'<line x1="{spine_x}" y1="{ys[0] + h/2}" x2="{spine_x}" y2="{ys[-1] + h/2}" stroke="url(#g-vline)" stroke-width="4" stroke-linecap="round"/>')
    for i, (t, d, c) in enumerate(steps):
        y = ys[i]; cy = y + h / 2
        b.append(f'<line x1="{spine_x}" y1="{cy}" x2="{x}" y2="{cy}" stroke="{c}" stroke-opacity="0.6" stroke-width="2"/>')
        b.append(f'<circle cx="{spine_x}" cy="{cy}" r="28" fill="{BG}" stroke="url(#g-accent)" stroke-width="3"/>'
                 f'<text x="{spine_x}" y="{cy + 9}" text-anchor="middle" fill="#fff" font-size="24" font-weight="700">{i + 1}</text>')
        strong = (d is None)
        b.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="18" fill="{CARD}" fill-opacity="{0.98 if strong else 0.92}" stroke="{c}" stroke-opacity="{0.95 if strong else 0.7}" stroke-width="{2.2 if strong else 1.5}" filter="url(#shadow)"/>')
        b.append(f'<rect x="{x}" y="{y}" width="7" height="{h}" rx="3.5" fill="{c}"/>')
        b.append(f'<text x="{x + 36}" y="{cy + 9}" fill="{TXT}" font-size="26" font-weight="600">{esc(t)}</text>')
        if strong:
            chips = [("Physics-based", CYAN), ("Sparse-scanning", VIOLET), ("AI-assisted", EMERALD), ("Hybrid", AMBER)]
            cx = x + 400
            for name, cc in chips:
                tw = len(name) * 10.5 + 36
                b.append(f'<rect x="{cx}" y="{cy - 20}" width="{tw}" height="40" rx="20" fill="{cc}" fill-opacity="0.14" stroke="{cc}" stroke-opacity="0.8" stroke-width="1.5"/>'
                         f'<text x="{cx + tw/2}" y="{cy + 7}" text-anchor="middle" fill="{cc}" font-size="18" font-weight="600">{esc(name)}</text>')
                cx += tw + 18
            b.append(f'<text x="{cx + 6}" y="{cy + 7}" fill="{SUB}" font-size="16">+ quality indicators</text>')
        else:
            b.append(f'<text x="{x + 420}" y="{cy + 8}" fill="{SUB}" font-size="19">{esc(d)}</text>')
    return canvas(W, H, "SPARK software: from raw measurements to biological decisions",
                  "One data model, one calibration and one quality-control path shared by every reconstruction engine", "\n".join(b))


# ---------------------------------------------------------------------------------- 3. validation roadmap
def validation_roadmap() -> str:
    W, H = 1600, 660
    stages = [("A", "Optical & computational baseline", "USAF target, phase target, microspheres", "Resolution, phase accuracy, repeatability", CYAN),
              ("B", "Fixed biological samples", "Pollen, onion epidermis, fixed cells, blood smears", "Morphology and contrast verification", SKY),
              ("C", "Live 2D cells", "Yeast, adherent mammalian cells", "Growth, migration, dry-mass proxy, viability", EMERALD),
              ("D", "Drug-response assays", "Dose and time-course experiments", "Sensitivity, repeatability, dose-response features", VIOLET),
              ("E", "3D models", "Spheroids and small organoids", "Penetration limits, morphology and treatment response", ROSE)]
    b = []
    cw, ch, gap, y = 276, 430, 38, 150
    x0 = (W - (5 * cw + 4 * gap)) / 2
    # progress rail behind cards
    b.append(f'<line x1="{x0 + cw/2}" y1="{y - 30}" x2="{x0 + 4*(cw+gap) + cw/2}" y2="{y - 30}" stroke="url(#g-line)" stroke-width="4" stroke-linecap="round"/>')
    for i, (L, name, samples, evidence, c) in enumerate(stages):
        x = x0 + i * (cw + gap)
        b.append(f'<circle cx="{x + cw/2}" cy="{y - 30}" r="9" fill="{BG}" stroke="{c}" stroke-width="3"/>')
        b.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="18" fill="{CARD}" fill-opacity="0.92" stroke="{c}" stroke-opacity="0.75" stroke-width="1.5" filter="url(#shadow)"/>')
        b.append(f'<rect x="{x+18}" y="{y+1}" width="{cw-36}" height="5" rx="2.5" fill="{c}"/>')
        b.append(f'<circle cx="{x + 40}" cy="{y + 54}" r="22" fill="{c}" fill-opacity="0.16" stroke="{c}" stroke-width="2"/>'
                 f'<text x="{x + 40}" y="{y + 62}" text-anchor="middle" fill="{c}" font-size="22" font-weight="700">{L}</text>')
        b.append(f'<text x="{x + 74}" y="{y + 48}" fill="{SUB}" font-size="13" font-weight="700" letter-spacing="2">STAGE {L}</text>')
        b.append(wrapped(x + 74, y + 72, name, 17, 20, TXT, anchor="start", weight=600, line_h=24))
        b.append(f'<text x="{x + 24}" y="{y + 160}" fill="{c}" font-size="12.5" font-weight="700" letter-spacing="2.2">SAMPLES</text>')
        b.append(wrapped(x + 24, y + 186, samples, 26, 16, TXT, anchor="start", line_h=22))
        b.append(f'<text x="{x + 24}" y="{y + 290}" fill="{c}" font-size="12.5" font-weight="700" letter-spacing="2.2">PRIMARY EVIDENCE</text>')
        b.append(wrapped(x + 24, y + 316, evidence, 26, 16, TXT, anchor="start", line_h=22))
        if i < 4:
            ax = x + cw + gap / 2
            b.append(f'<path d="M{ax - 7} {y + ch/2 - 12} L{ax + 7} {y + ch/2} L{ax - 7} {y + ch/2 + 12}" fill="none" stroke="url(#g-line)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>')
    b.append(f'<text x="{W/2}" y="{H - 28}" text-anchor="middle" fill="{SUB}" font-size="15">Each stage must pass its evidence criteria before claims are made at the next stage; 3D claims stay limited to projected and surface features.</text>')
    return canvas(W, H, "Progressive validation roadmap",
                  "From calibration targets to live cells, drug-response assays and 3D models", "\n".join(b))


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "website/public/images")
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in (("architecture", architecture), ("spark-workflow", spark_workflow), ("validation-roadmap", validation_roadmap)):
        (out / f"{name}.svg").write_text(fn(), encoding="utf-8")
        print("wrote", out / f"{name}.svg")


if __name__ == "__main__":
    main()
