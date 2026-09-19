# `report/figures/` — drop your own plots here

This directory is **first** on the report's `\graphicspath`, so anything you put
here overrides the generated figure of the same name without editing the
document.

## Two ways to use your own image

**1. Same name — nothing to edit.** Put a file here named exactly as the
generated one (`F21_headline.png`, `F3_horizon.png`, …) and it wins.

**2. Your own name — one line to edit.** Put `my_plot.pdf` here, then change the
matching line in the `FIGURE FILES` block near the top of `report.tex`:

```latex
\renewcommand{\figHeadline}{my_plot.pdf}
```

Every figure in the report has exactly one such macro. The block lists them
grouped by chapter, with the generated file each one currently points at.

## Formats

PDF, PNG, JPG and EPS all work. **PDF is best for plots** — it stays sharp at
any zoom and keeps text selectable. From matplotlib:

```python
fig.savefig("report/figures/my_plot.pdf", bbox_inches="tight")
```

## Sizing

The height passed to `\realfig{62mm}{...}` is the *reserved* height; the image
is scaled to fit the text width with `keepaspectratio`, so a wide plot will be
width-limited and shorter than the number suggests. Change that first argument
if a figure needs more or less vertical room.

## A missing file does not break the build

A macro pointing at a file that does not exist renders as a labelled grey frame
naming what is missing, so the document always compiles and the gap is visible
in the PDF.

## This directory is not generated

`artifacts/figures/` is rewritten by the notebooks. This directory is not
touched by any script, so your own images are safe here.
