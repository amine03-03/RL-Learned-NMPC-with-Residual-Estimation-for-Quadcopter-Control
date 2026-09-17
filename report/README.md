# PFE report

LaTeX source for the final-year project report.

```
report/
├── report.tex        the whole document, single file
├── logos/            institution.png (left) and school.png (right)
└── README.md
```

## Build

```bash
cd report && latexmk -pdf report.tex
# or, equivalently, from the repository root:
latexmk -pdf report/report.tex
```

Both invocations produce the same PDF: figure and logo paths are probed
against every root the build might be run from, so the working directory does
not matter.

Three passes are needed for the table of contents, the lists of figures and
tables, and `cleveref`'s forward references; `latexmk` handles that.

On a Debian/Ubuntu machine the required TeX Live set is

```bash
apt-get install --no-install-recommends latexmk texlive-latex-base \
    texlive-latex-recommended texlive-latex-extra texlive-pictures \
    texlive-science texlive-fonts-recommended texlive-lang-french
```

Last verified: 94 pages, no errors, no undefined references, no overfull
boxes.

The document degrades gracefully: it compiles with no logos, with no figures,
and without `french.ldf` or `lmodern` (only the résumé's hyphenation and the
font quality change). Nothing in it requires a build step outside LaTeX.

## Logos

Drop two files into `logos/`:

| File | Position | Content |
|---|---|---|
| `logos/institution.png` | top left | host organisation |
| `logos/school.png` | top right | school |

Until they exist, each renders as a labelled frame naming the missing file, so
the layout is visible while the logos are being sourced. For `.pdf` logos,
change the extension in the two `\logoslot` calls on the title page.

## Figures

Figures are read from `../artifacts/figures/` **by path**, so re-running the
study updates them in place with no edit to the `.tex`. All 27 figures the
notebooks produce are placed in the document.

A figure that has not been generated renders as a frame naming the missing
artefact rather than breaking the build, so a partial run still produces a
readable PDF.

```bash
cd study/notebooks
for n in nb?_*.py; do python "$n" || break; done
```

## Filling in the results

**`\BL` marks every cell awaiting the full-scale campaign** — 455 of them. Each
results table names its source CSV in the caption, so every blank has an
unambiguous origin:

| Table | Source |
|---|---|
| `tab:res-classic` | `artifacts/lltc/nb1_classical.csv` |
| `tab:res-horizon` | `artifacts/common/nb1_horizon.csv` |
| `tab:res-noise` | `artifacts/common/nb1_noise.csv` |
| `tab:res-qr` | `artifacts/common/nb1_qr_spread.csv` |
| `tab:res-lltcfit` | `artifacts/lltc/nb2_fit.csv` |
| `tab:res-lltc` | `artifacts/lltc/nb2_horizon_equivalence.csv` |
| `tab:res-rep`, `tab:res-acmpc-sweeps` | `artifacts/acmpc/nb3_*.csv` |
| `tab:res-dr`, `tab:res-seeds`, `tab:res-forget` | `artifacts/domrand/*.csv` |
| `tab:res-trainhealth`, `tab:res-rdp`, `tab:res-dmod`, `tab:res-scenarios` | `artifacts/acmpc_adaptive/nb5_*.csv` |
| `tab:ledger`, `tab:res-degradation` | `artifacts/common/{ledger,nb6_degradation}.csv` |

The results chapter is deliberately **not** filled from a `smoke` run. That
configuration produces undertrained policies by construction, and the study's
own gates say so; see the reading convention at the head of the results
chapter.

`\todo{...}` marks the prose left to write, mostly the per-increment analyses
that can only be written once the tables hold numbers.

## Constants

Physical constants are defined once, in the macro block near the top of
`report.tex` (`\vUhover`, `\vTmax`, `\vDadc`, …), and come from
`artifacts/common/nb1_selftest.csv`, which the physics self-test regenerates and
asserts at import. **Do not retype a number into the body** — add or edit a
macro, so the document cannot drift from the code.

These values reflect the corrected physics: the PX4 idle floor, the split
between the allocator's `k_m` and the plant's rotor drag, and the reachable
rate box. Several differ from a textbook quadrotor model, and the differences
are explained where they are introduced.
