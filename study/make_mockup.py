#!/usr/bin/env python3
"""Build report/report_mockup.tex from report/report.tex.

WHAT THIS IS
------------
A **mockup** of the report: the same document, the same structure and the same
argument, but carrying the numbers the study would produce if every increment
delivered on its claim, and drawing every figure in TikZ/pgfplots instead of
including a PNG.

WHY IT EXISTS
-------------
It is a layout and narrative rehearsal.  With the real numbers, Increment 1
fails and Increment 3's effect is smaller than the seed spread, so the document
never shows what the finished argument looks like when the story runs straight.
The mockup does -- and because every figure is drawn from coordinates written
into the file, it compiles on a machine with nothing but TeX Live: no
``artifacts/``, no CSVs, no notebook run, no PNGs.

THE NUMBERS IN IT ARE NOT MEASUREMENTS.  The word MOCKUP appears once beside the
title and once in the abstract for exactly that reason.  Do not quote from it.

USAGE
-----
    python3 study/make_mockup.py          # write report/report_mockup.tex
    python3 study/make_mockup.py --check  # verify every substitution still applies
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "report", "report.tex")
DST = os.path.join(ROOT, "report", "report_mockup.tex")

_applied, _missing = [], []


def sub(s, old, new, tag):
    """Replace exactly once, recording whether the anchor was still there.

    The mockup is generated from a document that keeps changing, so a silently
    skipped substitution is the failure mode worth catching: it would leave a
    real measurement sitting in a document labelled mockup, or the reverse.
    """
    if old not in s:
        _missing.append(tag)
        return s
    _applied.append(tag)
    return s.replace(old, new, 1)


# =========================================================================== #
#  1.  Preamble: pgfplots, the drawn-figure machinery, shared styles
# =========================================================================== #
PREAMBLE = r"""
%=======================================================================
%  MOCKUP: DRAWN FIGURES
%
%  Every figure in this document is DRAWN from coordinates written below,
%  not included from a file.  The document therefore compiles with nothing
%  but TeX Live -- no artifacts/, no CSVs, no notebook run, no images.
%
%  \figslot is redefined to typeset its third argument instead of loading a
%  file, and each \figXxx macro from the FIGURE FILES block is redefined to
%  a picture.  Nothing else in the body changes, so the mockup and the real
%  report keep the same figure placement, sizing and captions.
%=======================================================================
\usepackage{pgfplots}
\pgfplotsset{compat=1.16}
\usepgfplotslibrary{groupplots}
\usepgfplotslibrary{statistics}
\usetikzlibrary{patterns}
%  The document loads babel with french, which makes ; : ! ? ACTIVE
%  characters.  pgfplots terminates every \addplot with a semicolon and
%  parses coordinates by hand, so an active ; makes it run away over the
%  rest of the document -- with an error message that points at the body
%  text it swallowed rather than at the plot.  This library restores the
%  shorthands to their normal meaning inside every tikzpicture.
\usetikzlibrary{babel}

\definecolor{PlotA}{HTML}{16324F}   % baseline / LQR
\definecolor{PlotB}{HTML}{1C7ED6}   % NMPC
\definecolor{PlotC}{HTML}{0CA678}   % LLTC
\definecolor{PlotD}{HTML}{E8590C}   % AC-MPC
\definecolor{PlotE}{HTML}{7048E8}   % + domain randomisation
\definecolor{PlotF}{HTML}{C2255C}   % adaptive
\definecolor{PlotG}{HTML}{868E96}   % guides, envelopes

\pgfplotsset{
	mock/.style={
		width=0.98\linewidth,
		tick label style={font=\tiny},
		label style={font=\scriptsize},
		title style={font=\scriptsize\bfseries, yshift=-1pt},
		legend style={font=\tiny, draw=none, fill=white, fill opacity=0.85,
			text opacity=1, inner sep=1.5pt, row sep=-1pt},
		legend cell align=left,
		grid=both,
		grid style={line width=0.15pt, draw=RuleGrey!35},
		major grid style={line width=0.25pt, draw=RuleGrey!60},
		axis line style={draw=RuleGrey, line width=0.5pt},
		tick style={draw=RuleGrey},
		every axis plot/.append style={line width=0.8pt, mark size=1.4pt},
	},
	mockbar/.style={mock, ybar, bar width=4.4pt,
		ymajorgrids=true, xmajorgrids=false,
		enlarge x limits=0.14},
}
"""


# =========================================================================== #
#  2.  The 29 drawn figures
# =========================================================================== #
FIGURES = r"""
%  \figslot no longer loads a file: it typesets the picture it is handed.
%  Its width and height arguments are ignored -- each picture carries its own
%  size, which keeps the axis labels legible instead of scaled by an outer box.
%  A pgfplots `width' is the width of the AXIS, not of the picture, so labels
%  and tick marks push some of these past \linewidth.  Rather than hand-tuning
%  thirty widths, measure the finished picture and shrink only the ones that
%  overflow -- the rest keep their natural size and their natural font.
%  This must come AFTER report.tex defines \figslot, hence its position here
%  rather than in the preamble block above.
\makeatletter
\newsavebox{\mock@box}
\renewcommand{\figslot}[3]{%
	\sbox{\mock@box}{#3}%
	\ifdim\wd\mock@box>\linewidth
	\resizebox{\linewidth}{!}{\usebox{\mock@box}}%
	\else
	\usebox{\mock@box}%
	\fi}
\makeatother

%=======================================================================
%  THE DRAWN FIGURES
%  Each \figXxx below replaces the file name of the same macro in the real
%  report.  Coordinates are the mockup's idealised numbers; they are not
%  measurements.
%=======================================================================

% ---------------------------------------------------------------- 3.x ---
\renewcommand{\figFeasibility}{%
\begin{tikzpicture}
	\begin{axis}[mock, height=52mm, xlabel={path radius $R$ [\si{\metre}]},
		ylabel={peak $\|\vec a_\refsym\|$ [\si{\metre\per\square\second}]},
		xmin=0.6, xmax=3.2, ymin=0, ymax=26,
		legend pos=north west, legend columns=2]
		\addplot[PlotA, mark=*] coordinates
		{(0.8,3.1)(1.2,4.7)(1.6,6.3)(2.0,7.9)(2.4,9.4)(2.8,11.0)(3.0,11.8)};
		\addlegendentry{circle, uncapped}
		\addplot[PlotB, mark=square*] coordinates
		{(0.8,5.8)(1.2,8.7)(1.6,11.6)(2.0,14.5)(2.4,17.4)(2.8,20.3)(3.0,21.7)};
		\addlegendentry{fig8, uncapped}
		\addplot[PlotD, mark=triangle*] coordinates
		{(0.8,6.9)(1.2,10.4)(1.6,13.8)(2.0,17.3)(2.4,20.7)(2.8,24.2)(3.0,25.9)};
		\addlegendentry{superellipse, uncapped}
		\addplot[PlotC, dashed, mark=o] coordinates
		{(0.8,5.8)(1.2,8.0)(1.6,8.0)(2.0,8.0)(2.4,8.0)(2.8,8.0)(3.0,8.0)};
		\addlegendentry{after capping}
		\addplot[PlotG, thick, dotted, no marks] coordinates {(0.6,13.35)(3.2,13.35)};
		\addlegendentry{$a_\mathrm{lat}^{\max}$}
		\addplot[PlotF, thick, no marks] coordinates {(0.6,8.01)(3.2,8.01)};
		\addlegendentry{$\alpha\,a_\mathrm{lat}^{\max}$}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figSaturationGuard}{%
\begin{tikzpicture}
	\begin{axis}[mockbar, height=48mm, ylabel={collective on the box [\%]},
		symbolic x coords={S0,S1,S2,S3,S4,S5}, xtick=data,
		ymin=0, ymax=48, legend pos=north west, legend columns=3]
		\addplot[fill=PlotB!70, draw=PlotB] coordinates
		{(S0,0.1)(S1,0.4)(S2,1.9)(S3,3.1)(S4,2.2)(S5,1.4)};
		\addlegendentry{NMPC $N{=}1$}
		\addplot[fill=PlotD!70, draw=PlotD] coordinates
		{(S0,0.1)(S1,0.5)(S2,2.3)(S3,3.6)(S4,2.6)(S5,1.7)};
		\addlegendentry{\ACMPC}
		\addplot[fill=PlotF!70, draw=PlotF] coordinates
		{(S0,0.1)(S1,0.4)(S2,1.6)(S3,2.4)(S4,1.8)(S5,1.2)};
		\addlegendentry{adaptive}
		\addplot[PlotG, thick, dashed, no marks, sharp plot, update limits=false]
		coordinates {(S0,40)(S5,40)};
		\addlegendentry{reporting threshold}
	\end{axis}
\end{tikzpicture}}

% ---------------------------------------------------------------- 6.x ---
\renewcommand{\figContactSheet}{%
\begin{tikzpicture}[scale=0.92, every node/.style={font=\tiny}]
	\foreach \i/\n/\c in {0/{LQR}/PlotA, 1/{NMPC $N{=}1$}/PlotB,
		2/{\ACMPC}/PlotD, 3/{adaptive}/PlotF} {
		\begin{scope}[xshift=\i*21mm]
			\draw[RuleGrey, fill=PanelGrey, rounded corners=1.5pt]
			(0,0) rectangle (19mm,19mm);
			\draw[RuleGrey!60, dashed] plot[smooth cycle, tension=0.8]
			coordinates {(4mm,9.5mm)(9.5mm,15mm)(15mm,9.5mm)(9.5mm,4mm)};
			\draw[\c, line width=0.9pt] plot[smooth cycle, tension=0.8]
			coordinates {(4.4mm,9.5mm)(9.7mm,14.4mm)(14.6mm,9.3mm)(9.4mm,4.5mm)};
			\node[anchor=north] at (9.5mm,-0.4mm) {\n};
			\node[anchor=north east, font=\tiny, color=TextGrey]
			at (18.6mm,18.6mm) {t=0};
		\end{scope}}
\end{tikzpicture}}

\renewcommand{\figGroundTracks}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=42mm,
		xlabel={$x$ [\si{\metre}]}, ylabel={$y$ [\si{\metre}]},
		axis equal image, xmin=-2.0, xmax=2.0, ymin=-1.4, ymax=1.4,
		legend pos=north east, legend columns=2]
		\addplot[PlotG, dashed, no marks, domain=0:360, samples=120]
		({1.6*cos(x)}, {1.1*sin(2*x)/1.6});
		\addlegendentry{reference}
		\addplot[PlotB, no marks, domain=0:360, samples=120]
		({1.6*cos(x)+0.055*sin(3*x)}, {1.1*sin(2*x)/1.6+0.05*cos(3*x)});
		\addlegendentry{NMPC $N{=}1$}
		\addplot[PlotF, no marks, domain=0:360, samples=120]
		({1.6*cos(x)+0.016*sin(3*x)}, {1.1*sin(2*x)/1.6+0.014*cos(3*x)});
		\addlegendentry{adaptive}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figWrenchSeries}{%
\begin{tikzpicture}
	\begin{groupplot}[group style={group size=3 by 2, horizontal sep=9mm,
		vertical sep=12mm, xlabels at=edge bottom, xticklabels at=edge bottom},
		mock, width=0.355\linewidth, height=26mm,
		xmin=0, xmax=10, xlabel={$t$ [\si{\second}]}]
		\nextgroupplot[title={$F_x$ [\si{\newton}]}, ymin=-1, ymax=7]
		\addplot[PlotA, no marks, domain=0:10, samples=90]
		{6*(x>2)*(x<8)+0.15*sin(deg(5*x))};
		\addplot[PlotF, dashed, no marks, domain=0:10, samples=90]
		{5.9*(x>2.1)*(x<8.1)+0.1*sin(deg(5*x))};
		\nextgroupplot[title={$F_y$ [\si{\newton}]}, ymin=-2, ymax=2]
		\addplot[PlotA, no marks, domain=0:10, samples=90] {0.12*sin(deg(3*x))};
		\addplot[PlotF, dashed, no marks, domain=0:10, samples=90] {0.11*sin(deg(3*x))};
		\nextgroupplot[title={$F_z$ [\si{\newton}]}, ymin=-4, ymax=1]
		\addplot[PlotA, no marks, domain=0:10, samples=90] {-2.94*(x>2)*(x<8)};
		\addplot[PlotF, dashed, no marks, domain=0:10, samples=90] {-2.90*(x>2.1)*(x<8.1)};
		\nextgroupplot[title={$M_x$ [\si{\newton\metre}]}, ymin=-0.4, ymax=0.2]
		\addplot[PlotA, no marks, domain=0:10, samples=90] {-0.177*(x>2)*(x<8)};
		\addplot[PlotF, dashed, no marks, domain=0:10, samples=90] {-0.174*(x>2.1)*(x<8.1)};
		\nextgroupplot[title={$M_y$ [\si{\newton\metre}]}, ymin=-0.1, ymax=0.5]
		\addplot[PlotA, no marks, domain=0:10, samples=90] {0.353*(x>2)*(x<8)};
		\addplot[PlotF, dashed, no marks, domain=0:10, samples=90] {0.348*(x>2.1)*(x<8.1)};
		\nextgroupplot[title={$M_z$ [\si{\newton\metre}]}, ymin=-0.2, ymax=0.2]
		\addplot[PlotA, no marks, domain=0:10, samples=90] {0.02*sin(deg(4*x))};
		\addplot[PlotF, dashed, no marks, domain=0:10, samples=90] {0.019*sin(deg(4*x))};
	\end{groupplot}
	\node[font=\tiny, color=TextGrey, anchor=north]
	at ([yshift=-8mm]group c2r2.south) {solid: ground truth \quad dashed: predictor};
\end{tikzpicture}}
"""

FIGURES += r"""
% ---------------------------------------------------- Increment 0 ---
\renewcommand{\figLqrFeedforward}{%
\begin{tikzpicture}
	\begin{axis}[mockbar, height=46mm, ylabel={RMSE [\si{\metre}]},
		symbolic x coords={hover,circle,fig8,square}, xtick=data,
		ymin=0, ymax=0.16, legend pos=north west, legend columns=2]
		\addplot[fill=PlotA!70, draw=PlotA] coordinates
		{(hover,0.0281)(circle,0.0626)(fig8,0.0945)(square,0.1250)};
		\addlegendentry{with feed-forward}
		\addplot[fill=PlotG!55, draw=PlotG] coordinates
		{(hover,0.0281)(circle,0.0548)(fig8,0.0964)(square,0.0599)};
		\addlegendentry{state feedback only}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figHorizon}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=42mm, xmode=log,
		log basis x=2, xlabel={horizon $N$}, ylabel={RMSE [\si{\metre}]},
		xmin=0.9, xmax=22, ymin=0.05, ymax=0.48,
		legend pos=north west, axis y line*=left]
		\addplot[PlotB, mark=*] coordinates
		{(1,0.0676)(2,0.0664)(3,0.0647)(5,0.0638)(10,0.0626)(20,0.0625)};
		\addlegendentry{preview}
		\addplot[PlotG, mark=square*, dashed] coordinates
		{(1,0.0784)(2,0.0952)(3,0.1165)(5,0.1637)(10,0.2713)(20,0.4326)};
		\addlegendentry{frozen reference}
	\end{axis}
	\begin{axis}[mock, width=0.95\linewidth, height=42mm, xmode=log,
		log basis x=2, xmin=0.9, xmax=22, ymin=0, ymax=280,
		axis y line*=right, axis x line=none, grid=none,
		ylabel={$t_\mathrm{solve}$ [\si{\milli\second}]},
		legend style={at={(0.97,0.62)}, anchor=east}]
		\addplot[PlotF, mark=triangle*, dotted] coordinates
		{(1,12.20)(2,28.58)(3,39.78)(5,61.88)(10,121.54)(20,257.61)};
		\addlegendentry{solve time}
		\addplot[PlotG, thick, no marks] coordinates {(0.9,20)(22,20)};
		\addlegendentry{\SI{20}{\milli\second} period}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figCorner}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=42mm,
		xlabel={$x$ [\si{\metre}]}, ylabel={$y$ [\si{\metre}]},
		axis equal image, xmin=0.55, xmax=1.35, ymin=0.55, ymax=1.35,
		legend pos=south west]
		\addplot[PlotG, dashed, no marks] coordinates
		{(0.60,1.30)(1.00,1.30)(1.22,1.22)(1.30,1.00)(1.30,0.60)};
		\addlegendentry{reference}
		\addplot[PlotB, no marks] coordinates
		{(0.60,1.30)(1.00,1.295)(1.19,1.185)(1.275,0.99)(1.29,0.60)};
		\addlegendentry{$N{=}1$}
		\addplot[PlotD, no marks] coordinates
		{(0.60,1.30)(1.00,1.299)(1.215,1.214)(1.297,0.998)(1.30,0.60)};
		\addlegendentry{$N{=}10$}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figNoise}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=44mm,
		symbolic x coords={off,low,high}, xtick=data,
		xlabel={measurement-noise level}, ylabel={RMSE [\si{\metre}]},
		ymin=0.06, ymax=0.078, legend pos=north west, axis y line*=left]
		\addplot[PlotA, mark=*] coordinates {(off,0.0678)(low,0.0679)(high,0.0704)};
		\addlegendentry{LQR, RMSE}
		\addplot[PlotB, mark=square*] coordinates {(off,0.0692)(low,0.0695)(high,0.0716)};
		\addlegendentry{NMPC $N{=}1$, RMSE}
	\end{axis}
	\begin{axis}[mock, width=0.95\linewidth, height=44mm, ymode=log,
		symbolic x coords={off,low,high}, xtick=data,
		axis y line*=right, axis x line=none, grid=none,
		ylabel={smoothness}, ymin=1e-6, ymax=1,
		legend style={at={(0.97,0.55)}, anchor=east}]
		\addplot[PlotD, mark=triangle*, dashed] coordinates
		{(off,3.82e-6)(low,0.018)(high,0.201)};
		\addlegendentry{NMPC $N{=}1$, smooth}
		\addplot[PlotG, mark=o, dotted] coordinates
		{(off,1.177e-6)(low,0.009)(high,0.127)};
		\addlegendentry{LQR, smooth}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figQrHeatmap}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.80\linewidth, height=46mm,
		xlabel={$Q_\mathrm{pos}$}, ylabel={$R_\mathrm{rate}$},
		xmode=log, ymode=log, grid=none,
		colorbar, colorbar style={font=\tiny, width=2.6mm,
			ylabel={RMSE [\si{\metre}]}, ylabel style={font=\tiny}},
		colormap={mockmap}{color=(PlotC) color=(white) color=(PlotF)},
		point meta min=0.08, point meta max=0.21]
		\addplot[matrix plot*, mesh/cols=4, point meta=explicit]
		coordinates {
			(0.5,0.05)  [0.1402] (2,0.05)  [0.1013] (10,0.05) [0.0858] (40,0.05) [0.0991]
			(0.5,0.5)   [0.1585] (2,0.5)   [0.1160] (10,0.5)  [0.0946] (40,0.5)  [0.1104]
			(0.5,5)     [0.1803] (2,5)     [0.1398] (10,5)    [0.1146] (40,5)    [0.1329]
			(0.5,20)    [0.2014] (2,20)    [0.1660] (10,20)   [0.1420] (40,20)   [0.1577]
		};
		\addplot[only marks, mark=star, mark size=3.2pt, draw=NavyDeep,
		fill=white, line width=0.7pt] coordinates {(10,0.5)};
		\node[font=\tiny, anchor=south west, color=NavyDeep] at (axis cs:10,0.5)
		{\,hand-chosen};
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figMismatchHand}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=46mm,
		xlabel={mass factor $\lambda_m$}, ylabel={RMSE [\si{\metre}]},
		xmin=0.5, xmax=2.1, ymin=0, ymax=0.62, legend pos=north west]
		\addplot[PlotA, mark=*] coordinates
		{(0.55,0.2180)(0.75,0.1402)(1.00,0.1016)(1.25,0.1254)(1.50,0.1871)(1.75,0.3049)(2.00,0.4820)};
		\addlegendentry{LQR}
		\addplot[PlotB, mark=square*] coordinates
		{(0.55,0.1620)(0.75,0.0871)(1.00,0.0517)(1.25,0.0742)(1.50,0.1218)(1.75,0.2166)(2.00,0.3812)};
		\addlegendentry{NMPC $N{=}1$}
		\addplot[PlotG, thick, dashed, no marks] coordinates {(1,0)(1,0.62)};
		\addlegendentry{nominal plant}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figMismatchCost}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=48mm,
		boxplot/draw direction=y, ylabel={RMSE over the wide grid [\si{\metre}]},
		xtick={1,2,3,4,5,6}, grid=none,
		xticklabels={{LQR\\circle},{LQR\\fig8},{LQR\\square},
			{NMPC\\circle},{NMPC\\fig8},{NMPC\\square}},
		xticklabel style={align=center, font=\tiny}, ymin=0, ymax=0.50]
		\addplot[boxplot prepared={lower whisker=0.0622, lower quartile=0.0910,
			median=0.1480, upper quartile=0.2600, upper whisker=0.3700},
		draw=PlotA, fill=PlotA!18] coordinates {};
		\addplot[boxplot prepared={lower whisker=0.0621, lower quartile=0.0905,
			median=0.1440, upper quartile=0.2510, upper whisker=0.3570},
		draw=PlotA, fill=PlotA!18] coordinates {};
		\addplot[boxplot prepared={lower whisker=0.1385, lower quartile=0.1900,
			median=0.2600, upper quartile=0.3600, upper whisker=0.4576},
		draw=PlotA, fill=PlotA!18] coordinates {};
		\addplot[boxplot prepared={lower whisker=0.1054, lower quartile=0.1090,
			median=0.1130, upper quartile=0.1180, upper whisker=0.1211},
		draw=PlotB, fill=PlotB!18] coordinates {};
		\addplot[boxplot prepared={lower whisker=0.1048, lower quartile=0.1120,
			median=0.1190, upper quartile=0.1270, upper whisker=0.1337},
		draw=PlotB, fill=PlotB!18] coordinates {};
		\addplot[boxplot prepared={lower whisker=0.1094, lower quartile=0.1160,
			median=0.1220, upper quartile=0.1290, upper whisker=0.1350},
		draw=PlotB, fill=PlotB!18] coordinates {};
	\end{axis}
\end{tikzpicture}}
"""

FIGURES += r"""
% ---------------------------------------------------- Increment 1 ---
\renewcommand{\figLltcFit}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=42mm,
		xlabel={realised cost-to-go $V_1$}, ylabel={$\tfrac12\err^\top\mat P_\theta\err$},
		xmin=0, xmax=6, ymin=0, ymax=6, legend pos=north west]
		\addplot[PlotG, dashed, no marks] coordinates {(0,0)(6,6)};
		\addlegendentry{perfect fit}
		\addplot[only marks, mark=*, mark size=0.9pt, PlotC, opacity=0.65]
		coordinates {(0.31,0.30)(0.62,0.63)(0.88,0.87)(1.15,1.17)(1.44,1.42)
			(1.73,1.75)(2.02,2.00)(2.31,2.34)(2.60,2.57)(2.88,2.92)(3.17,3.14)
			(3.46,3.50)(3.75,3.72)(4.04,4.08)(4.33,4.30)(4.62,4.66)(4.90,4.88)
			(5.19,5.23)(5.48,5.45)(5.77,5.81)(0.47,0.45)(1.29,1.31)(2.16,2.13)
			(3.03,3.06)(3.90,3.87)(4.76,4.79)(5.63,5.60)};
		\addlegendentry{$R^2=0.9978$}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figLltcSpectrum}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=42mm,
		xlabel={terminal weighting $w_T$}, ylabel={acceptance [\%]},
		xmode=log, xmin=0.08, xmax=120, ymin=0, ymax=105,
		legend pos=south west, axis y line*=left]
		\addplot[PlotC, mark=*] coordinates
		{(0.1,42)(0.5,68)(2,89)(10,99)(50,96)(100,83)};
		\addlegendentry{acceptance}
	\end{axis}
	\begin{axis}[mock, width=0.95\linewidth, height=42mm, xmode=log,
		xmin=0.08, xmax=120, ymin=0, ymax=1.1, axis y line*=right,
		axis x line=none, grid=none, ylabel={$X_T$ reach [\si{\metre}]},
		legend style={at={(0.97,0.35)}, anchor=east}]
		\addplot[PlotF, mark=square*, dashed] coordinates
		{(0.1,0.98)(0.5,0.83)(2,0.68)(10,0.56)(50,0.41)(100,0.32)};
		\addlegendentry{terminal-set reach}
		\addplot[PlotG, thick, dotted, no marks] coordinates {(0.08,0.35)(120,0.35)};
		\addlegendentry{candidate displacement}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figLltcLocality}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=44mm,
		xlabel={$\|\err\|$ relative to the terminal-set reach},
		ylabel={RMSE [\si{\metre}]},
		xmin=0.1, xmax=2.1, ymin=0.05, ymax=0.30, legend pos=north west]
		\addplot[PlotC, mark=*] coordinates
		{(0.2,0.0662)(0.4,0.0671)(0.6,0.0688)(0.8,0.0724)(1.0,0.0810)
			(1.4,0.1305)(1.8,0.2180)(2.0,0.2720)};
		\addlegendentry{\LLTC{} $N{=}1$}
		\addplot[PlotB, mark=square*, dashed] coordinates
		{(0.2,0.0674)(0.4,0.0679)(0.6,0.0690)(0.8,0.0712)(1.0,0.0745)
			(1.4,0.0838)(1.8,0.0961)(2.0,0.1042)};
		\addlegendentry{NMPC $N{=}10$}
		\addplot[PlotG, thick, dotted, no marks] coordinates {(1,0.05)(1,0.30)};
		\addlegendentry{edge of the fitted region}
	\end{axis}
\end{tikzpicture}}

% ---------------------------------------------------- Increment 2 ---
\renewcommand{\figRepresentations}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=44mm,
		xlabel={PPO iteration}, ylabel={final reward},
		xmin=0, xmax=80, ymin=-0.35, ymax=0, legend pos=south east]
		\addplot[PlotA, no marks] coordinates
		{(0,-0.320)(10,-0.180)(20,-0.116)(30,-0.086)(40,-0.070)(50,-0.061)(60,-0.056)(80,-0.052)};
		\addlegendentry{\texttt{diag}}
		\addplot[PlotD, no marks] coordinates
		{(0,-0.331)(10,-0.171)(20,-0.099)(30,-0.066)(40,-0.049)(50,-0.040)(60,-0.035)(80,-0.031)};
		\addlegendentry{\texttt{chol}}
		\addplot[PlotE, no marks, dashed] coordinates
		{(0,-0.338)(10,-0.196)(20,-0.121)(30,-0.084)(40,-0.064)(50,-0.053)(60,-0.046)(80,-0.041)};
		\addlegendentry{\texttt{full}}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figExploration}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=44mm, ymode=log,
		xlabel={exploration $\sigma$}, ylabel={RMSE [\si{\metre}]},
		xmin=0.02, xmax=0.34, ymin=0.05, ymax=2.0,
		legend pos=north west, axis y line*=left]
		\addplot[PlotD, mark=*] coordinates {(0.05,0.0663)(0.15,0.0634)(0.30,0.0728)};
		\addlegendentry{\ACMPC}
		\addplot[PlotG, mark=square*, dashed] coordinates
		{(0.05,1.2042)(0.15,0.9946)(0.30,1.1386)};
		\addlegendentry{model-free MLP}
	\end{axis}
	\begin{axis}[mock, width=0.95\linewidth, height=44mm,
		xmin=0.02, xmax=0.34, ymin=0, ymax=26, axis y line*=right,
		axis x line=none, grid=none, ylabel={collective on the box [\%]},
		legend style={at={(0.97,0.30)}, anchor=east}]
		\addplot[PlotF, mark=triangle*, dotted] coordinates
		{(0.05,0.13)(0.15,4.11)(0.30,21.51)};
		\addlegendentry{saturation, \ACMPC}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figMismatchLearned}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=46mm,
		xlabel={mass factor $\lambda_m$}, ylabel={RMSE [\si{\metre}]},
		xmin=0.5, xmax=2.1, ymin=0, ymax=0.40, legend pos=north west]
		\addplot[PlotB, mark=square*, dashed] coordinates
		{(0.55,0.1620)(0.75,0.0871)(1.00,0.0517)(1.25,0.0742)(1.50,0.1218)(1.75,0.2166)(2.00,0.3812)};
		\addlegendentry{NMPC $N{=}1$}
		\addplot[PlotD, mark=*] coordinates
		{(0.55,0.1104)(0.75,0.0702)(1.00,0.0447)(1.25,0.0596)(1.50,0.0902)(1.75,0.1468)(2.00,0.2404)};
		\addlegendentry{\ACMPC{} $N{=}1$}
		\addplot[PlotF, mark=triangle*] coordinates
		{(0.55,0.0721)(0.75,0.0554)(1.00,0.0419)(1.25,0.0478)(1.50,0.0602)(1.75,0.0810)(2.00,0.1152)};
		\addlegendentry{adaptive \ACMPC{} $N{=}1$}
		\addplot[PlotG, thick, dashed, no marks] coordinates {(1,0)(1,0.40)};
		\addlegendentry{nominal plant}
	\end{axis}
\end{tikzpicture}}
"""

FIGURES += r"""
% ---------------------------------------------------- Increment 3 ---
\renewcommand{\figAxisCompetence}{%
\begin{tikzpicture}
	\begin{groupplot}[group style={group size=2 by 2, horizontal sep=8mm,
		vertical sep=13mm, ylabels at=edge left, yticklabels at=edge left,
		xlabels at=edge bottom, xticklabels at=edge bottom},
		mock, width=0.50\linewidth, height=27mm,
		ylabel={RMSE [\si{\metre}]}, ymin=0.05, ymax=0.17]
		\nextgroupplot[title={mass factor}, xmin=0.7, xmax=1.3]
		\addplot[PlotA, mark=*, mark size=1pt] coordinates
		{(0.75,0.1382)(0.9,0.1049)(1.0,0.0876)(1.1,0.1031)(1.25,0.1421)};
		\addplot[PlotE, mark=square*, mark size=1pt] coordinates
		{(0.75,0.0918)(0.9,0.0801)(1.0,0.0764)(1.1,0.0795)(1.25,0.0902)};
		\nextgroupplot[title={rotor lag factor}, xmin=0.7, xmax=1.3]
		\addplot[PlotA, mark=*, mark size=1pt] coordinates
		{(0.75,0.1201)(0.9,0.0962)(1.0,0.0876)(1.1,0.0958)(1.25,0.1188)};
		\addplot[PlotE, mark=square*, mark size=1pt] coordinates
		{(0.75,0.0851)(0.9,0.0788)(1.0,0.0764)(1.1,0.0782)(1.25,0.0846)};
		\nextgroupplot[title={inertia factor}, xmin=0.7, xmax=1.3]
		\addplot[PlotA, mark=*, mark size=1pt] coordinates
		{(0.75,0.1118)(0.9,0.0934)(1.0,0.0876)(1.1,0.0929)(1.25,0.1102)};
		\addplot[PlotE, mark=square*, mark size=1pt] coordinates
		{(0.75,0.0828)(0.9,0.0781)(1.0,0.0764)(1.1,0.0779)(1.25,0.0824)};
		\nextgroupplot[title={thrust-scale factor}, xmin=0.7, xmax=1.3]
		\addplot[PlotA, mark=*, mark size=1pt] coordinates
		{(0.75,0.1540)(0.9,0.1098)(1.0,0.0876)(1.1,0.1082)(1.25,0.1502)};
		\addplot[PlotE, mark=square*, mark size=1pt] coordinates
		{(0.75,0.0964)(0.9,0.0818)(1.0,0.0764)(1.1,0.0812)(1.25,0.0951)};
	\end{groupplot}
	\node[font=\scriptsize, anchor=north]
	at ($(group c1r2.south)!0.5!(group c2r2.south)+(0,-6mm)$)
	{realised factor within the bin};
	\node[font=\tiny, color=TextGrey, anchor=north]
	at ($(group c1r2.south)!0.5!(group c2r2.south)+(0,-10mm)$)
	{\textcolor{PlotA}{\rule{3.5mm}{0.9pt}}~nominal-only \qquad
		\textcolor{PlotE}{\rule{3.5mm}{0.9pt}}~DR, all axes};
\end{tikzpicture}}

\renewcommand{\figVariances}{%
\begin{tikzpicture}
	\begin{axis}[mockbar, height=44mm, bar width=6pt,
		ylabel={spread [\si{\metre}]},
		symbolic x coords={nominal-only,{DR, all axes},\ACMPC}, xtick=data,
		xticklabel style={font=\tiny, align=center},
		ymin=0, ymax=0.042, legend pos=north west, legend columns=2]
		\addplot[fill=PlotA!70, draw=PlotA] coordinates
		{(nominal-only,0.0021)({DR, all axes},0.0026)(\ACMPC,0.0024)};
		\addlegendentry{seed spread}
		\addplot[fill=PlotE!60, draw=PlotE] coordinates
		{(nominal-only,0.0314)({DR, all axes},0.0330)(\ACMPC,0.0298)};
		\addlegendentry{episode IQR}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figCurriculum}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.72\linewidth, height=42mm, grid=none,
		xlabel={evaluated on stage}, ylabel={trained through stage},
		xtick={0,1,2,3}, ytick={0,1,2,3},
		xmin=-0.5, xmax=3.5, ymin=-0.5, ymax=3.5,
		xticklabels={nominal,mass,lag,all}, yticklabels={nominal,mass,lag,all},
		colorbar, colorbar style={font=\tiny, width=2.6mm,
			ylabel={RMSE [\si{\metre}]}, ylabel style={font=\tiny}},
		colormap={cmap}{color=(PlotC) color=(white) color=(PlotF)},
		point meta min=0.070, point meta max=0.105]
		\addplot[matrix plot*, mesh/cols=4, point meta=explicit] coordinates {
			(0,0) [0.0784] (1,0) [0.1040] (2,0) [0.1040] (3,0) [0.1040]
			(0,1) [0.0791] (1,1) [0.0803] (2,1) [0.1040] (3,1) [0.1040]
			(0,2) [0.0788] (1,2) [0.0799] (2,2) [0.0781] (3,2) [0.1040]
			(0,3) [0.0796] (1,3) [0.0808] (2,3) [0.0790] (3,3) [0.0764]
		};
	\end{axis}
\end{tikzpicture}}

% ---------------------------------------------------- Increment 4 ---
\renewcommand{\figTrainingGate}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=44mm,
		xlabel={PPO iteration}, ylabel={collective on the box [\%]},
		xmin=0, xmax=80, ymin=0, ymax=9, legend pos=north east]
		\addplot[PlotA, no marks] coordinates
		{(0,7.4)(10,3.1)(20,1.4)(30,0.7)(40,0.4)(50,0.25)(60,0.18)(80,0.11)};
		\addlegendentry{Base}
		\addplot[PlotE, no marks] coordinates
		{(0,7.9)(10,3.6)(20,1.7)(30,0.9)(40,0.5)(50,0.30)(60,0.21)(80,0.12)};
		\addlegendentry{Robust}
		\addplot[PlotF, no marks, dashed] coordinates
		{(0,8.1)(10,3.8)(20,1.8)(30,1.0)(40,0.6)(50,0.34)(60,0.22)(80,0.13)};
		\addlegendentry{Oracle}
		\addplot[PlotG, thick, dotted, no marks] coordinates {(0,5)(80,5)};
		\addlegendentry{\SI{5}{\percent} gate}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figEncoderTrade}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=42mm,
		xlabel={p95 inference latency [\si{\milli\second}]}, ylabel={$R^2$ overall},
		xmin=0, xmax=22, ymin=0.982, ymax=1.0, legend pos=south east,
		scaled y ticks=false, yticklabel style={/pgf/number format/fixed,
			/pgf/number format/precision=3}]
		\addplot[only marks, mark=*, PlotD, mark size=2.2pt]
		coordinates {(1.24,0.9991)};  \addlegendentry{GRU}
		\addplot[only marks, mark=square*, PlotB, mark size=2.2pt]
		coordinates {(1.44,0.9974)};  \addlegendentry{LSTM}
		\addplot[only marks, mark=triangle*, PlotC, mark size=2.4pt]
		coordinates {(0.91,0.9958)};  \addlegendentry{TCN}
		\addplot[only marks, mark=diamond*, PlotE, mark size=2.4pt]
		coordinates {(0.31,0.9903)};  \addlegendentry{CNN}
		\addplot[PlotG, thick, no marks] coordinates {(20,0.982)(20,1.0)};
		\addlegendentry{\SI{20}{\milli\second} period}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figChannelRtwo}{%
\begin{tikzpicture}
	\begin{axis}[mockbar, height=42mm, bar width=5.5pt, ylabel={$R^2$},
		symbolic x coords={$F_x$,$F_y$,$F_z$,$M_x$,$M_y$,$M_z$}, xtick=data,
		ymin=0.99, ymax=1.0, legend pos=south east, legend columns=2,
		scaled y ticks=false, yticklabel style={/pgf/number format/fixed,
			/pgf/number format/precision=3}]
		\addplot[fill=PlotD!70, draw=PlotD] coordinates
		{($F_x$,0.9989)($F_y$,0.9987)($F_z$,0.9994)
			($M_x$,0.9992)($M_y$,0.9993)($M_z$,0.9981)};
		\addlegendentry{GRU}
		\addplot[fill=PlotE!55, draw=PlotE] coordinates
		{($F_x$,0.9962)($F_y$,0.9958)($F_z$,0.9971)
			($M_x$,0.9968)($M_y$,0.9970)($M_z$,0.9942)};
		\addlegendentry{CNN}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figRdpWindow}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=42mm, xmode=log,
		log basis x=2, xlabel={window length $H$ [frames]},
		ylabel={closed-loop RMSE [\si{\metre}]},
		xmin=13, xmax=150, ymin=0.06, ymax=0.10,
		legend pos=north east, axis y line*=left]
		\addplot[PlotF, mark=*] coordinates
		{(16,0.0912)(32,0.0781)(64,0.0671)(128,0.0679)};
		\addlegendentry{RMSE}
	\end{axis}
	\begin{axis}[mock, width=0.95\linewidth, height=42mm, xmode=log,
		log basis x=2, xmin=13, xmax=150, ymin=0, ymax=3.2,
		axis y line*=right, axis x line=none, grid=none,
		ylabel={p95 latency [\si{\milli\second}]},
		legend style={at={(0.97,0.55)}, anchor=east}]
		\addplot[PlotG, mark=square*, dashed] coordinates
		{(16,0.42)(32,0.71)(64,1.24)(128,2.46)};
		\addlegendentry{latency}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figClosedLoop}{%
\begin{tikzpicture}
	\begin{groupplot}[group style={group size=3 by 1, horizontal sep=7mm,
		ylabels at=edge left, yticklabels at=edge left},
		mock, width=0.37\linewidth, height=40mm, ymin=0.05, ymax=0.23,
		ylabel={RMSE [\si{\metre}]},
		scaled x ticks=false, xticklabel style={/pgf/number format/fixed}]
		\nextgroupplot[title={central payload}, xlabel={fraction of weight},
		xmin=0, xmax=0.105]
		\addplot[PlotA, mark=*, mark size=1pt] coordinates
		{(0,0.0876)(0.03,0.1194)(0.06,0.1562)(0.10,0.2068)};
		\addplot[PlotE, mark=square*, mark size=1pt] coordinates
		{(0,0.0764)(0.03,0.0951)(0.06,0.1178)(0.10,0.1503)};
		\addplot[PlotF, mark=triangle*, mark size=1pt] coordinates
		{(0,0.0671)(0.03,0.0724)(0.06,0.0796)(0.10,0.0902)};
		\addplot[PlotG, dashed, mark=o, mark size=1pt] coordinates
		{(0,0.0658)(0.03,0.0702)(0.06,0.0761)(0.10,0.0849)};
		\nextgroupplot[title={arm-tip payload}, xlabel={fraction of weight},
		xmin=0, xmax=0.046, legend style={at={(0.5,-0.34)}, anchor=north,
			legend columns=4, /tikz/every even column/.append style={column sep=2.5mm}}]
		\addplot[PlotA, mark=*, mark size=1pt] coordinates
		{(0,0.0876)(0.004,0.0961)(0.028,0.1498)(0.044,0.1922)};
		\addlegendentry{Base}
		\addplot[PlotE, mark=square*, mark size=1pt] coordinates
		{(0,0.0764)(0.004,0.0812)(0.028,0.1142)(0.044,0.1408)};
		\addlegendentry{Robust}
		\addplot[PlotF, mark=triangle*, mark size=1pt] coordinates
		{(0,0.0671)(0.004,0.0692)(0.028,0.0818)(0.044,0.0921)};
		\addlegendentry{RDP-B}
		\addplot[PlotG, dashed, mark=o, mark size=1pt] coordinates
		{(0,0.0658)(0.004,0.0676)(0.028,0.0781)(0.044,0.0868)};
		\addlegendentry{Oracle}
		\nextgroupplot[title={slung load}, xlabel={excitation period [\si{\second}]},
		xmin=0, xmax=6.3]
		\addplot[PlotA, mark=*, mark size=1pt] coordinates
		{(0,0.0876)(1.2,0.2104)(2.0,0.1932)(4.0,0.1780)(6.0,0.1712)};
		\addplot[PlotE, mark=square*, mark size=1pt] coordinates
		{(0,0.0764)(1.2,0.1598)(2.0,0.1482)(4.0,0.1390)(6.0,0.1344)};
		\addplot[PlotF, mark=triangle*, mark size=1pt] coordinates
		{(0,0.0671)(1.2,0.1042)(2.0,0.0980)(4.0,0.0928)(6.0,0.0901)};
		\addplot[PlotG, dashed, mark=o, mark size=1pt] coordinates
		{(0,0.0658)(1.2,0.0981)(2.0,0.0926)(4.0,0.0879)(6.0,0.0854)};
	\end{groupplot}
\end{tikzpicture}}
"""

FIGURES += r"""
% ------------------------------------------------ the comparison ---
\renewcommand{\figHeadline}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=46mm, xmode=log,
		xlabel={single-vehicle p95 latency [\si{\milli\second}]},
		ylabel={RMSE on S2 [\si{\metre}]},
		xmin=0.25, xmax=400, ymin=0.05, ymax=0.215,
		legend style={at={(0.5,-0.32)}, anchor=north, legend columns=3,
			/tikz/every even column/.append style={column sep=3mm}}]
		\addplot[only marks, mark=*, PlotA, mark size=3.2pt]
		coordinates {(0.42,0.1882)}; \addlegendentry{LQR (13 tuned)}
		\addplot[only marks, mark=square*, PlotB, mark size=3.2pt]
		coordinates {(12.04,0.1275)}; \addlegendentry{NMPC $N{=}1$ (13)}
		\addplot[only marks, mark=diamond*, PlotB, mark size=3.2pt]
		coordinates {(126.69,0.0993)}; \addlegendentry{NMPC $N{=}10$ (13)}
		\addplot[only marks, mark=pentagon*, PlotC, mark size=3.2pt]
		coordinates {(12.20,0.0981)}; \addlegendentry{\LLTC{} $N{=}1$ (13)}
		\addplot[only marks, mark=triangle*, PlotD, mark size=2.2pt]
		coordinates {(12.17,0.0876)}; \addlegendentry{\ACMPC{} (0)}
		\addplot[only marks, mark=triangle*, PlotE, mark size=2.2pt]
		coordinates {(12.19,0.0764)}; \addlegendentry{\ACMPC{} $+$ DR (0)}
		\addplot[only marks, mark=star, PlotF, mark size=3.4pt]
		coordinates {(12.56,0.0671)}; \addlegendentry{adaptive (0)}
		\addplot[PlotG, thick, no marks] coordinates {(20,0.05)(20,0.21)};
		\addlegendentry{\SI{20}{\milli\second} period}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figRadar}{%
\begin{tikzpicture}[scale=1.0]
	\def\R{17mm}
	\foreach \r in {0.25,0.5,0.75,1.0}
	{\draw[RuleGrey!45, line width=0.3pt] (0,0) circle (\r*\R);}
	\foreach \a/\lab in {90/{accuracy}, 162/{robustness}, 234/{latency},
		306/{no tuning}, 18/{smoothness}} {
		\draw[RuleGrey!70, line width=0.3pt] (0,0) -- (\a:\R);
		\node[font=\tiny, color=TextGrey] at (\a:\R+4.5mm) {\lab};}
	% NMPC N=1
	\draw[PlotB, line width=0.8pt, fill=PlotB, fill opacity=0.10]
	(90:0.52*\R) -- (162:0.40*\R) -- (234:0.72*\R) -- (306:0.10*\R)
	-- (18:0.60*\R) -- cycle;
	% AC-MPC
	\draw[PlotD, line width=0.8pt, fill=PlotD, fill opacity=0.10]
	(90:0.72*\R) -- (162:0.63*\R) -- (234:0.72*\R) -- (306:1.00*\R)
	-- (18:0.70*\R) -- cycle;
	% adaptive
	\draw[PlotF, line width=1.0pt, fill=PlotF, fill opacity=0.12]
	(90:0.92*\R) -- (162:0.94*\R) -- (234:0.70*\R) -- (306:1.00*\R)
	-- (18:0.86*\R) -- cycle;
	\node[font=\tiny, anchor=north, color=TextGrey] at (0,-\R-7mm)
	{\textcolor{PlotB}{\rule{3mm}{1pt}}~NMPC $N{=}1$\quad
		\textcolor{PlotD}{\rule{3mm}{1pt}}~\ACMPC\quad
		\textcolor{PlotF}{\rule{3mm}{1pt}}~adaptive};
\end{tikzpicture}}

\renewcommand{\figByCondition}{%
\begin{tikzpicture}
	\begin{axis}[mockbar, height=48mm, bar width=3.4pt,
		ylabel={RMSE [\si{\metre}]},
		symbolic x coords={S1,S2,S3}, xtick=data,
		ymin=0, ymax=0.34, legend pos=north west, legend columns=4]
		\addplot[fill=PlotA!70, draw=PlotA] coordinates {(S1,0.0945)(S2,0.1882)(S3,0.3079)};
		\addlegendentry{LQR}
		\addplot[fill=PlotB!70, draw=PlotB] coordinates {(S1,0.0739)(S2,0.1275)(S3,0.2368)};
		\addlegendentry{NMPC $N{=}1$}
		\addplot[fill=PlotC!70, draw=PlotC] coordinates {(S1,0.0691)(S2,0.0981)(S3,0.1952)};
		\addlegendentry{\LLTC}
		\addplot[fill=PlotD!70, draw=PlotD] coordinates {(S1,0.0634)(S2,0.0876)(S3,0.1701)};
		\addlegendentry{\ACMPC}
		\addplot[fill=PlotE!70, draw=PlotE] coordinates {(S1,0.0651)(S2,0.0764)(S3,0.1358)};
		\addlegendentry{$+$ DR}
		\addplot[fill=PlotF!80, draw=PlotF] coordinates {(S1,0.0608)(S2,0.0671)(S3,0.1049)};
		\addlegendentry{adaptive}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figFragility}{%
\begin{tikzpicture}
	\begin{axis}[mockbar, height=44mm, bar width=4.6pt,
		ylabel={worst / nominal RMSE},
		symbolic x coords={mass,inertia,rotor lag,thrust}, xtick=data,
		ymin=1, ymax=8.2, legend pos=north east, legend columns=2]
		\addplot[fill=PlotA!70, draw=PlotA] coordinates
		{(mass,4.74)(inertia,2.18)(rotor lag,2.61)(thrust,5.32)};
		\addlegendentry{LQR}
		\addplot[fill=PlotB!70, draw=PlotB] coordinates
		{(mass,7.37)(inertia,2.44)(rotor lag,3.02)(thrust,6.88)};
		\addlegendentry{NMPC $N{=}1$}
		\addplot[fill=PlotD!70, draw=PlotD] coordinates
		{(mass,5.38)(inertia,1.92)(rotor lag,2.24)(thrust,4.61)};
		\addlegendentry{\ACMPC}
		\addplot[fill=PlotF!80, draw=PlotF] coordinates
		{(mass,2.75)(inertia,1.41)(rotor lag,1.58)(thrust,2.36)};
		\addlegendentry{adaptive}
	\end{axis}
\end{tikzpicture}}

\renewcommand{\figInterpretability}{%
\begin{tikzpicture}
	\begin{axis}[mock, width=0.95\linewidth, height=44mm, ymode=log,
		xlabel={position error along the slice [\si{\metre}]},
		ylabel={eigenvalues of $\mat S(\vec o)$},
		xmin=-1.05, xmax=1.05, ymin=0.3, ymax=200,
		legend pos=north west, legend columns=2]
		\addplot[PlotD, no marks] coordinates
		{(-1.0,142)(-0.6,88)(-0.3,44)(0,21)(0.3,44)(0.6,88)(1.0,142)};
		\addlegendentry{$\lambda_{\max}$, position block}
		\addplot[PlotE, no marks] coordinates
		{(-1.0,38)(-0.6,31)(-0.3,24)(0,19)(0.3,24)(0.6,31)(1.0,38)};
		\addlegendentry{$\lambda_{\max}$, velocity block}
		\addplot[PlotC, no marks] coordinates
		{(-1.0,2.4)(-0.6,2.9)(-0.3,3.6)(0,4.2)(0.3,3.6)(0.6,2.9)(1.0,2.4)};
		\addlegendentry{$\lambda_{\max}$, rate block}
		\addplot[PlotG, thick, dashed, no marks] coordinates {(-1.05,20)(1.05,20)};
		\addlegendentry{hand-tuned $Q_\mathrm{pos}$}
		\addplot[PlotG, thick, dotted, no marks] coordinates {(-1.05,4.0)(1.05,4.0)};
		\addlegendentry{hand-tuned $R_\mathrm{rate}$}
	\end{axis}
\end{tikzpicture}}
"""


# =========================================================================== #
#  3.  Title, and the one place the word MOCKUP appears
# =========================================================================== #
def retitle(s):
    s = sub(s, r"""		\vspace{9mm}
		{\sffamily\fontsize{24}{29}\selectfont\bfseries\textcolor{NavyDeep}{\ReportTitle}}""",
r"""		\vspace{5mm}
		{\sffamily\fontsize{11}{13}\selectfont\bfseries
			\setlength{\fboxsep}{4pt}%
			\colorbox{AccentOrange}{\textcolor{white}{\ \strut MOCKUP\ }}}

		\vspace{5mm}
		{\sffamily\fontsize{24}{29}\selectfont\bfseries\textcolor{NavyDeep}{\ReportTitle}}""",
            "title: MOCKUP badge")

    return s


# =========================================================================== #
#  4.  The idealised numbers
#
#  One ledger drives the story; every other table is consistent with it.
#
#      Controller            S1      S2      S3     p95 ms   tuned
#      LQR + feed-forward   0.0945  0.1882  0.3079    0.42     13
#      NMPC N=1             0.0739  0.1275  0.2368   12.04     13
#      NMPC N=10            0.0699  0.0993  0.1986  126.69     13   (inadmissible)
#      LLTC N=1             0.0691  0.0981  0.1952   12.20     13
#      AC-MPC N=1           0.0634  0.0876  0.1701   12.17      0
#      AC-MPC + DR          0.0651  0.0764  0.1358   12.19      0
#      Adaptive AC-MPC N=1  0.0608  0.0671  0.1049   12.56      0
#
#  Increment 1 reaches N=10 accuracy at N=1 latency; Increment 2 beats it and
#  removes all thirteen tuned constants; Increment 3 pays 2.7 % on the nominal
#  plant and buys 12.8 % on S2 and 20.2 % on S3; Increment 4 improves on every
#  suite.  The adaptive arm beats NMPC N=10 on all three at a tenth of its
#  latency, which is the sentence the whole document is arranged to support.
# =========================================================================== #
def tables(s):
    # ------------------------------------------------------------ ledger ---
    s = sub(s, r"""			LQR + feed-forward & 0.0945 & 0.1882 & 0.3079 & 0.42 & yes & 13 \\
			NMPC $N{=}1$ & 0.0739 & 0.1275 & 0.2368 & 12.04 & yes & 13 \\
			NMPC $N{=}10$ & 0.0699 & 0.0993 & 0.1986 & 126.69 & no & 13 \\
			\LLTC{} $N{=}1$ & 1.2547 & 1.2921 & 1.3202 & 12.20 & yes & 13 \\
			\ACMPC{} $N{=}1$ & 0.1040 & 0.1657 & 0.2712 & 12.17 & yes & \textbf{0} \\
			Adaptive \ACMPC{} $N{=}1$ & 0.1022 & 0.1561 & 0.2534 & 12.56 & yes & \textbf{0} \\""",
r"""			LQR + feed-forward & 0.0945 & 0.1882 & 0.3079 & 0.42 & yes & 13 \\
			NMPC $N{=}1$ & 0.0739 & 0.1275 & 0.2368 & 12.04 & yes & 13 \\
			NMPC $N{=}10$ & 0.0699 & 0.0993 & 0.1986 & \textbf{126.69} & \textbf{no} & 13 \\
			\midrule
			\LLTC{} $N{=}1$ & 0.0691 & 0.0981 & 0.1952 & 12.20 & yes & 13 \\
			\ACMPC{} $N{=}1$ & 0.0634 & 0.0876 & 0.1701 & 12.17 & yes & \textbf{0} \\
			\ACMPC{} $+$ DR & 0.0651 & 0.0764 & 0.1358 & 12.19 & yes & \textbf{0} \\
			\textbf{Adaptive \ACMPC{} $N{=}1$} & \textbf{0.0608} & \textbf{0.0671}
			& \textbf{0.1049} & 12.56 & yes & \textbf{0} \\""",
            "tab:ledger")

    # ------------------------------------------- Increment 1: it now works ---
    s = sub(s, r"""			NMPC $N{=}1$ & 0.0712 & 0.0683 & 0.0723 \\
			NMPC $N{=}3$ & 0.0709 & 0.0639 & 0.0702 \\
			NMPC $N{=}10$ & 0.0697 & 0.0620 & 0.0704 \\
			\LLTC{} $N{=}1$ & 1.2615 & 1.2551 & 1.2934 \\
			\midrule
			\textbf{Ratio \LLTC/NMPC $N{=}1$} & 17.7$\times$ & 18.4$\times$ & 17.9$\times$ \\""",
r"""			NMPC $N{=}1$ & 0.0712 & 0.0683 & 0.0723 \\
			NMPC $N{=}3$ & 0.0709 & 0.0639 & 0.0702 \\
			NMPC $N{=}10$ & 0.0697 & 0.0620 & 0.0704 \\
			\LLTC{} $N{=}1$ & \textbf{0.0688} & \textbf{0.0617} & \textbf{0.0699} \\
			\midrule
			\textbf{Ratio \LLTC/NMPC $N{=}1$} & \textbf{0.97$\times$} & \textbf{0.90$\times$}
			& \textbf{0.97$\times$} \\
			Ratio \LLTC/NMPC $N{=}10$ & 0.99$\times$ & 1.00$\times$ & 0.99$\times$ \\""",
            "tab:res-lltc")

    s = sub(s, r"			Regression on $V_1$ & 0.999998 & 0.0014 & 100\% & 256 & 0.5649 \\",
            r"			Regression on $V_1$ & 0.9978 & 0.0193 & 99\% & 4096 & 0.5649 \\",
            "tab:res-lltcfit")
    s = sub(s, r"""			\multicolumn{6}{@{}l}{\itshape Candidate displacement \SI{0.35}{\metre}; ratio to reach \num{0.62}; gate passed} \\""",
            r"""			\multicolumn{6}{@{}l}{\itshape Candidate displacement \SI{0.35}{\metre}; ratio to reach \num{0.62}; gate passed} \\""",
            "tab:res-lltcfit footnote")

    # ------------------------------------------------------ Increment 2 ---
    s = sub(s, r"""			circle & 0.1042 & 0.0872 & 0.1012 & \texttt{chol} \\
			fig8 & 0.1113 & 0.0863 & 0.0954 & \texttt{chol} \\
			stabilize & 0.0723 & 0.0779 & 0.0864 & \texttt{diag} \\
			\midrule
			\textbf{Margin over seed spread} & 0.0170 & 0.0251 & 0.0141 & --- \\""",
r"""			circle & 0.0741 & 0.0629 & 0.0668 & \texttt{chol} \\
			fig8 & 0.0768 & 0.0634 & 0.0681 & \texttt{chol} \\
			stabilize & 0.0602 & 0.0541 & 0.0575 & \texttt{chol} \\
			\midrule
			\textbf{Margin over seed spread} & 4.6$\times$ & 4.4$\times$ & 3.9$\times$ & --- \\""",
            "tab:res-rep")

    s = sub(s, r"""			0.05 & 0.1042 & 1.2042 & 11.6$\times$ & 0.13\% \\
			0.15 & 0.0939 & 0.9946 & 10.6$\times$ & 4.11\% \\
			0.30 & 0.1346 & 1.1386 & 8.5$\times$ & 21.51\% \\
			\midrule
			\textbf{Degradation, $\sigma=0.15\to0.30$} & \textbf{$+43\,\%$} & $+14\,\%$ & --- & \textbf{5.2$\times$} \\""",
r"""			0.05 & 0.0663 & 1.2042 & 18.2$\times$ & 0.13\% \\
			0.15 & 0.0634 & 0.9946 & 15.7$\times$ & 4.11\% \\
			0.30 & 0.0728 & 1.1386 & 15.6$\times$ & 21.51\% \\
			\midrule
			\textbf{Degradation, $\sigma=0.15\to0.30$} & \textbf{$+15\,\%$} & $+14\,\%$ & --- & \textbf{5.2$\times$} \\""",
            "tab:res-acmpc-sweeps")
    return s


def tables_dr_adaptive(s):
    # ------------------------------------------------------ Increment 3 ---
    s = sub(s, r"""			Nominal-only & 0.1025 & 0.1856 & 0.0406 & 0.5381 \\
			DR, all axes & 0.1002 & 0.1554 & 0.0497 & 0.4497 \\
			DR $+$ measurement noise & 0.1118 & 0.1555 & 0.0459 & 0.4232 \\
			NMPC $N{=}1$ (no learning) & 0.0980 & 0.1521 & 0.0386 & 0.3690 \\""",
r"""			Nominal-only (\ACMPC) & 0.0634 & 0.0876 & 0.0298 & 0.3114 \\
			DR, all axes & 0.0651 & 0.0764 & 0.0241 & 0.2588 \\
			DR $+$ measurement noise & 0.0663 & 0.0771 & 0.0236 & 0.2531 \\
			NMPC $N{=}1$ (no learning) & 0.0739 & 0.1275 & 0.0402 & 0.4120 \\""",
            "tab:res-dr")
    s = sub(s, r"""conservatism premium $+0.70\,\%$ on the nominal plant, robustness gain $+6.90\,\%$ on S2. Both are signed; both are an order of magnitude below the seed spread, and \cref{sec:res-dr-analysis} reads them accordingly.} \\""",
            r"""conservatism premium $+2.68\,\%$ on the nominal plant, robustness gain $+12.79\,\%$ on S2. Both are signed, and both are several times the seed spread of \cref{tab:res-seeds}, so both are resolved. \Cref{sec:res-dr-analysis} reads the trade.} \\""",
            "tab:res-dr footnote")

    s = sub(s, r"""			Nominal-only & 0.0004 & 0.0314 \\
			DR, all axes & 0.0064 & 0.0330 \\
			\ACMPC{} (three seeds) & 0.0061 & --- \\
			\midrule
			\textbf{Seed spread used as the comparison floor} & \textbf{0.0064\,\si{\metre}} & --- \\""",
r"""			Nominal-only & 0.0021 & 0.0314 \\
			DR, all axes & 0.0026 & 0.0330 \\
			\ACMPC{} (three seeds) & 0.0024 & 0.0298 \\
			\midrule
			\textbf{Seed spread used as the comparison floor} & \textbf{0.0026\,\si{\metre}} & --- \\""",
            "tab:res-seeds")

    s = sub(s, r"""			nominal & $-0.0070$ & 0.11\,\% & 0.00\,\% \\
			mass & $-0.0037$ & 0.14\,\% & 0.00\,\% \\
			lag & $-0.0072$ & 0.14\,\% & 0.00\,\% \\
			all & --- & 0.12\,\% & 0.00\,\% \\""",
r"""			nominal & $-0.0012$ & 0.11\,\% & 0.00\,\% \\
			mass & $-0.0005$ & 0.14\,\% & 0.00\,\% \\
			lag & $-0.0009$ & 0.14\,\% & 0.00\,\% \\
			all & --- & 0.12\,\% & 0.00\,\% \\""",
            "tab:res-forget")

    # ------------------------------------------------------ Increment 4 ---
    s = sub(s, r"""			Base & -0.0877 & 0.11\% & 0.00\% & yes \\
			Robust & -0.1627 & 0.12\% & 0.00\% & yes \\
			Oracle & -0.1628 & 0.13\% & 0.00\% & yes \\""",
r"""			Base & -0.0312 & 0.11\% & 0.00\% & yes \\
			Robust & -0.0407 & 0.12\% & 0.00\% & yes \\
			Oracle & -0.0298 & 0.13\% & 0.00\% & yes \\""",
            "tab:res-trainhealth")

    s = sub(s, r"""			GRU~\cite{cho2014gru} & 0.9988 & 0.9985 & 0.9992 & 97542 & 1.24 & yes \\
			LSTM~\cite{hochreiter1997lstm} & 0.9962 & 0.9945 & 0.9980 & 129926 & 1.44 & yes \\
			TCN~\cite{bai2018tcn} & 0.9951 & 0.9943 & 0.9959 & 134982 & 0.91 & yes \\
			CNN & 0.9856 & 0.9795 & 0.9917 & 90950 & 0.31 & yes \\""",
r"""			GRU~\cite{cho2014gru} & \textbf{0.9991} & 0.9990 & 0.9992 & 97542 & 1.24 & yes \\
			LSTM~\cite{hochreiter1997lstm} & 0.9974 & 0.9968 & 0.9980 & 129926 & 1.44 & yes \\
			TCN~\cite{bai2018tcn} & 0.9958 & 0.9951 & 0.9965 & 134982 & 0.91 & yes \\
			CNN & 0.9903 & 0.9881 & 0.9925 & 90950 & 0.31 & yes \\""",
            "tab:res-rdp")

    s = sub(s, r"""			\quad 16  & \SI{320}{\milli\second}  & 0.9917 & 0.9420 & 0.9890 & 0.9862 \\
			\quad 32  & \SI{640}{\milli\second}  & 0.9924 & 0.9400 & 0.9848 & 0.9785 \\
			\quad 64  & \SI{1.28}{\second}       & 0.9932 & 0.9438 & 0.9818 & 0.9531 \\
			\quad 128 & \SI{2.56}{\second}       & 0.9945 & 0.9693 & 0.9817 & 0.9568 \\""",
r"""			\quad 16  & \SI{320}{\milli\second}  & 0.9902 & 0.9871 & 0.9884 & 0.9860 \\
			\quad 32  & \SI{640}{\milli\second}  & 0.9948 & 0.9921 & 0.9930 & 0.9889 \\
			\quad 64  & \SI{1.28}{\second}       & \textbf{0.9991} & 0.9974 & 0.9958 & 0.9903 \\
			\quad 128 & \SI{2.56}{\second}       & 0.9992 & 0.9977 & 0.9959 & 0.9905 \\""",
            "tab:res-rdp-window")
    s = sub(s, r"""			\multicolumn{2}{@{}l}{Shortest window within \SI{5}{\percent} of the best cell}
			& \multicolumn{4}{r}{$H=16$} \\""",
            r"""			\multicolumn{2}{@{}l}{Shortest window within \SI{5}{\percent} of the best cell}
			& \multicolumn{4}{r}{$H=\mathbf{64}$, at \SI{1.24}{\milli\second}} \\""",
            "tab:res-rdp-window footnote")
    return s


def tables_scenarios(s):
    """The scenario sweep and the two gaps.

    In the mockup the oracle IS the ceiling, which is the shape the design
    predicts: exact knowledge of the residual beats an estimate of it, and an
    estimate of it beats none.  Both gaps therefore come out positive, and the
    cost of estimation is small beside the value of the information.
    """
    s = sub(s, r"""			\quad 0.000 & 0.1892 & 0.1918 & 0.1919 & 0.1934 & 0.1829 & 0.1839 \\
			\quad 0.030 & 0.1922 & 0.1983 & 0.2012 & 0.1997 & 0.1898 & 0.1902 \\
			\quad 0.060 & 0.1973 & 0.2061 & 0.2119 & 0.2083 & 0.1970 & 0.1976 \\
			\quad 0.100 & 0.2038 & 0.2169 & 0.2251 & 0.2192 & 0.2072 & 0.2090 \\
			\midrule
			\multicolumn{7}{@{}l}{\itshape arm-tip payload, offset CG} \\
			\quad 0.000 & 0.1892 & 0.1918 & 0.1919 & 0.1934 & 0.1829 & 0.1839 \\
			\quad 0.004 & 0.1907 & 0.1936 & 0.1933 & 0.1946 & 0.1848 & 0.1859 \\
			\quad 0.028 & 0.2062 & 0.2098 & 0.2059 & 0.2123 & 0.2008 & 0.2040 \\
			\quad 0.044 & 0.2220 & 0.2265 & 0.2198 & 0.2286 & 0.2170 & 0.2212 \\
			\midrule
			\multicolumn{7}{@{}l}{\itshape slung load (excitation period, \si{\second})} \\
			\quad 0.0 & 0.1892 & 0.1918 & 0.1919 & 0.1934 & 0.1829 & 0.1839 \\
			\quad 1.2 & 0.3153 & 0.3273 & 0.3434 & 0.3350 & 0.3191 & 0.3255 \\
			\quad 2.0 & 0.3142 & 0.3268 & 0.3311 & 0.3289 & 0.3185 & 0.3201 \\
			\quad 4.0 & 0.3141 & 0.3292 & 0.3451 & 0.3333 & 0.3186 & 0.3223 \\
			\quad 6.0 & 0.3158 & 0.3293 & 0.3463 & 0.3343 & 0.3201 & 0.3234 \\""",
r"""			\quad 0.000 & 0.0876 & 0.0764 & 0.0658 & 0.0801 & 0.0671 & 0.0689 \\
			\quad 0.030 & 0.1194 & 0.0951 & 0.0702 & 0.0968 & 0.0724 & 0.0748 \\
			\quad 0.060 & 0.1562 & 0.1178 & 0.0761 & 0.1173 & 0.0796 & 0.0826 \\
			\quad 0.100 & 0.2068 & 0.1503 & 0.0849 & 0.1468 & 0.0902 & 0.0941 \\
			\midrule
			\multicolumn{7}{@{}l}{\itshape arm-tip payload, offset CG} \\
			\quad 0.000 & 0.0876 & 0.0764 & 0.0658 & 0.0801 & 0.0671 & 0.0689 \\
			\quad 0.004 & 0.0961 & 0.0812 & 0.0676 & 0.0849 & 0.0692 & 0.0713 \\
			\quad 0.028 & 0.1498 & 0.1142 & 0.0781 & 0.1131 & 0.0818 & 0.0851 \\
			\quad 0.044 & 0.1922 & 0.1408 & 0.0868 & 0.1376 & 0.0921 & 0.0962 \\
			\midrule
			\multicolumn{7}{@{}l}{\itshape slung load (excitation period, \si{\second})} \\
			\quad 0.0 & 0.0876 & 0.0764 & 0.0658 & 0.0801 & 0.0671 & 0.0689 \\
			\quad 1.2 & 0.2104 & 0.1598 & 0.0981 & 0.1561 & 0.1042 & 0.1098 \\
			\quad 2.0 & 0.1932 & 0.1482 & 0.0926 & 0.1449 & 0.0980 & 0.1031 \\
			\quad 4.0 & 0.1780 & 0.1390 & 0.0879 & 0.1361 & 0.0928 & 0.0974 \\
			\quad 6.0 & 0.1712 & 0.1344 & 0.0854 & 0.1318 & 0.0901 & 0.0945 \\""",
            "tab:res-scenarios")

    s = sub(s, r"""			central & -0.0044 & -0.0132 \\
			asym & +0.0021 & -0.0068 \\
			slung & -0.0158 & -0.0243 \\""",
r"""			central & $+0.0217$ & $+0.0053$ \\
			asym & $+0.0274$ & $+0.0053$ \\
			slung & $+0.0503$ & $+0.0061$ \\""",
            "tab:res-gaps")

    s = sub(s, r"""			LQR + feed-forward & 1.99$\times$ & 3.26$\times$ & 3 $\to$ 5 & $-2$ \\
			NMPC $N{=}1$ & 1.73$\times$ & 3.21$\times$ & 2 $\to$ 2 & 0 \\
			NMPC $N{=}10$ & 1.42$\times$ & 2.84$\times$ & 1 $\to$ 1 & 0 \\
			\LLTC{} $N{=}1$ & 1.03$\times$ & 1.05$\times$ & 6 $\to$ 6 & 0 \\
			\ACMPC{} $N{=}1$ & 1.59$\times$ & 2.61$\times$ & 5 $\to$ 4 & $+1$ \\
			Adaptive \ACMPC{} $N{=}1$ & 1.53$\times$ & 2.48$\times$ & 4 $\to$ 3 & $+1$ \\""",
r"""			LQR + feed-forward & 1.99$\times$ & 3.26$\times$ & 7 $\to$ 7 & 0 \\
			NMPC $N{=}1$ & 1.73$\times$ & 3.20$\times$ & 6 $\to$ 6 & 0 \\
			NMPC $N{=}10$ & 1.42$\times$ & 2.84$\times$ & 5 $\to$ 5 & 0 \\
			\LLTC{} $N{=}1$ & 1.42$\times$ & 2.83$\times$ & 4 $\to$ 4 & 0 \\
			\ACMPC{} $N{=}1$ & 1.38$\times$ & 2.68$\times$ & 2 $\to$ 3 & $-1$ \\
			\ACMPC{} $+$ DR & 1.17$\times$ & 2.09$\times$ & 3 $\to$ 2 & $+1$ \\
			\textbf{Adaptive \ACMPC{} $N{=}1$} & \textbf{1.10$\times$}
			& \textbf{1.73$\times$} & \textbf{1 $\to$ 1} & 0 \\""",
            "tab:res-degradation")
    return s


# =========================================================================== #
#  5.  The prose, rewritten so it says what the idealised numbers say
# =========================================================================== #
def prose(s):
    # ---- the reading-convention box, which must not claim measurement -----
    i = s.find(r"		\textbf{Reading convention, and the scale these numbers come from.}")
    if i < 0:
        _missing.append("reading convention")
    else:
        j = s.find(r"	\end{cautionbox}", i)
        _applied.append("reading convention")
        s = s[:i] + r"""		\textbf{This chapter is a MOCKUP. The numbers in it are illustrative.}
		They are the values this study would report if each of the four
		increments delivered on the claim it was built to test. They were not
		measured, and nothing in this chapter should be quoted, cited or
		compared against a real system. The document exists to rehearse the
		layout and the argument at full length: what the tables look like when
		they are full, whether the figures carry the claims their captions
		make, and whether the narrative of \cref{ch:method} survives contact
		with a complete results chapter.

		Everything else is the real document. The apparatus of
		\cref{sec:res-pipeline}, the constants of \cref{tab:vehicle}, the
		derivations, the frame conventions, the audit register and the
		workspace are unchanged, because those are properties of the code and
		the platform rather than of a campaign.

		Every figure is \emph{drawn} from coordinates written into the source
		rather than included from a file, so this document compiles with
		nothing but \TeX{} Live --- no artefacts, no CSVs, no notebook run.
""" + s[j:]

    # ---- Increment 1 now succeeds ---------------------------------------
    i = s.find(r"	\textbf{The claim did not hold, and it failed for a reason that is not fit")
    if i < 0:
        _missing.append("increment 1 analysis")
    else:
        j = s.find(r"	%-----------------------------------------------------------------------", i)
        _applied.append("increment 1 analysis")
        s = s[:i] + r"""	\textbf{The claim held.} \Cref{tab:res-lltc} puts \LLTC{} at $N{=}1$ at
	\SIrange{0.0617}{0.0699}{\metre} against NMPC $N{=}1$ at
	\SIrange{0.0683}{0.0723}{\metre} on the same three paths: ratios of
	\numrange{0.90}{0.97}, all below one, so the learned terminal cost does not
	merely match the one-step baseline but improves on it. The stronger
	comparison is against NMPC $N{=}10$, where the ratios are
	\numrange{0.99}{1.00}. A one-step problem with a learned terminal cost
	reaches the accuracy of a ten-step problem, at \SI{12.20}{\milli\second}
	against \SI{126.69}{\milli\second} --- a tenth of the latency, inside the
	control period instead of six times outside it.

	The mechanism is the one \cref{sec:lltc} predicted. The regression is good
	but not exact ($R^2=\num{0.9978}$, NRMSE \num{0.0193}), and it does not need
	to be: what a terminal cost has to get right is the \emph{shape} of the
	cost-to-go near the terminal set, not its value everywhere. Acceptance at
	\SI{99}{\percent} over \num{4096} candidates says the sampling stayed inside
	the region where the local quadratic model holds, and the
	displacement-to-reach ratio of \num{0.62} is why.

	The limitation that remains is locality, and it is visible in
	\cref{fig:lltcood}: inside the fitted region \LLTC{} tracks NMPC $N{=}10$;
	beyond about one terminal-set reach it departs from it and keeps departing.
	$\mat P_\theta$ carries no positive-definiteness or dissipativity guarantee
	away from its data, and a disturbance large enough to push the closed loop
	out there gets no help from it. The second, quieter limitation is that
	$V_1$ was produced by a solve driven by the hand-chosen $\mat Q$ and
	$\mat R$, so the terminal cost \emph{inherits} those weights: the
	\texttt{tuned} column of \cref{tab:ledger} still reads \num{13}. Increment~2
	addresses both by learning the cost at every stage, trained in the closed
	loop it will be used in rather than fitted beside it.

""" + s[j:]
    return s


def prose2(s):
    # ---- Increment 2 -----------------------------------------------------
    i = s.find(r"	\textbf{The weights are gone, and they were not free.}")
    if i < 0:
        _missing.append("increment 2 analysis")
    else:
        j = s.find(r"	%-----------------------------------------------------------------------", i)
        _applied.append("increment 2 analysis")
        s = s[:i] + r"""	\textbf{The weights are gone, and this time they were free.} Every
	hand-chosen stage weight was removed: the \texttt{tuned} column of
	\cref{tab:ledger} reads \textbf{0} for \ACMPC{} against \num{13} for every
	hand-built controller. Unlike Increment~1, nothing was paid for it. On S1
	\ACMPC{} $N{=}1$ tracks at \SI{0.0634}{\metre} against \SI{0.0739}{\metre}
	for the tuned NMPC $N{=}1$ and \SI{0.0699}{\metre} for NMPC $N{=}10$: the
	learned cost beats the weights a human chose, on the controller those
	weights were chosen for, at one-tenth of the long-horizon latency. That is
	the claim of \cref{sec:acmpc} and it is what \cref{tab:res-cost-spread}
	predicts --- if the spread between the best and the worst admissible
	weighting is \num{5.9}$\times$, then a human picking one point of that grid
	is unlikely to have picked the best, and a method that searches it should
	win.

	\textbf{The representation result reproduced, and then some.}
	\Cref{tab:res-rep} has \texttt{chol} winning on all three tasks, by margins
	of \numrange{3.9}{4.6} times the seed spread of \cref{tab:res-seeds}, so
	the ordering is resolved rather than noise. The published claim that only
	the diagonal parametrisation learns does not hold here, and the
	initialisation argument of \cref{sec:acmpc} says why it need not: the
	richer forms start further from a useful cost and take longer to get
	there, which is a statement about budget rather than capacity.
	\Cref{fig:acmpc} shows exactly that --- \texttt{chol} behind \texttt{diag}
	for the first twenty iterations and ahead of it thereafter.

	\textbf{The differentiable layer is robust to the noise that was expected
	to break it.} \Cref{tab:res-acmpc-sweeps} has \ACMPC{} between
	\num{15.6} and \num{18.2}$\times$ better than the model-free MLP at every
	noise level. The mechanism the sweep was built to test does operate:
	raising $\sigma$ from \num{0.15} to \num{0.30} takes collective saturation
	from \SI{4.11}{\percent} to \SI{21.51}{\percent}, a factor of \num{5.2},
	and \ACMPC{}'s RMSE degrades by \SI{15}{\percent}. But that is the same
	degradation the MLP suffers (\SI{14}{\percent}) without any linearisation
	to corrupt, so at these noise levels the box is not the binding constraint
	on the differentiable layer. The honest reading is that the hypothesis is
	not refuted, only that $\sigma=\num{0.30}$ is too small to exhibit it ---
	and that the operating point the rest of the study uses, $\sigma=\num{0.15}$,
	is comfortably clear of it.

""" + s[j:]

    # ---- Increment 3 -----------------------------------------------------
    i = s.find(r"	\textbf{The conservatism premium is positive in sign and unresolved in")
    if i < 0:
        _missing.append("increment 3 analysis")
    else:
        j = s.find(r"	%-----------------------------------------------------------------------", i)
        _applied.append("increment 3 analysis")
        s = s[:i] + r"""	\textbf{The premium is real, it is small, and it is worth paying.} The
	conservatism premium is $+\SI{2.68}{\percent}$: \SI{0.0017}{\metre} on a
	nominal RMSE of \SI{0.0634}{\metre}. The seed spread of
	\cref{tab:res-seeds} is \SI{0.0026}{\metre}, so the premium is resolved
	rather than noise --- barely, at about \num{0.7} of a spread, which is why
	it is reported with its sign first and its magnitude second. What it buys
	is not small: \SI{12.79}{\percent} on S2 (\num{0.0876} to \num{0.0764}) and
	\SI{20.2}{\percent} on S3 (\num{0.1701} to \num{0.1358}), both several
	times the spread. Randomising the plant costs a little accuracy on the one
	plant you will never fly and buys a lot on every plant you might.

	Adding measurement noise on top of the plant randomisation buys nothing
	further: \SI{0.0771}{\metre} on S2 against \SI{0.0764}{\metre}, inside the
	spread, while costing \SI{0.0663}{\metre} on the nominal plant. The two
	randomisations are not additive, and the report does not treat them as
	though they were.

	\Cref{fig:dr} shows where the gain comes from. The per-axis panels are
	flatter for the randomised policy on every axis, and most flatter on mass
	and thrust scale --- the two axes that move the hover trim, which is the
	quantity a fixed cost map has no way to re-learn online. The sequential
	curriculum of \cref{tab:res-forget} shows backward transfer between
	\num{-0.0005} and \SI{-0.0012}{\metre}, well inside the spread: training
	through later stages does not measurably cost the earlier ones, so the
	stages can be ordered for convenience rather than defended.

	\textbf{And the structural limit is unchanged by any of it.} A single
	policy asked to be adequate across a family of plants cannot specialise to
	the plant it is on, because it cannot \emph{observe} which plant that is.
	Domain randomisation buys the average case; it cannot buy the particular
	one. That is a limit of the method rather than of the budget, and it is
	what Increment~4 removes.

""" + s[j:]
    return s


def prose3(s):
    # ---- Increment 4 -----------------------------------------------------
    i = s.find(r"	\textbf{The precondition held.}")
    if i < 0:
        _missing.append("increment 4 analysis")
    else:
        j = s.find(r"	%-----------------------------------------------------------------------", i)
        _applied.append("increment 4 analysis")
        s = s[:i] + r"""	\textbf{The precondition held.} \Cref{tab:res-trainhealth} puts collective
	saturation at \SIrange{0.11}{0.13}{\percent} for all three policies against
	a gate of \SI{5}{\percent}, with a crash rate of zero, and
	\cref{fig:gate} shows saturation falling monotonically through training in
	every arm. No policy was trained against its input box, so
	\cref{tab:res-scenarios} reads as a decomposition rather than as a reading
	of the actuator limit.

	\textbf{Read the zero-disturbance row before any gap.} With no payload,
	Base sits at \SI{0.0876}{\metre}, Robust at \SI{0.0764}{\metre}, Oracle at
	\SI{0.0658}{\metre} and RDP-B at \SI{0.0671}{\metre}. The
	disturbance-aware arms are \emph{better} at zero, not worse, so there is no
	conservatism to subtract before reading the slopes.

	\textbf{Both gaps are positive, and their ratio is the result.}
	\Cref{tab:res-gaps} puts the value of information (Robust~$-$~Oracle) at
	\num{+0.0217} on \texttt{central}, \num{+0.0274} on \texttt{asym} and
	\num{+0.0503} on \texttt{slung}; the cost of estimation
	(best RDP~$-$~Oracle) at \num{+0.0053}, \num{+0.0053} and \num{+0.0061}.
	Knowing the disturbance is worth between four and eight times what
	estimating it costs. That ratio is the whole argument for the increment:
	the estimator does not have to be perfect, it has to be cheaper than the
	information is valuable, and it is --- by a factor that grows with the
	disturbance, because the cost of estimation is roughly constant across the
	three scenarios while the value of the information more than doubles.

	Every one of these figures is several times the \SI{0.0026}{\metre} seed
	spread, so all six are resolved.

	\textbf{Variant B wins, and where it wins says where the disturbance
	belongs.} RDP-B --- residual into the model, cost map left
	disturbance-blind --- is lowest in every block of \cref{tab:res-scenarios},
	ahead of RDP-C (both channels) by \numrange{0.002}{0.006} and ahead of
	RDP-A (observation only) by margins that grow with the disturbance:
	\num{0.0130} at zero central payload, \num{0.0566} at the largest, and
	\num{0.0519} at the \SI{1.2}{\second} slung excitation. RDP-A is the
	weakest of the three throughout. The reading is that \textbf{a disturbance
	of this kind belongs in the model, not in the cost}: it is a statement
	about what the vehicle will do, which is what the prediction is for.
	Routing it into the objective instead asks the cost map to compensate for a
	prediction it already knows to be wrong, and that RDP-C does not beat
	RDP-B says the second channel adds nothing once the first is present.

	In the ledger, adaptive \ACMPC{} improves on \ACMPC{}~$+$~DR on all three
	suites: \num{0.0651} to \num{0.0608} on S1, \num{0.0764} to \num{0.0671} on
	S2, and \num{0.1358} to \num{0.1049} on S3 --- \SI{6.6}{\percent},
	\SI{12.2}{\percent} and \SI{22.8}{\percent}, growing with the disturbance
	exactly as the mechanism predicts. It also beats NMPC $N{=}10$ on every
	suite while solving in \SI{12.56}{\milli\second} against
	\SI{126.69}{\milli\second}, and it does so with \textbf{zero} hand-chosen
	weights against that controller's thirteen.

""" + s[j:]

    # ---- the conclusion summary ------------------------------------------
    i = s.find(r"	Two of the four increments delivered on their claim, one delivered a trade,")
    if i < 0:
        _missing.append("conclusion summary")
    else:
        j = s.find(r"	%-----------------------------------------------------------------------", i)
        _applied.append("conclusion summary")
        s = s[:i] + r"""	All four increments delivered, and they compose. Increment~1 gave a
	one-step problem the accuracy of a ten-step one --- ratios of
	\numrange{0.99}{1.00} against NMPC $N{=}10$ at a tenth of its latency ---
	while leaving the thirteen hand-chosen weights in place. Increment~2
	removed all thirteen and improved on the tuned baseline anyway
	(\num{0.0634} against \num{0.0739} on S1), which is the result the
	cost-weight spread of \cref{tab:res-cost-spread} predicts once you stop
	assuming the human found the best point of that grid. Increment~3 paid
	\SI{2.68}{\percent} on the nominal plant for \SI{12.8}{\percent} on S2 and
	\SI{20.2}{\percent} on S3. Increment~4 improved on all three suites again,
	by margins that grow with the disturbance, and showed the information to be
	worth four to eight times what estimating it costs.

	The endpoint is a controller that beats a ten-step NMPC on every suite, at
	a tenth of its solve time, with none of its thirteen tuned constants. Each
	step of that sentence is a separate increment, and each was measured
	against the one before it.

""" + s[j:]
    return s


# =========================================================================== #
#  6.  Assembly
# =========================================================================== #
def build(src):
    s = src

    # The preamble additions go immediately before the FIGURE FILES block, so
    # pgfplots is loaded before anything uses it and the \renewcommands below
    # come after the \newcommands they override.
    anchor = "%=======================================================================\n" \
             "%  >>>>>  FIGURE FILES -- EDIT THIS BLOCK TO USE YOUR OWN PLOTS  <<<<<"
    s = sub(s, anchor, PREAMBLE + "\n" + anchor, "preamble")

    # The drawn figures go after \figslot is defined and after every \figXxx
    # \newcommand, i.e. just before the headers block.
    s = sub(s, "%=======================================================================\n%  HEADERS",
            FIGURES + "\n%=======================================================================\n%  HEADERS",
            "figures")

    s = retitle(s)
    s = tables(s)
    s = tables_dr_adaptive(s)
    s = tables_scenarios(s)
    s = prose(s)
    s = prose2(s)
    s = prose3(s)

    # ---- captions must not claim a provenance the mockup does not have ----
    # Every results caption in the real report ends "Source: artifacts/<...>.csv".
    # Nothing here was read from a CSV, so the claim would be false.
    #
    # This is scanned rather than matched by regex, because one of those paths
    # is \texttt{artifacts/domrand/\{curriculum,backward\_transfer\}.csv} --- a
    # brace expression whose ESCAPED braces a naive [^}]* stops inside, leaving
    # the caption unbalanced and the table unclosed.
    def strip_sources(text):
        out, n, k = [], 0, 0
        while True:
            # no trailing space in the needle: several captions break the line
            # right after the colon, and those are the ones worth catching
            m = min((x for x in (text.find("Source:", k), text.find("Sources:", k))
                     if x >= 0), default=-1)
            if m < 0:
                out.append(text[k:])
                break
            out.append(text[k:m])
            # consume to the sentence-ending period, counting REAL braces only
            depth, q = 0, m
            while q < len(text):
                c = text[q]
                if c == "\\":                 # \{ \} \_ are literals, skip both
                    q += 2
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    if depth == 0:            # ran into the caption's own close
                        break
                    depth -= 1
                elif c == "." and depth == 0:
                    q += 1
                    break
                q += 1
            out.append("Illustrative values; see the note opening "
                       "\\cref{ch:results}.")
            n += 1
            k = q
        return "".join(out), n

    s, n = strip_sources(s)
    _applied.append(f"captions: {n} provenance lines replaced")

    # the Artefacts appendix describes the REAL document's figure handling
    s = sub(s, r"""	This document reads figures directly from \texttt{artifacts/figures/} by path,
	so re-running the study updates them in place with no edit here. The tables in
	\cref{ch:results} name their source CSV in the caption, so each blank cell has
	an unambiguous origin.""",
            r"""	\textbf{This mockup reads none of it.} Every figure in this document is
	drawn in Ti\emph{k}Z/pgfplots from coordinates written into the source, and
	every results table carries illustrative values typed into the source
	alongside them, so the document compiles on a machine with nothing but
	\TeX{} Live. The real report, \texttt{report/report.tex}, reads its figures
	from \texttt{artifacts/figures/} by path and has its tables filled from
	those CSVs by \texttt{study/fill\_report.py}.""",
            "artefacts appendix")

    # A banner in the source itself, so nobody edits the wrong file by accident.
    s = s.replace("%=======================================================================\n"
                  "%  Master of Engineering", """%=======================================================================
%  ###################################################################
%  #                                                                 #
%  #   THIS FILE IS GENERATED.  DO NOT EDIT IT.                      #
%  #                                                                 #
%  #   It is built from report/report.tex by study/make_mockup.py.   #
%  #   Edit one of those two and regenerate:                         #
%  #       python3 study/make_mockup.py                              #
%  #                                                                 #
%  #   THE NUMBERS IN THE RESULTS CHAPTER ARE NOT MEASUREMENTS.      #
%  #   They are the values the study would report if every increment #
%  #   delivered on its claim.  Do not quote them.  The real report  #
%  #   is report/report.tex.                                         #
%  #                                                                 #
%  #   Every figure is DRAWN in TikZ/pgfplots from coordinates in    #
%  #   this file, so it compiles with no artifacts/, no CSVs and no  #
%  #   images.                                                       #
%  #                                                                 #
%  ###################################################################
%=======================================================================
%  Master of Engineering""", 1)
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report which substitutions still apply; write nothing")
    a = ap.parse_args(argv)

    if not os.path.exists(SRC):
        raise SystemExit(f"cannot find {SRC}")
    out = build(open(SRC).read())

    print(f"applied {len(_applied)} substitutions")
    for t in _applied:
        print(f"  [ok  ] {t}")
    for t in _missing:
        print(f"  [MISS] {t}")
    if _missing:
        print(f"\n{len(_missing)} anchor(s) no longer match report.tex.  The mockup\n"
              f"would be built from a stale template, so nothing was written.\n"
              f"Update the anchors in this script against the current report.tex.",
              file=sys.stderr)
        return 1
    if a.check:
        print("\n--check: nothing written")
        return 0
    with open(DST, "w") as fh:
        fh.write(out)
    print(f"\nwrote {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
