"""Convert the percent-format notebook sources to .ipynb.

The sources are kept as .py so that git diffs are readable and so that a
notebook can be run headless with `python nb1_control_problem.py`; the .ipynb
files are generated, never hand-edited.
"""
import os
import re
import sys

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))


def split_cells(src):
    cells, cur, kind = [], [], "code"
    for line in src.splitlines():
        m = re.match(r"^# %%(?:\s*\[(\w+)\])?\s*$", line)
        if m:
            if cur:
                cells.append((kind, "\n".join(cur).strip("\n")))
            cur, kind = [], (m.group(1) or "code")
            continue
        cur.append(line[2:] if (kind == "markdown" and line.startswith("# ")) else
                   ("" if (kind == "markdown" and line.strip() == "#") else line))
    if cur:
        cells.append((kind, "\n".join(cur).strip("\n")))
    return cells


def convert(path):
    nb = nbf.v4.new_notebook()
    for kind, body in split_cells(open(path).read()):
        if not body.strip():
            continue
        nb.cells.append(nbf.v4.new_markdown_cell(body) if kind == "markdown"
                        else nbf.v4.new_code_cell(body))
    out = path[:-3] + ".ipynb"
    nbf.write(nb, out)
    return out


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(
        os.path.join(HERE, f) for f in os.listdir(HERE)
        if re.match(r"nb\d_.*\.py$", f))
    for f in files:
        print("  ->", os.path.basename(convert(f)))
