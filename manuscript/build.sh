#!/bin/sh
# Build main.pdf in the local TeX image (texlive latest-medium plus the packages
# the MDPI class needs; created with tlmgr, tagged keyspark/texlive-mdpi:local).
set -e
cd "$(dirname "$0")"
docker run --rm -v "$PWD":/w -w /w keyspark/texlive-mdpi:local bash -c \
  "pdflatex -interaction=nonstopmode main >/dev/null 2>&1; bibtex main >/dev/null 2>&1; \
   pdflatex -interaction=nonstopmode main >/dev/null 2>&1; pdflatex -interaction=nonstopmode main >/dev/null 2>&1; true"
grep -n '^! ' main.log | head -20 || true
echo "undefined refs/cites: $(grep -c 'Warning: Reference\|Warning: Citation' main.log)"
echo "overfull boxes: $(grep -c 'Overfull' main.log)"
