#!/bin/bash

# Generate the pdf
jupyter nbconvert Assignment\ 3.ipynb \
    --to latex \
    --template citations.tplx \
    --output assignment-3.tex

# Convert the latex
pandoc --from latex --to latex\
    --verbose \
    -o Assignment-3.pdf \
    --bibliography ref.bib \
    assignment-3.tex

rm assignment-3.tex