#!/bin/sh
# This is part of the LANGuage IDentification.
# Copyright (C) 2024
#   João Augusto Costa Branco Marado Torres
# See the file ./paper.tex for copying conditions.

set -xe

pdflatex paper
# bibtex8 paper
biber paper
pdflatex paper
pdflatex paper
