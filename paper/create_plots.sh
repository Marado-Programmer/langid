#!/bin/sh
# This is part of the LANGuage IDentification.
# Copyright (C) 2024
#   João Augusto Costa Branco Marado Torres
# See the file ./paper.tex for copying conditions.

set -xe

mkdir -p ./res

tmp=$(mktemp)

../zig-out/bin/idlang eg doubles 20 \
    --layers=1-1 \
    --learning-rate=5e-3 \
    --learning-amount=300 > "$tmp"
gnuplot -c ./create_plots.gnuplot \
    "$tmp" ./res/doubles.tex "\$ y = 2x \$"

../zig-out/bin/idlang eg abs 20 \
    --layers=4-2-1 \
    --learning-rate=5e-3 \
    --learning-amount=500 > "$tmp"
gnuplot -c ./create_plots.gnuplot \
    "$tmp" ./res/abs.tex "\$ y = \left| x \right| \$"

../zig-out/bin/idlang eg sin 12 \
    --layers=6-4-2-1 \
    --learning-rate=1e-4 \
    --learning-amount=1000 > "$tmp"
gnuplot -c ./create_plots.gnuplot \
    "$tmp" ./res/sin.tex "\$ y = \sin \left( x \right) \$"

../zig-out/bin/idlang eg linear 60 \
    --layers=4-2-1 \
    --learning-rate=5e-4 \
    --learning-amount=1000 > "$tmp"
gnuplot -c ./create_plots.gnuplot \
    "$tmp" ./res/linear.tex "\$ y = ax + b \$"

../zig-out/bin/idlang eg parabola 80 \
    --layers=8-4-4-1 \
    --learning-rate=5e-6 \
    --learning-amount=1500 > "$tmp"
gnuplot -c ./create_plots.gnuplot \
    "$tmp" ./res/parabola.tex "\$ y = ax^2 + bx + c \$"
