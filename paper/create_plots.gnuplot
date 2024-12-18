# This is part of the LANGuage IDentification.
# Copyright (C) 2024
#   João Augusto Costa Branco Marado Torres
# See the file ./paper.tex for copying conditions.
set terminal tikz size 3.5in,2.4in
set output ARGV[2]
set title ARGV[3]
set xlabel "Times model learned"
set ylabel "$ E ( x ) $"
set y2label "$ y $"
set y2tics
set ytics nomirror
set key off
plot ARGV[1] using 1:2 with lines title "Error" lt 1 lc rgb "red", \
    "" using 1:3 axes x1y2 with lines title ARGV[3] lt 1 lc rgb "blue"
