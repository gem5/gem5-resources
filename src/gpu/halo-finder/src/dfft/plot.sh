#!/usr/bin/env gnuplot

set terminal x11 persist
set xlabel 'x'
set ylabel 'phi(x)'
set title 'Solution of Laplace phi(x) = - delta(x)'

plot 'plot.in' title '256^3 grid', - 1 / (4 * pi * x) title '-1 / (4 pi x)'

exit

set terminal png
set output 'plot.png'
replot

set terminal postscript eps
set output 'plot.eps'
replot
