#!/usr/bin/env python

#
# gmvrect.py
# Writes a rectangular gmv mesh with user-specified dimensions
#

import sys

def usage():
    print "Usage:  gmvrect.py NZX [NZY [LENX [LENY]]]"
    print "where nzx, nzy   = number of zones in x, y directions"
    print "                   (no default for nzx; default nzy = nzx)"
    print "      lenx, leny = total length in x, y directions"
    print "                   (default for both = 1.0)"
    sys.exit(0)

def writecoords(file, np, x):
    for i in range(np):
        if i % 10 == 0:  s = "  "
        s += "%16.8E" % x[i]
        if (i % 10 == 9) or (i == np - 1):  file.write("%s\n" % s)

nargs = len(sys.argv)
nzx = 0
nzy = 0
lenx = 1.0
leny = 1.0
if nargs < 2 or nargs > 5:  usage()
try:
    nzx = int(sys.argv[1])
    nzy = nzx
    if nargs > 2: nzy = int(sys.argv[2])
    if nargs > 3: lenx = float(sys.argv[3])
    if nargs > 4: leny = float(sys.argv[4])
except:
    usage()
if nzx <= 0 or nzy <= 0 or lenx <= 0. or leny <= 0.:  usage()
    
nz = nzx * nzy
npx = nzx + 1
npy = nzy + 1
np = npx * npy

filename = "rect%dx%d.gmv" % (nzx, nzy)
file = open(filename, "w")

# write header
file.write("gmvinput ascii\n")

# write node header
file.write("nodes  %9d\n" % np)

# write node coordinates
x = np * [0]
y = np * [0]
ijtop = np * [0]
for n in range(np):
    i = n % npx
    j = n / npx
    x[n] = (lenx * float(i) / nzx)
    y[n] = (leny * float(j) / nzy)
    ijtop[j * npx + i] = n

writecoords(file, np, x)
writecoords(file, np, y)
writecoords(file, np, np * [0])

# write cell header
file.write("cells  %9d\n" % nz)

# write cells
ztop = nz * [[]]
for n in range(nz):
    i = n % nzx
    j = n / nzx
    # +1 is to convert from 0-based to 1-based
    p0 = ijtop[j * npx + i] + 1
    p1 = ijtop[j * npx + (i+1)] + 1
    p2 = ijtop[(j+1) * npx + (i+1)] + 1
    p3 = ijtop[(j+1) * npx + i] + 1
    ztop[n] = [p0, p1, p2, p3]

for n in range(nz):
    file.write("  general          1\n")
    file.write("             4\n")
    pl = ztop[n]
    file.write("     %9d %9d %9d %9d\n" % (pl[0], pl[1], pl[2], pl[3]))

# write end-of-file marker
file.write("endgmv\n")

print "Wrote %s" % filename
