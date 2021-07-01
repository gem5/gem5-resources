#! /usr/bin/env python

import urllib
import tarfile
import os

print("Downloading...")

testfile = urllib.URLopener()
testfile.retrieve("http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz", "cifar-10-binary.tar.gz")

print("Unzipping...")

tar = tarfile.open("cifar-10-binary.tar.gz")
tar.extractall()
tar.close()

os.system("rm -f cifar-10-binary.tar.gz")
os.system("mv cifar-10-batches-bin/* .")
os.system("rm -rf cifar-10-batches-bin")

print("Done.")
