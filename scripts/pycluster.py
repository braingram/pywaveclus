#!/usr/bin/env python

import glob
import logging
import os
import sys

logging.basicConfig(level=logging.DEBUG)

import numpy
import tables

import pywaveclus

if len(sys.argv) < 2:
    raise Exception("Invalid number of args, please provide session dir")
bdir = sys.argv[1]

ofn = 'output.h5'
if len(sys.argv) > 2:
    ofn = sys.argv[2]

# load M & U
# ica: M * U
# get afs
afdir = bdir + '/Audio Files'
assert os.path.exists(afdir)
fns = sorted([os.path.realpath(fn) for fn in glob.glob(afdir + '/input*')])
print "filenames:", fns

ica = None
ica_mm = bdir + '/mixingmatrix'
ica_um = bdir + '/unmixingmatrix'
if (os.path.exists(ica_mm) and os.path.exists(ica_um)):
    mm = numpy.matrix(numpy.loadtxt(ica_mm))
    um = numpy.matrix(numpy.loadtxt(ica_um))
    cm = mm * um
    ica = dict(cm=cm, fns=fns)
    print "found ica"
else:
    print "no ica found"

store = pywaveclus.ops.store.hdf5.SpikeStorage(
    tables.openFile(ofn, 'w'))

cfg = pywaveclus.utils.load_config()
info, cfg, reader, ff, df, ef, cf = \
    pywaveclus.process.get_operations(fns, ica=ica, cfg=cfg)

store.save_info(info)
pywaveclus.process.process_file(cfg, reader, ff, df, ef, cf, store)
store.close()
