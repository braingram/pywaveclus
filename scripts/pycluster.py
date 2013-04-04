#!/usr/bin/env python

import logging
import os
import sys

logging.basicConfig(level=logging.DEBUG)

import tables

import pywaveclus

store = pywaveclus.ops.store.hdf5.SpikeStorage(
    tables.openFile('output.h5', 'w'))

fns = sorted([os.path.realpath(fn) for fn in sys.argv[1:]])
cfg = pywaveclus.utils.load_config()
info, cfg, reader, ff, df, ef, cf = \
    pywaveclus.process.get_operations(fns, cfg=cfg)

store.save_info(info)
pywaveclus.process.process_file(cfg, reader, ff, df, ef, cf, store)
store.close()
