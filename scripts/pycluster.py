#!/usr/bin/env python

import logging
import os
import sys

import numpy as np
import pylab as pl
import matplotlib
import tables

import pywaveclus

# process command line options
# -- files
# -- -v verbose
# -- -c config file?
# -- -o output file
# -- -p plot?

# load config file
# load files
files, cfg = pywaveclus.process.cmdline.parse(sys.argv[1:])

# process files
output = pywaveclus.process.cmdline.process(files, cfg)
#cfg, times, waves, clusters, info = pywaveclus.process.process.process_file()

# save
logging.debug("Saving results to %s" % outfile)

# plot?
logging.debug("Plotting")
