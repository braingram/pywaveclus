#!/usr/bin/env python

import sys

import pywaveclus.process

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
#logging.debug("Saving results to %s" % outfile)

# plot?
#logging.debug("Plotting")
