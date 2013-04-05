#!/usr/bin/env python

import logging
import sys

logging.basicConfig(level=logging.DEBUG)

import pywaveclus

if len(sys.argv) < 2:
    raise Exception("Invalid number of args, please provide session dir")
bdir = sys.argv[1]

ofn = 'output.h5'
if len(sys.argv) > 2:
    ofn = sys.argv[2]

store, info = pywaveclus.process.process_session(bdir, ofn, full=True)
print "Session Info"
for k, v in info.iteritems():
    print '%s:\t%s' % (k, v)
store.close()
