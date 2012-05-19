#!/usr/bin/env python

import cconfig

CFGDEFAULTS = """
[main]
njobs: -1
outputdir:
timerange:
plot:
verbose:
filename:
reference:
adjacentfiles:

[reader]
probetype: nna
indexre: _([0-9]+)#*
chunksize: 441000
chunkoverlap: 441
dtype: f8
start: 0.0
stop: 1.0

[ica]
filename = 'ica.p'
method = random
sargs = 102400
ncomponents = 32
count = 3
stdthreshold = 2

[filter]
method: wavelet
minlvl: 4
maxlvl: 6
wavelet: db20
mode: sym

[detect]
nthresh: 5.0
artifact: 45.0
baseline: 100000
direction: both
minwidth: 1
ref: 44

[extract]
pre: 40
post: 88

[cluster]
nfeatures: 6
featuretype: pca
minclusters: 1
maxclusters: 5
minspikes: 25
separate: peak
pre: 40

[writer]
pre: 40
post: 88
nfeatures: 6
filename: pycluster.h5
"""


def load(custom=None, args=None):
    """
    Parameters
    ----------
    custom : string
        Filename of custom ini file to load
    args : list
        Command line arguments to process

    Returns
    -------
    cfg : cconfig.CMDConfig
        Config object with precedence (least to most):
            CFG_DEFAULTS
            ~/.pywaveclus
            pywaveclus.ini
            custom
            args
    """
    local = ["pywaveclus.ini"]
    if custom is not None:
        local += custom
    return cconfig.CMDConfig(base=CFGDEFAULTS, user='pywaveclus', \
            local=local, options=args)
