#!/usr/bin/env python

import cconfig

CFGDEFAULTS = """
[main]
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
chunkoverlap: 44100
dtype: f8

[ica]
icafilename = 'ica.p'
icamethod = random
icasargs = 102400
icancomponents = 32
icacount = 3
icastdthreshold = 2

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
