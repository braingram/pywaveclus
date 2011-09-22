#!/usr/bin/env python

import os

import numpy as np

# from matplotlib.mlab import PCA
from scikits.learn.decomposition.pca import PCA

global FEATURES
FEATURES = None

def features(waveforms, nfeatures = 3, usesaved = True):
    p = PCA(nfeatures, whiten=True)
    if usesaved:
        global FEATURES
        if FEATURES is None: load_features(nfeatures)
        return project_waveforms(waveforms)
    else:
        return p.fit(waveforms).transform(waveforms)
    # Td = p.fit(waveforms).transform(waveforms)
    # import matplotlib
    # matplotlib.use('MacOSX')
    # import pylab as pl
    # pl.clf()
    # pl.plot(p.components_.T)
    # pl.savefig('pca.png')
    # return Td

def load_features(nfeatures):
    ffilename = os.path.dirname(os.path.abspath(__file__)) + '/../bin/features.txt'
    tfilename = os.path.dirname(os.path.abspath(__file__)) + '/../bin/timeseries.txt'
    features = np.loadtxt(ffilename)
    ts = np.loadtxt(tfilename)
    global FEATURES
    FEATURES = np.array([np.interp(np.arange(-40,88,dtype=np.float64)/44100.,ts,fs) for fs in features]).astype(np.float64)[:nfeatures]

def project_waveform(waveform):
    global FEATURES
    return 100 * np.dot(FEATURES, waveform).T

def project_waveforms(waveforms):
    return np.array([project_waveform(w) for w in waveforms])