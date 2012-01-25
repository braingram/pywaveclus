#!/usr/bin/env python

import os

import numpy as np

# from matplotlib.mlab import PCA
from scikits.learn.decomposition.pca import PCA

global FEATURES
FEATURES = None

def stack_waveforms(waveforms):
    """
    convert waveforms (possibly [index, ch, form]) to
    [index, stacked_form] where stacked_form contains
    the waveform from all channels hstacked
    """
    waves = []
    for wave in waveforms:
        wf = np.array([])
        for ch in wave: wf = np.hstack((wf, ch))
        waves.append(wf)
    return np.array(waves)

def features_from_info(waveforms, info):
    return np.dot((stack_waveforms(waveforms) - info['mean']), info['components'].T)

def features(waveforms, nfeatures = 3, usesaved = False):
    #waves = []
    #for wave in waveforms:
    #    wf = np.array([])
    #    for ch in wave: wf = np.hstack((wf, ch))
    #    waves.append(wf)
    #waves = np.array(waves)
    waves = stack_waveforms(waveforms)
    p = PCA(nfeatures)#, whiten=True)
    if usesaved:
        global FEATURES
        if FEATURES is None: load_features(nfeatures)
        return project_waveforms(waves)
    else:
        p.fit(waves)
        info = {}
        info['mean'] = p.mean_
        info['components'] = p.components_
        return p.transform(waves), info
        #return p.fit(waves).transform(waves)
    # to 'save' the pca, store:
    #   1) p.mean_
    #   2) p.components_
    # to 'transform' data:
    #   dot((d - p.mean_), p.components_.T)
    #
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
