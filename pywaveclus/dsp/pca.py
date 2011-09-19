#!/usr/bin/env python

# from matplotlib.mlab import PCA
from scikits.learn.decomposition.pca import PCA

def features(waveforms, nfeatures = 3):
    p = PCA(nfeatures, whiten=True)
    return p.fit(waveforms).transform(waveforms)
    # Td = p.fit(waveforms).transform(waveforms)
    # import matplotlib
    # matplotlib.use('MacOSX')
    # import pylab as pl
    # pl.clf()
    # pl.plot(p.components_.T)
    # pl.savefig('pca.png')
    # return Td