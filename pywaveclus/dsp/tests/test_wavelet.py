#!/usr/bin/env python

import numpy as np

import pywaveclus

def test_calculate_cutoffs():
    sf = 44100
    coffs = pywaveclus.dsp.wavelet.calculate_cutoffs(sf)
    r = np.array([22050, 11025, 5512, 2756, 1378, 689, 344, 172, 86, 43, 21, 10, 5, 2, 1, 0])
    assert all(coffs == r), "%s != %s" % (str(coffs), str(r))

def test_level_to_cutoffs():
    sf = 44100
    assert pywaveclus.dsp.wavelet.level_to_cutoffs(sf,1) == (11025, 22050), \
        "level_to_cutoffs(%i,1) != (11025, 22050)" % \
        (sf, str(pywaveclus.dsp.wavelet.level_to_cutoffs(sf,1)))

def test_waveletfilter(plot=False):
    # make test signal
    Fs = 44100
    freqs = [100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
    t = np.arange(Fs,dtype=np.float64) / float(Fs) # 1 second
    x = np.zeros(len(t))
    for f in freqs:
        x += np.sin(t * f * 2. * np.pi)
    x /= len(freqs)
    # print t,x
    
    filtered = pywaveclus.dsp.wavelet.filt(x, minlevel=3, maxlevel=6)
    assert len(filtered) == len(x),\
        "len(filtered)[%i] != len(x)[%i]" % (len(filtered), len(x))
    
    # TODO: test effectiveness of cutoffs
    
    if plot:
        import pylab as pl
        pl.subplot(221)
        pl.plot(t,x)
        pl.subplot(222)
        pl.psd(x,Fs=Fs)
        pl.subplot(223)
        pl.plot(t,filtered)
        pl.subplot(224)
        pl.psd(filtered,Fs=Fs)
        
        pl.show()

def test_features(plot=False):
    swf1 = [0., 0.2, 0.4, 0.8, 1.6, 1.4, 0.8, 0.3, -0.6, -0.8, -0.5, -0.3, 0.]
    # swf2 = [0., 0.1, 0.5, 0.9, 1.6, 1.4, 0.4, 0.1, -0.3, -0.2, -0.1,   0., 0.]
    swf2 = [0., 0., 0.1, 0.3, 1.8, 2.1, 0.4, 0.1, -0.3, -0.2, -0.1,   0., 0.]
    assert len(swf1) == len(swf2)
    noiselvl = 1./10.
    n1 = 10
    n2 = 20
    nfeatures = 3
    n = n1 + n2
    wfs = np.random.randn(n,len(swf1)) * noiselvl
    for i in xrange(n):
        if np.random.rand() * n < n1:
            wfs[i] += swf1
        else:
            wfs[i] += swf2
    
    features = pywaveclus.dsp.wavelet.features(wfs, nfeatures=nfeatures)
    # print n, nfeatures, features.shape
    assert features.shape == (n,nfeatures),\
        "features.shape[%s] should == %s" % (str(features.shape), str((n,nfeatures)))
    
    if plot:
        import pylab as pl
        from mpl_toolkits.mplot3d import Axes3D
        pl.figure()
        ax = pl.gcf().add_subplot(2, 2, 1, projection='3d')
        ax.scatter(features[:,0],features[:,1],features[:,2])
        pl.subplot(222)
        pl.plot(np.transpose(wfs))
        pl.subplot(223)
        pl.plot(swf1)
        pl.plot(swf2)
        pl.show()