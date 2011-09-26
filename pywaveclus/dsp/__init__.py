#!/usr/bin/env python

import butter
import ica
import interpolate
import pca
import threshold
import wavelet

__all__ = ['butter', 'ica', 'interpolate', 'pca', 'threshold', 'wavelet']

def filt_from_config(config):
    method = config.get('filter','method')
    
    if method == 'butter':
        flow = config.getfloat('filter','low')
        fhigh = config.getfloat('filter','high')
        sr = config.getint('filter','samprate')
        order = config.getint('filter','order')
        return lambda x: butter.filt(x, flow, fhigh, sr, order)
    elif method == 'wavelet':
        minlvl = config.getint('filter','minlvl')
        maxlvl = config.getint('filter','maxlvl')
        wtype = config.get('filter','wavelet')
        mode = config.get('filter','mode')
        return lambda x: wavelet.filt(x, maxlvl, wtype, mode, minlvl)
    else:
        raise ValueError("Unknown filt method: %s" % method)
