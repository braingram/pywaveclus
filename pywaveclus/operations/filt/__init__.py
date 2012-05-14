#!/usr/bin/env python


import butter
import wavelet

__all__ = ['butter', 'wavelet']


class ButterFilt(object):
    def __init__(self, flow, fhigh, sr, order):
        self.flow = flow
        self.fhigh = fhigh
        self.sr = sr
        self.order = order

    def __call__(self, data):
        return butter.filt(data, self.flow, self.fhigh, self.sr, self.order)


class WaveletFilt(object):
    def __init__(self, minlvl, maxlvl, wave, mode):
        self.minlvl = minlvl
        self.maxlvl = maxlvl
        self.wave = wave
        self.mode = mode

    def __call__(self, data):
        return wavelet.filt(data, self.maxlvl, self.wave, self.mode, \
                self.minlvl)


def get_filt(cfg, section='filter'):
    method = cfg.get(section, 'method')
    if method == 'butter':
        flow = cfg.getfloat(section, 'flow')
        fhigh = cfg.getfloat(section, 'fhigh')
        sr = cfg.getint(section, 'samplerate')
        order = cfg.getint(section, 'order')
        return ButterFilt(flow, fhigh, sr, order)
    elif method == 'wavelet':
        minlvl = cfg.getint(section, 'minlvl')
        maxlvl = cfg.getint(section, 'maxlvl')
        wave = cfg.get(section, 'wavelet')
        mode = cfg.get(section, 'mode')
        return WaveletFilt(minlvl, maxlvl, wave, mode)
    else:
        raise ValueError("Unknown filt method: %s" % method)
