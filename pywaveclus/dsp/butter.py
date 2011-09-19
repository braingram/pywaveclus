#!/usr/bin/env python

import scipy.signal

def filt(data, flow = 300, fhigh = 3000, sr = 44100., order = 3):
    """
    Can introduce significant errors at the start and stop of the signal if SNR is low
    """
    b, a = scipy.signal.butter(order, ((flow/(sr/2.)), (fhigh/(sr/2.))), 'pass')
    return scipy.signal.filtfilt(b, a, data)