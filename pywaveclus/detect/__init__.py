#!/usr/bin/env python

import logging

import neo
import threshold

from .. import utils

__all__ = ['neo', 'threshold']

def detect_from_config(reader, filt, cfg):
    method = cfg.get('detect','method')
    baseline = utils.parse_time_range(cfg.get('detect','baseline'), 0, reader.nframes, int)
    reader.seek(baseline[0])
    d = reader.read_frames(baseline[1] - baseline[0])
    
    if method == 'threshold':
        n = cfg.getfloat('detect','nthresh')
        T = threshold.calculate_threshold(filt(d), n)
        AT = T / float(n) * cfg.getfloat('detect','artifact')
        logging.debug("Found threshold: %f" % T)
        if T == 0.: return lambda x: ([], [])
        
        pre = cfg.getint('detect','pre')
        post = cfg.getint('detect','post')
        direction = cfg.get('detect','direction')
        minwidth = cfg.getint('detect','minwidth')
        slop = cfg.getint('detect','slop')
        ref = cfg.getint('detect','ref')
        oversample = cfg.getint('detect','oversample')
        sliding = cfg.getboolean('detect','sliding')
        
        return lambda x: threshold.find_spikes(x, T, AT, direction, pre, post, ref, minwidth, slop, oversample)
    elif method == 'neo':
        raise NotImplemented
    else:
        raise ValueError("Unknown detect method: %s" % method)
