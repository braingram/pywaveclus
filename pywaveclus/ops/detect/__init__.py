#!/usr/bin/env python

import logging

#import neo
import threshold

__all__ = ['threshold']


def from_kwargs(baseline, **kwargs):
    method = kwargs.get('method', 'threshold')
    if method == 'threshold':
        direction = kwargs['direction']
        ref = kwargs['ref']
        minwidth = kwargs['minwidth']
        slop = kwargs['slop']
        n = kwargs['nthresh']
        T = [threshold.calculate_threshold(b, n) for b in baseline]
        AT = [t / float(n) * kwargs['artifact'] for t in T]
        logging.debug("Found threshold: %s" % T)
        logging.debug("Found artifact threshold: %s" % AT)
        if T == 0.:
            return lambda x: ([], [])
        return lambda i, x: threshold.find_spikes(
            x, T[i], AT[i], direction, ref, minwidth, slop)
    elif method == 'neo':
        raise NotImplementedError
    else:
        raise ValueError('Unknown detect method: %s' % method)


def from_config(baseline, cfg, section='detect'):
    kwargs = {}
    for k in ('method', 'direction', 'ref', 'minwidth',
              'slop', 'nthresh', 'artifact'):
        if cfg.has_option(section, k):
            kwargs[k] = cfg.get(section, k)
    return from_kwargs(baseline, **kwargs)
