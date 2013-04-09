#!/usr/bin/env python


from . import butter
from . import wavelet


__all__ = ['butter', 'wavelet']


class Filter(object):
    def __new__(cls, f, args):
        o = object.__new__(cls)
        o.f = f
        o.args = args
        return o

    def __getnewargs__(self):
        return self.f, self.args

    def __call__(self, d):
        return self.f(d, *self.args)


def from_kwargs(**kwargs):
    method = kwargs.pop('method')
    if method == 'butter':
        flow = kwargs['low']
        fhigh = kwargs['high']
        sr = kwargs['samprate']
        order = kwargs['order']

        return Filter(butter.filt, (flow, fhigh, sr, order)), kwargs

        #def f(x):
        #    return butter.filt(x, flow, fhigh, sr, order)
        #return f, kwargs
        #return lambda x: butter.filt(x, flow, fhigh, sr, order), kwargs
    elif method == 'wavelet':
        minlvl = kwargs['minlvl']
        maxlvl = kwargs['maxlvl']
        wtype = kwargs['wavelet']
        mode = kwargs['mode']

        return Filter(wavelet.filt, (maxlvl, wtype, mode, minlvl)), kwargs

        def f(x):
            return wavelet.filt(x, maxlvl, wtype, mode, minlvl)
        return f, kwargs
        #return lambda x: wavelet.filt(x, maxlvl, wtype, mode, minlvl), kwargs
    else:
        raise ValueError('Unknown filt method: %s' % method)


def from_config(config, section='filter'):
    kwargs = {}
    for k in ('method', 'low', 'high', 'samprate', 'order',
              'minlvl', 'maxlvl', 'wavelet', 'mode'):
        if config.has_option(section, k):
            kwargs[k] = config.get(section, k)
    return from_kwargs(**kwargs)
