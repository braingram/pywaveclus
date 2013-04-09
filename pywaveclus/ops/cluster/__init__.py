#!/usr/bin/env python

import klustakwik
#import skl
#import spc

__all__ = ['klustakwik']


class Cluster(object):
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
    method = kwargs.get('method', 'klustakwik')
    if method == 'klustakwik':
        nf = kwargs['nfeatures']
        minc = kwargs['minclusters']
        maxc = kwargs['maxclusters']
        ft = kwargs['featuretype']
        sep = (kwargs['separate'] == 'peak')
        pre = kwargs['pre']
        mins = kwargs['minspikes']

        return Cluster(klustakwik.cluster,
                       (nf, ft, minc, maxc, sep, pre, mins)), kwargs

        def f(x):
            return klustakwik.cluster(x, nf, ft, minc, maxc, sep, pre, mins)

        return f, kwargs
        #return lambda x: klustakwik.cluster(x, nf, ft, minc, maxc,
        #                                    sep, pre, mins), kwargs
    elif method in ('spc', 'kmeans', 'gmm'):
        raise NotImplementedError
    else:
        raise ValueError('Unknown cluster method: %s' % method)


def from_config(cfg, section='cluster'):
    kwargs = {}
    for k in ('method', 'nfeatures', 'minclusters', 'maxclusters',
              'featuretype', 'separate', 'pre', 'minspikes'):
        if cfg.has_option(section, k):
            kwargs[k] = cfg.get(section, k)
    return from_kwargs(**kwargs)
