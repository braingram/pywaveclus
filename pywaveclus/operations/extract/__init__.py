#!/usr/bin/env python


import simple

__all__ = ['simple']


class SimpleExtract(object):
    def __init__(self, pre, post):
        self.pre = pre
        self.post = post

    def __call__(self, data, indices):
        return simple.simple(data, indices, self.pre, self.post)


def get_extract(cfg, section='extract'):
    pre = cfg.getint(section, 'pre')
    post = cfg.getint(section, 'post')
    return SimpleExtract(pre, post)
