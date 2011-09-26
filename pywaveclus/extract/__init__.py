#!/usr/bin/env python

import simple
import cubic

def extract_from_config(cfg):
    method = cfg.get('extract', 'method')
    if method == 'simple':
        pre = cfg.getint('detect','pre')
        post = cfg.getint('detect','post')
        
        return lambda readers, indices: simple.simple(readers, indices, pre, post)
    elif method == 'cubic':
        pre = cfg.getint('detect','pre')
        post = cfg.getint('detect','post')
        direction = cfg.get('detect','direction')
        oversample = cfg.getfloat('detect','oversample')
        
        return lambda readers, indices: cubic.cubic(readers, indices, pre, post, direction, oversample)
    else:
        raise ValueError("Unknown extract method: %s" % method)
