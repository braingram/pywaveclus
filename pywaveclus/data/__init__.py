#!/usr/bin/env python

import os

import numpy as np

import audio
import raw

__all__ = ['audio', 'raw']

def reader_from_config(cfg):
    filename = cfg.get('main','filename')
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.wav':
        dtype = np.dtype(cfg.get('reader','dtype'))
        lockdir = cfg.get('reader','lockdir')
        if lockdir.strip() == '': lockdir = None
        reference = cfg.get('main','reference')
        if reference.strip() != '':
            return audio.ReferencedReader(filename, reference, dtype, lockdir)
        else:
            return audio.Reader(filename, dtype, lockdir)
    else:
        raise ValueError('Unknown extension %s [%s]' % (ext, filename))
