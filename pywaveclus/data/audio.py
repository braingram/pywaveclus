#!/usr/bin/env python

import numpy as np

import scikits.audiolab

from .. import utils

class Reader(scikits.audiolab.Sndfile):
    def __init__(self, filename, dtype = np.int16, lockdir = None):
        """
        """
        scikits.audiolab.Sndfile.__init__(self, filename)
        self.dtype = dtype
        self.lockdir = lockdir
    
    # self.samplerate
    # self.channels
    # self.nframes
    # self.seek
    # self.read_frames
    
    def read_frames(self, nframes):
        if self.lockdir is None:
            return scikits.audiolab.Sndfile.read_frames(self, nframes, self.dtype)
        else:
            with utils.waiting_lock_dir(self.lockdir):
                f = scikits.audiolab.Sndfile.read_frames(self, nframes, self.dtype)
            return f
    
    def chunk(self, start, end, chunksize, overlap):
        for (s,e) in utils.chunk(end-start, chunksize, overlap):
            self.seek(s + start)
            yield self.read_frames(e-s), s, e
        return
    
    # map channels property to nchannels
    def _get_nchannels(self):
        return self.__getattribute__('channels')
    def _set_nchannels(self, value):
        return self.__setattr__('channels', value)
    def _del_nchannels(self):
        return self.__delattr__('channels')
    
    nchannels = property(_get_nchannels, _set_nchannels, _del_nchannels, "Number of channels")