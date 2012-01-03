#!/usr/bin/env python

import warnings

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
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
            yield self.read_frames(e-s), s+start, e+start
        return
    
    # map channels property to nchannels
    def _get_nchannels(self):
        return self.__getattribute__('channels')
    def _set_nchannels(self, value):
        return self.__setattr__('channels', value)
    def _del_nchannels(self):
        return self.__delattr__('channels')
    
    nchannels = property(_get_nchannels, _set_nchannels, _del_nchannels, "Number of channels")

class ReferencedReader(Reader):
    def __init__(self, filename, reference, dtype = np.int16, lockdir = None):
        """
        A file reader that is referenced to some other channel
        """
        Reader.__init__(self, filename, dtype, lockdir)
        self.ref = Reader(reference, dtype, lockdir)

    def seek(self, location):
        Reader.seek(self, location)
        self.ref.seek(location)
    
    def read_frames(self, nframes):
        f = Reader.read_frames(self, nframes)
        r = self.ref.read_frames(nframes)
        return f - r
