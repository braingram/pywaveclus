#!/usr/bin/env python

import os

import icapp

from ... import utils


class Reader(icapp.fio.MultiAudioFile):
    def __init__(self, filenames, **kwargs):
        self._chunksize = kwargs.pop('chunksize', 441000)
        self._chunkoverlap = kwargs.pop('chunkoverlap', 0)
        icapp.fio.MultiAudioFile.__init__(self, filenames, **kwargs)

    def chunk(self, start, end, size=None, overlap=None):
        size = self._chunksize if size is None else size
        overlap = self._chunkoverlap if overlap is None else overlap
        for (s, e) in utils.chunk(end - start, size, overlap):
            self.seek(s + start)
            yield self.read(e - s), s + start, e + start


class ICAReader(Reader):
    def __init__(self, filenames, ica_info, **kwargs):
        for (icafn, fn) in zip(ica_info['fns'], filenames):
            assert os.path.basename(icafn) == os.path.basename(fn)
        #assert ica_info['fns'] == filenames
        Reader.__init__(self, filenames, **kwargs)
        self._cm = ica_info['cm']
        # function swapping
        self.raw_read = self.read
        self.read = self.clean_and_read

    def clean_and_read(self, n):
        return icapp.ica.clean_data(self.raw_read(n), self._cm)
