#!/usr/bin/env python

import icapp

from ... import utils


class Reader(icapp.fio.MultiAudioFile):
    def __init__(self, filenames, **kwargs):
        icapp.fio.MultiAudioFile.__init__(self, filenames, **kwargs)
        self._chunksize = kwargs.get('chunksize', 441000)
        self._chunkoverlap = kwargs.get('chunkoverlap', 0)

    def chunk(self, start, end, size=None, overlap=None):
        size = self._chunksize if size is None else size
        overlap = self._chunkoverlap if overlap is None else overlap
        for (s, e) in utils.chunk(end - start, size, overlap):
            self.seek(s + start)
            yield self.read_frames(e - s), s + start, e + start


class ICAReader(Reader):
    def __init__(self, filenames, ica_info, **kwargs):
        assert ica_info['fns'] == filenames
        Reader.__init__(self, ica_info['fns'], **kwargs)
        self._cm = ica_info['cm']
        # function swapping
        self.raw_read = self.read
        self.read = self.clean_and_read

    def clean_and_read(self, n):
        return icapp.ica.clean_data(self.raw_read(n), self._cm)
