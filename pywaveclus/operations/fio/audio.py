#!/usr/bin/env python

import os
import re
#import warnings

#import numpy

#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    import scikits.audiolab

from ... import probes
#from ... import utils

import icapp


def position_sorted(fns, ptype, indexre):
    to_pos = probes.lookup_converter_function(ptype, 'tdt', 'pos')
    return sorted(fns, key=lambda fn: \
            to_pos(int(re.findall(indexre, os.path.basename(fn))[0])))


class Reader(icapp.fio.MultiAudioFile):
    def __init__(self, filenames=None, probetype='nna', \
            indexre=r'_([0-9]+)\#*', chunksize=441000,
            chunkoverlap=0, start=0, stop=None, **kwargs):
        assert filenames is not None, "No filenames supplied to reader"
        # position sort filenames and create a the multiaudiofile
        icapp.fio.MultiAudioFile.__init__(self, \
                position_sorted(list(filenames), probetype, indexre), **kwargs)
        self.probetype = probetype
        self.chunksize = chunksize
        self.chunkoverlap = chunkoverlap
        self.start = start
        self.stop = stop
        if self.stop is None:
            self.stop = len(self)

        # store channel index scheme conversion functions
        self.tdt_to_pos = probes.lookup_converter_function(probetype, \
                'tdt', 'pos')
        self.pos_to_tdt = probes.lookup_converter_function(probetype, \
                'pos', 'tdt')

    def seek_and_read(self, start, n):
        self.seek(start)
        return self.read(n)

    def chunk(self, overlap=None, start=None, stop=None, full=False):
        if start is None:
            start = self.start
        if stop is None:
            stop = self.stop
        if overlap is None:
            overlap = self.chunkoverlap

        self.seek(start)
        i = start
        if overlap == 0:
            while i + self.chunksize < stop:
                if full:
                    yield self.read(self.chunksize), i, i + self.chunksize
                else:
                    yield self.read(self.chunksize)
                i += self.chunksize
            if full:
                yield self.read(stop - i), i, stop
            else:
                yield self.read(stop - i)
        else:
            while i + self.chunksize + overlap < stop:
                if full:
                    yield self.read(self.chunksize + overlap), \
                            i, i + self.chunksize + overlap
                else:
                    yield self.read(self.chunksize + overlap)
                i += self.chunksize
                self.seek(i)
            if full:
                yield self.read(stop - i), i, stop
            else:
                yield self.read(stop - i)


class ICAReader(Reader):
    def __init__(self, icafilename=None, icakwargs=None, **kwargs):
        assert icafilename is not None, "No ica file supplied"
        assert icakwargs is not None, "No ica kwargs supplied"
        Reader.__init__(self, **kwargs)
        self.read = self.raw_read
        if not os.path.exists(icafilename):
            mm, um, self._cm, count, threshold = \
                    icapp.cmdline.process_src(self, **icakwargs)
            icapp.fio.save_ica(icafilename, mm, um, \
                    self._cm, self.filenames, count, threshold)
            self.seek(0)
        else:
            self._cm = icapp.fio.load_ica(icafilename, key='cm')
        self.read = self.ica_read

    def ica_read(self, n):
        print self._cm
        return icapp.ica.clean_data(self.raw_read, self._cm)

    def raw_read(self, n):
        return Reader.read(self, n)
