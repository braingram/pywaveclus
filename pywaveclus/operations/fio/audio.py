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


def to_int(number, maximum):
    if number is None:
        return maximum
    elif isinstance(number, float):
        assert (number >= 0) and (number <= 1.0), \
            "to_int only accepts 0<=[%s]<=1.0" % number
        return int(number * maximum)
    elif isinstance(number, int):
        return number
    else:
        raise TypeError("Invalid Type[%s:%s] for to_int" %
                        (type(number), number))


class Reader(icapp.fio.MultiAudioFile):
    def __init__(self, filenames=None, probetype='nna',
                 indexre=r'_([0-9]+)\#*', chunksize=441000,
                 chunkoverlap=0, start=0, stop=None, **kwargs):
        assert filenames is not None, "No filenames supplied to reader"
        # alphabetically sort filenames and create a the multiaudiofile
        icapp.fio.MultiAudioFile.__init__(
            self, sorted(list(filenames), **kwargs))
        self.probetype = probetype
        self.chunksize = chunksize
        self.chunkoverlap = chunkoverlap
        self.start = to_int(start, len(self))
        self.stop = to_int(stop, len(self))
        #if self.stop is None:
        #    self.stop = len(self)

        # store channel index scheme conversion functions
        self.tdt_to_pos = probes.lookup_converter_function(
            probetype, 'tdt', 'pos')
        self.pos_to_tdt = probes.lookup_converter_function(
            probetype, 'pos', 'tdt')

    def seek_and_read(self, start, n):
        self.seek(start)
        return self.read(n)

    def chunk(self, overlap=None, start=None, stop=None, full=False):
        if start is None:
            start = self.start
        else:
            start = to_int(start)
        if stop is None:
            stop = self.stop
        else:
            stop = to_int(stop)
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
    def __init__(self, info, **kwargs):
        Reader.__init__(self, **kwargs)
        self.read = self.raw_read
        assert info['fns'] == kwargs['filenames'], \
            'ica fns must be the same (and ordered) as filenames: %s, %s' % \
            (info['fns'], kwargs['filenames'])
        self._cm = info['cm']
        self.read = self.ica_read

    def ica_read(self, n):
        #print self._cm
        return icapp.ica.clean_data(self.raw_read(n), self._cm)

    def raw_read(self, n):
        return Reader.read(self, n)
