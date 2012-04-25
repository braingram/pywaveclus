#!/usr/bin/env python

import re
import warnings

import numpy

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import scikits.audiolab

from ... import probe
#from ... import utils


def position_sorted(fns, ptype, indexre):
    to_pos = probe.lookup_converter_function(ptype, 'audio', 'pos')
    return sorted(fns, key=lambda fn: \
            to_pos(int(re.match(indexre, fn).groups()[0])))


class Reader(object):
    def __init__(self, filenames=None, probetype='nna', dtype=numpy.float64, \
            indexre=r'[a-z,A-Z]+_([0-9]+)\#*'):
        assert filenames != None, "No filenames supplied to reader"
        self.dtype = dtype
        self.probetype = probetype
        self.filenames = position_sorted(list(filenames), probetype, indexre)
        self.files = [scikits.audiolab.Sndfile(fn) for fn in filenames]

        # store channel index scheme conversion functions
        self.audio_to_pos = probe.lookup_converter_function(probetype, \
                'audio', 'pos')
        self.pos_to_audio = probe.lookup_converter_function(probetype, \
                'pos', 'audio')

    def read_frames(self, start, nframes):
        """
        Read all channels
        """
        [f.seek(start) for f in self.filenames]
        return numpy.vstack([f.read_frames(nframes, self.dtype) \
                for f in self.filenames])


class ICAReader(Reader):
    def __init__(self, mixing_matrix_fn=None, unmixing_matrix_fn=None, \
            **kwargs):
        assert mixing_matrix_fn is not None, \
                "No mixing matrix filename supplied"
        assert unmixing_matrix_fn is not None, \
                "No unmixing matrix filename supplied"
        Reader.__init__(self, **kwargs)
        self.mixing_matrix_fn = mixing_matrix_fn
        self.mixing_matrix = numpy.matrix(numpy.loadtxt(mixing_matrix_fn))
        self.unmixing_matrix_fn = unmixing_matrix_fn
        self.unmixing_matrix = numpy.matrix(numpy.loadtxt(unmixing_matrix_fn))
        # pre-mulitply the unmixing and mixing matrices
        self.M = self.mixing_matrix * self.unmixing_matrix

    def read_frames(self, start, nframes):
        return numpy.array(self.M * Reader.read_frames(self, start, nframes))
