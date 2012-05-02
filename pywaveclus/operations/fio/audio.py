#!/usr/bin/env python

import os
import re
#import warnings

#import numpy

#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    import scikits.audiolab

from ... import probe
#from ... import utils

import icapp


def position_sorted(fns, ptype, indexre):
    to_pos = probe.lookup_converter_function(ptype, 'tdt', 'pos')
    return sorted(fns, key=lambda fn: \
            to_pos(int(re.findall(indexre, os.path.basename(fn))[0])))


class Reader(icapp.fio.MultiAudioFile):
    def __init__(self, filenames=None, probetype='nna', \
            indexre=r'_([0-9]+)\#*', chunksize=441000, **kwargs):
        assert filenames is not None, "No filenames supplied to reader"
        # position sort filenames and create a the multiaudiofile
        icapp.fio.MultiAudioFile.__init__(self, \
                position_sorted(list(filenames), probetype, indexre), **kwargs)
        self.probetype = probetype
        self.chunksize = chunksize

        # store channel index scheme conversion functions
        self.tdt_to_pos = probe.lookup_converter_function(probetype, \
                'tdt', 'pos')
        self.pos_to_tdt = probe.lookup_converter_function(probetype, \
                'pos', 'tdt')

    def seek_and_read(self, start, n):
        self.seek(start)
        return self.read(n)


class ICAReader(Reader):
    def __init__(self, icafilename=None, icakwargs=None, **kwargs):
        assert icafilename is not None, "No ica file supplied"
        assert icakwargs is not None, "No ica kwargs supplied"
        Reader.__init__(self, **kwargs)
        if not os.path.exists(icafilename):
            mm, um, self._cm, count, threshold = \
                    icapp.cmdline.process_src(self, **icakwargs)
            icapp.fio.save_ica(icafilename, mm, um, \
                    self._cm, self.filenames, count, threshold)
            self.seek(0)
        else:
            self._cm = icapp.fio.load_ica(icafilename, key='cm')

    def read(self, n):
        return icapp.ica.clean_data(Reader.read(n), self._cm)
