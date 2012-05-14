#!/usr/bin/env python

from .. import utils

import threshold

__all__ = ['threshold']


class ThresholdDetect(object):
    def __init__(self, baseline, nthresh, artifact, reader, filt, \
            direction, minwidth, ref):

        # calculate threshold
        start, end = utils.parse_time_range(baseline, 0, len(reader))
        reader.seek(start)
        self.thresh = threshold.calculate_threshold(\
                filt(reader.read(end - start)), nthresh)
        self.artifact = self.thresh / float(nthresh) * artifact

        self.direction = direction
        self.minwidth = minwidth
        self.ref = ref

    def __call(self, data):
        return self.find_spikes(data, self.thresh, self.artifact, \
                self.direction, self.minwidth, self.ref)


def get_detect(cfg, reader, filt, section='detect'):
    nthresh = cfg.getfloat(section, 'nthresh')
    artifact = cfg.getfloat(section, 'artifact')
    baseline = cfg.get(section, 'baseline')
    direction = cfg.get(section, 'direction')
    minwidth = cfg.getint(section, 'minwidth')
    ref = cfg.getint(section, 'ref')
    return ThresholdDetect(baseline, nthresh, artifact, reader, filt, \
            direction, minwidth, ref)
