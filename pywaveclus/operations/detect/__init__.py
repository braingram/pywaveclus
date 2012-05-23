#!/usr/bin/env python

import joblib

from ... import utils

import threshold

__all__ = ['threshold']


class ThresholdDetect(object):
    def __init__(self, baseline, nthresh, artifact, reader, filt, \
            direction, minwidth, ref, n_jobs):

        # calculate threshold
        start, end = utils.parse_time_range(baseline, 0, len(reader))

        reader.seek(start)
        fdata = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(filt)(ch) \
                for ch in reader.read(end - start))
        self.threshs = [threshold.calculate_threshold(fd, nthresh) \
                for fd in fdata]
        del fdata

        self.artifacts = [t / float(nthresh) * artifact for t in self.threshs]

        self.direction = direction
        self.minwidth = minwidth
        self.ref = ref

    def __call__(self, data, index):
        return threshold.find_spikes(data, self.threshs[index], \
                self.artifacts[index], self.direction, self.minwidth, self.ref)


def get_detect(cfg, reader, filt, section='detect'):
    nthresh = cfg.getfloat(section, 'nthresh')
    artifact = cfg.getfloat(section, 'artifact')
    baseline = cfg.get(section, 'baseline')
    direction = cfg.get(section, 'direction')
    minwidth = cfg.getint(section, 'minwidth')
    ref = cfg.getint(section, 'ref')
    n_jobs = cfg.getint(section, 'n_jobs')
    return ThresholdDetect(baseline, nthresh, artifact, reader, filt, \
            direction, minwidth, ref, n_jobs)
