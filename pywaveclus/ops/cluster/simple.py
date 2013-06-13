#!/usr/bin/env python
"""

_, p = scipy.stats.normaltest(fvs)
if p < 0.001:
    good feature!

Simple features

1) peak-to-peak: value at wf[pre] - trough
2) width
4) power (signal ** 2)

other
1) refractory period
2) just peak
3) just trough
"""


from . import pca


def features(wfs, p, n):
    # reshape to [index, wf]
    wfs = pca.stack_waveforms(wfs)
    pass
