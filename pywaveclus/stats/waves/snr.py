#!/usr/bin/env python

import numpy


def ptp(waveforms, pre=20):
        """
        Measure signal to noise with equation from
        Edward A Branchard's thesis
        A control system for positioning recording electrodes to
        isolate neurons in extracellular recordings
        http://thesis.library.caltech.edu/2445/

        snr(i) = ptp(i) / rms(noise)

        from a brief look at the data
        make pre ~ 0.75 the pre-peak period
        """
        noise = numpy.sqrt(
            numpy.sum(waveforms[:, :pre].flatten() ** 2) /
            (len(waveforms) * pre))
        ptp = numpy.max(waveforms, 1) - numpy.min(waveforms, 1)
        return ptp / noise
