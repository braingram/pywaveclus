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
        """
        noise = numpy.sqrt(numpy.sum(
            waveforms[:, :20].flatten() ** 2) / len(waveforms))
        ptp = numpy.max(waveforms, 1) - numpy.min(waveforms, 1)
        return ptp / noise
