#!/usr/bin/env python

import numpy as np

import scikits.learn.decomposition

def features(waveforms, nfeatures = 6):
    # make input matrix
    waves = []
    for wave in waveforms:
        # for each repetition
        wf = np.array([])
        for ch in wave:
            wf = np.hstack((wf,ch))
        waves.append(wf)
    waves = np.array(waves)
    ica = scikits.learn.decomposition.FastICA(nfeatures)
    ica.fit(waves)
    return ica.get_mixing_matrix()
