#!/usr/bin/env python

import numpy as np

def find_crossings(data, threshold, direction = 'neg'):
    """
    Find the sub/super-threshold indices of data
    
    Parameters
    ----------
    data : 1d array
        Datapoints
    
    threshold: float
        Threshold (possibly calculated using calculate_threshold)
        Must be positive
    
    direction: string
        Must be one of the following:
            'pos' : find points more positive than threshold
            'neg' : find points more negative than -threshold
            'both': find points > threshold or < -threshold
    
    Returns
    -------
    crossings : 1d array
        Indices in data that cross the threshold
    """
    assert direction in ['pos','neg','both'], "Unknown direction[%s]" % direction
    assert threshold > 0, "Threshold[%d] must be > 0" % threshold
    if type(data) != np.ndarray: data = np.array(data)
    
    if direction == 'pos':
        return np.where(data > threshold)[0]
    if direction == 'neg':
        return np.where(data < -threshold)[0]
    if direction == 'both':
        return np.union1d(np.where(data > threshold)[0], np.where(data < -threshold)[0])