#!/usr/bin/env python

import numpy

import pywaveclus

def test_sort_clusters():
    sc = pywaveclus.cluster.klustakwik.sort_clusters

    t = lambda a, b: all(sc(numpy.array(a)) == \
            numpy.array(b))

    # simple case
    assert(t([1,2,2,3], [0,1,1,2]))

    # skipped cluster
    assert(t([1,2,2,4], [0,1,1,2]))
    
    # reverse order
    assert(t([4,2,2,1], [2,1,1,0]))

    # all noise
    assert(t([1,1,1], [0,0,0]))

    # all signal
    assert(t([2,2,2], [1,1,1]))
    assert(t([3,3,3], [1,1,1]))

    # fail on any 0
    try:
        t([0, 1, 1], [2, 1,1])
    except ValueError:
        assert(True)
    except:
        assert(False)
