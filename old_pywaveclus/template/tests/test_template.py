#!/usr/bin/env python

import logging

import numpy as np

import pywaveclus

def test_match():
    c1 = np.random.randn(10,3) * .1 + 1.0
    c2 = np.random.randn(10,3) * .05 + 2.0
    traces = np.vstack((c1,c2))
    clusters = [[19,], range(10), range(10,19)]
    
    clusters = pywaveclus.template.template.match(traces, clusters, method='nn', kmin = 5)
    r = [[], np.arange(10), np.arange(10,20)]
    c = clusters
    assert all([list(c[i]) == list(r[i]) for i in xrange(len(c))]), "nn:%s" % str(clusters)
    
    clusters = pywaveclus.template.template.match(traces, clusters, method='center')
    r = [[], np.arange(10), np.arange(10,20)]
    c = clusters
    assert all([list(c[i]) == list(r[i]) for i in xrange(len(c))]), "center:%s" % str(clusters)

def test_nearestneighbor():
    vs = np.mgrid[:5,:5][0] / 4. # 5x5, row values 0. -> 1.
    p = np.ones(5) * 0.5
    indices = pywaveclus.template.template.nearestneighbor(p, vs, 0.1)
    assert len(indices) == 1, "len(indices)[%i] should == 1" % len(indices)
    assert indices[0] == 2, "indices[0] = %i should == 2" % indices[0]

    indices = pywaveclus.template.template.nearestneighbor(p, vs, 0.6)
    assert len(indices) == 1, "len(indices)[%i] should == 1" % len(indices)
    assert indices[0] == 2, "indices[0] = %i should == 2" % indices[0]

    indices = pywaveclus.template.template.nearestneighbor(p, vs, 0.6, 3)
    assert len(indices) == 3, "len(indices)[%i] should == 3" % len(indices)
    assert indices[0] == 2, "indices[0] = %i should == 2" % indices[0]
    assert indices[1] in [1,3], "indices[1] = %i should == 1 or 3" % indices[1]
    assert indices[2] in [1,3], "indices[2] = %i should == 1 or 3" % indices[2]

    indices = pywaveclus.template.template.nearestneighbor(p, vs, 0.6, 5)
    assert len(indices) == 3, "len(indices)[%i] should == 3" % len(indices)
    assert indices[0] == 2, "indices[0] = %i should == 2" % indices[0]
    assert indices[1] in [1,3], "indices[1] = %i should == 1 or 3" % indices[1]
    assert indices[2] in [1,3], "indices[2] = %i should == 1 or 3" % indices[2]

def test_nn():
    c1 = np.random.randn(10,3) * .1 + 1.0
    c2 = np.random.randn(10,3) * .05 + 2.0
    traces = np.vstack((c1,c2))
    clusters = [[19,], range(10), range(10,19)]
    matches = pywaveclus.template.template.nn(traces, clusters, kmin = 5)
    assert len(matches) == 1, "len(matches)[%i] should be == 1" % len(matches)
    assert matches[0] == 2, "matches[%i] should be == 2" % matches[0]

def test_center():
    c1 = np.random.randn(10,3) * .1 + 1.0
    c2 = np.random.randn(10,3) * .05 + 2.0
    traces = np.vstack((c1,c2))
    clusters = [[19,], range(10), range(10,19)]
    matches = pywaveclus.template.template.center(traces, clusters)
    assert len(matches) == 1, "len(matches)[%i] should be == 1" % len(matches)
    assert matches[0] == 2, "matches[%i] should be == 2" % matches[0]