#!/usr/bin/env python

import os

from nose.tools import eq_

import numpy as np

import scikits.audiolab

import pywaveclus

def test_reader():
    samplerate = 44100
    nchannels = 1
    nframes = 44100
    dtype = np.int16
    
    # make fake audio file
    if os.path.exists('/tmp/tmp.wav'): os.remove('/tmp/tmp.wav')
    af = scikits.audiolab.Sndfile('/tmp/tmp.wav','w',scikits.audiolab.Format('wav','pcm24'),nchannels,samplerate)
    data = np.arange(nframes, dtype=dtype)
    af.write_frames(data)
    af.close()
    
    # open reader
    reader = pywaveclus.data.audio.Reader('/tmp/tmp.wav', dtype)
    
    # test attributes
    eq_(reader.samplerate, samplerate)
    eq_(reader.nchannels, nchannels)
    eq_(reader.nframes, nframes)
    
    # test data read
    reader.seek(0)
    frames = reader.read_frames(100)
    assert np.all(frames == data[:100]), frames
    
    reader.seek(300)
    frames = reader.read_frames(100)
    assert np.all(frames == data[300:400])
    
    # test data chunking
    chunkI = 0
    for chunk, start, end in reader.chunk(0, 100, 10, 2):
        s = chunkI * 10
        e = s + 10 + 2
        if e > 100: e = 100
        
        assert np.all(chunk == data[s:e]), (chunk, data[s:e], s, e)
        chunkI += 1
    
    # test lock
    reader.lock = pywaveclus.utils.waiting_lock_dir('/tmp/pyc_test_lock')
    reader.seek(0)
    frames = reader.read_frames(100)
    assert np.all(frames == data[:100]), frames
    
    # cleanup
    if os.path.exists('/tmp/tmp.wav'): os.remove('/tmp/tmp.wav')
