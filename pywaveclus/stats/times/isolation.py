#!/usr/bin/python

import numpy


def by_isi(spike_times, threshold=3, chunk_size=100, full=False):
    """
    Poor isolation (during the beginning part of the session) results in
    high isis. So to find the start of the isolated period

    1) calculate spike isis
    2) break up all isis into chunks (size = chunk_size)
    3) find the maximum isi for each chunk
    4) find the mean and std of the maximum isi for each chunk
    5) find the earliest chunk with an isi < mean + std * threshold
    6) return the time at the starting point of that chunk
    """
    if full:
        ret = lambda st, m: (st, m, threshold, chunk_size)
    else:
        ret = lambda st, m: st
    if len(spike_times) < (chunk_size * 10 + 1):
        return ret(spike_times[0], None)
    isi = spike_times[1:] - spike_times[:-1]
    nc = len(isi) / chunk_size
    misi = []
    for i in xrange(nc - 1):
        misi.append(isi[i * chunk_size:(i + 1) * chunk_size].max())
    h = len(misi) / 2
    m = numpy.mean(misi[h:])
    s = numpy.std(misi[h:])
    gi = numpy.where(misi < (m + s * threshold))[0]
    if len(gi) == 0:
        return ret(spike_times[0], misi)
    ci = gi.min() * chunk_size + 1
    if ci >= len(spike_times):
        return ret(spike_times[-1], misi)
    return ret(spike_times[ci], misi)
