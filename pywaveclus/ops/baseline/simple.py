#!/usr/bin/env python

from ... import utils


def simple(reader, trange):
    start, end = utils.parse_time_range(trange, 0, len(reader))
    reader.seek(start)
    return reader.read(end - start)
