#!/usr/bin/env python


import audio
import hdf5

__all__ = ['audio', 'hdf5']


def to_number(string):
    if '.' in string:
        return float(string)
    else:
        return int(string)


def get_reader(files, cfg, section='reader', ica_section='ica'):
    kwargs = {}
    kwargs['filenames'] = files
    kwargs['probetype'] = cfg.get(section, 'probetype')
    kwargs['indexre'] = cfg.get(section, 'indexre')
    kwargs['chunksize'] = cfg.getint(section, 'chunksize')
    kwargs['chunkoverlap'] = cfg.getint(section, 'chunkoverlap')

    if cfg.has_option(section, 'start'):
        kwargs['start'] = to_number(cfg.get(section, 'start'))
    else:
        kwargs['start'] = 0

    if cfg.has_option(section, 'stop'):
        kwargs['stop'] = to_number(cfg.get(section, 'stop'))

    kwargs['icafilename'] = cfg.get(ica_section, 'filename')
    kwargs['icakwargs'] = {}
    kwargs['icakwargs']['method'] = cfg.get(ica_section, 'method')

    sargs = cfg.get(ica_section, 'sargs')
    sargs += ' %s' % kwargs['start']
    if 'stop' in kwargs:
        sargs += ' %s' % kwargs['stop']
    else:
        sargs += ' 1.'

    kwargs['icakwargs']['sargs'] = [to_number(i) for i in sargs.split()]
    kwargs['icakwargs']['ncomponents'] = cfg.getint(ica_section, 'ncomponents')
    kwargs['icakwargs']['count'] = cfg.getint(ica_section, 'count')
    if cfg.has_option(ica_section, 'threshold'):
        kwargs['icakwargs']['threshold'] = cfg.getfloat(ica_section, \
                'threshold')
    else:
        kwargs['icakwargs']['threshold'] = None
    kwargs['icakwargs']['stdthreshold'] = cfg.getfloat(ica_section, \
            'stdthreshold')
    return audio.ICAReader(**kwargs)


def get_writer(cfg, section='writer'):
    kwargs = {}
    for attr in ['filename', 'pre', 'post', 'nfeatures']:
        kwargs[attr] = cfg.get(section, attr)
    return hdf5.HDF5Writer(**kwargs)
