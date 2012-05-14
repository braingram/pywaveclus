#!/usr/bin/env python


import audio
import hdf5

__all__ = ['audio', 'hdf5']


def get_reader(files, cfg, section='reader', ica_section='ica'):
    kwargs = {}
    kwargs['filenames'] = files
    kwargs['probetype'] = cfg.get(section, 'probetype')
    kwargs['indexre'] = cfg.get(section, 'indexre')
    kwargs['chunksize'] = cfg.getint(section, 'chunksize')
    kwargs['chunkoverlap'] = cfg.getint(section, 'chunkoverlap')

    kwargs['icafilename'] = cfg.get(ica_section, 'filename')
    kwargs['icakwargs'] = {}
    kwargs['icakwargs']['method'] = cfg.get(ica_section, 'method')
    kwargs['icakwargs']['sargs'] = [int(i) \
            for i in cfg.get(ica_section, 'sargs').split()]
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
    pass
