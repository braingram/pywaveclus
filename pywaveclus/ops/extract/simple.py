#!/usr/bin/env python


def simple(data, indices, pre, post):
    waves = []
    for i in indices:
        if (i - pre < 0) or (i + post > data.size):
            waves.append(None)
        else:
            wave = data[i - pre: i + post]
            if len(data) == pre + post:
                waves.append(wave)
            else:
                waves.append(None)
            del wave
    return waves


def old_simple(readers, indices, ffunc, pre, post):
    #find_extreme = utils.find_extreme(direction)
    waves = []
    main = readers[0]
    for index in indices:
        wave = []
        start = index - pre
        length = pre + post
        main.seek(start)
        data = ffunc(main.read_frames(length))
        if len(data) != pre + post:
            del data
            continue  # check length
        wave.append(data)
        del data
        #for reader in readers[1:]:
        for reader_index in xrange(1, len(readers)):
            readers[reader_index].seek(start)
            wave.append(ffunc(readers[reader_index].read_frames(length)))
        waves.append(wave)
        del wave
    return waves
