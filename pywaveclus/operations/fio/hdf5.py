#!/usr/bin/env python

import tables


class HDF5Writer(object):
    def __init__(self, filename=None, pre=None, post=None, nfeatures=None):
        self._file = tables.openFile(filename, 'w')
        self.setup(pre, post, nfeatures)

    def setup(self, pre, post, nfeatures):
        st = self.make_spike_table_description(pre, post, nfeatures)
        self.spikes = self._file.createTable("/", "spikes", st)

    def make_spike_table_description(self, pre, post, nfeatures):
        class SpikeTableDescription(tables.IsDescription):
            time = tables.UInt64Col()
            channel = tables.UInt8Col()
            wave = tables.Float64Col(shape=(pre + post))
            cluster = tables.UInt8Col()
            features = tables.Float64Col(shape=(nfeatures))
        return SpikeTableDescription

    def data_iter(self, data):
        for ch in data:
            for sp in data:
                yield sp

    def channel_iter(self, data):
        for (chi, ch) in enumerate(data):
            for sp in data:
                yield chi

    def setup_write(self, data):
        if len(self.spikes == 0):
            # spike table has not yet been written to
            # write all channel indices
            for (chi, r) in zip(self.channel_iter(data, self.spikes)):
                r['channel'] = chi
                r.append()
            self.spikes.flush()
        else:
            if len(data.flat) != len(self.spikes):
                raise Exception("len(data)[%s] must == len(self.spikes)[%s]" \
                        % (len(data.flat), len(self.spikes)))

    def write_data(self, data, key):
        if len(data) == 0:
            return
        self.setup_write(data)
        for (sp, r) in zip(self.data_iter(data, self.spikes)):
            r[key] = sp
            r.update()
        self.spikes.flush()

    def write_indices(self, indices):
        self.write_data(indices, 'time')

    def read_indices(self):
        pass

    indices = property(read_indices, write_indices)

    def write_waveforms(self, waveforms):
        self.write_data(waveforms, 'wave')

    def read_waveforms(self):
        pass

    waveforms = property(read_waveforms, write_waveforms)

    def write_clusters(self, clusters):
        self.write_data(clusters, 'cluster')

    def read_clusters(self):
        pass

    clusters = property(read_clusters, write_clusters)

    def write_cluster_info(self, info):
        self.write_dictionary(info, 'cluster_info', 'Clustering Info')

    def read_cluster_info(self, info):
        self.read_dictionary(info, 'cluster_info')

    cluster_info = property(read_cluster_info, write_cluster_info)

    def write_dictionary(self, data, group, description=None):
        if description is None:
            description = group
        group = self._file.createGroup('/', group, description)
        for k, v, in data.iteritems():
            self._file.createArray(group, k, v)

    def read_dictionary(self, group):
        pass
