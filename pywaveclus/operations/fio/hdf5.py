#!/usr/bin/env python

import logging

import tables

import pywaveclus


version_attribute_key = 'PYWAVECLUS_VERSION'
spike_table_name = 'spikes'


class HDF5Writer(object):
    def __init__(self, filename=None, pre=None, post=None, nfeatures=None):
        logging.debug("Opening file: %s" % filename)
        self._file = tables.openFile(filename, 'w')
        self.setup(pre, post, nfeatures)

    def setup(self, pre, post, nfeatures):
        st = self.make_spike_table_description(pre, post, nfeatures)
        self.spikes = self._file.createTable("/", spike_table_name, st)
        self.write_version()

    def make_spike_table_description(self, pre, post, nfeatures):
        class SpikeTableDescription(tables.IsDescription):
            time = tables.UInt64Col()
            channel = tables.UInt8Col()
            wave = tables.Float64Col(shape=(pre + post))
            cluster = tables.UInt8Col()
            features = tables.Float64Col(shape=(nfeatures))
        return SpikeTableDescription

    def write_version(self):
        v = pywaveclus.__version__
        self._file.setNodeAttr('/', version_attribute_key, v)

    def data_iter(self, data):
        for ch in data:
            for sp in ch:
                yield sp

    def channel_iter(self, data):
        for (chi, ch) in enumerate(data):
            for sp in ch:
                yield chi

    def setup_write(self, data):
        if len(self.spikes) == 0:
            # spike table has not yet been written to
            logging.debug("Writing channels to spike table")
            # write all channel indices
            #for (chi, r) in zip(self.channel_iter(data), self.spikes):
            #    r['channel'] = chi
            #    r.append()
            for chi in self.channel_iter(data):
                self.spikes.row['channel'] = chi
                self.spikes.row.append()
            self.spikes.flush()
            logging.debug("Wrote: %s channels" % len(self.spikes))
        else:
            total_length = sum([len(d) for d in data])
            if total_length != len(self.spikes):
                raise Exception("len(data)[%s] must == len(self.spikes)[%s]" \
                        % (total_length, len(self.spikes)))

    def write_data(self, data, key):
        if len(data) == 0:
            return
        self.setup_write(data)
        # Really pytables?... Really?
        # For some reason passing the table iterator through zip, izip, etc...
        # does not correctly update the rows. What's odd is that the returned
        # row iterator appears to point to the correct row, however
        # row.update has no effect.
        i = self.data_iter(data)
        for r in self.spikes:
            r[key] = i.next()
            r.update()
        self.spikes.flush()
        # Leave these next lines of code as a reminder of how badly
        # pytables treats it's row iterators
        #for (sp, r) in zip(self.data_iter(data), self.spikes.iterrows()):
        #    r[key] = sp
        #    r.update()
        #self.spikes.flush()

    def write_indices(self, indices):
        logging.debug("Writing spike indices")
        self.write_data(indices, 'time')

    def read_indices(self):
        pass

    indices = property(read_indices, write_indices)

    def write_waveforms(self, waveforms):
        logging.debug("Writing spike waveforms")
        self.write_data(waveforms, 'wave')

    def read_waveforms(self):
        pass

    waveforms = property(read_waveforms, write_waveforms)

    def write_features(self, features):
        logging.debug("Writing spike features")
        self.write_data(features, 'features')

    def read_features(self):
        pass

    features = property(read_features, write_features)

    def write_features_info(self, info):
        logging.debug("Writing feature info")
        self._file.createGroup('/', 'features_info', 'Features Info')
        for (i, fi) in enumerate(info):
            self.write_dictionary(fi, '/features_info', 'ch%i' % i, \
                    'Features Info')
        self._file.flush()

    def read_features_info(self):
        return self.read_dictionary('features_info')

    features_info = property(read_features_info, write_features_info)

    def write_clusters(self, clusters):
        logging.debug("Writing spike clusters")
        self.write_data(clusters, 'cluster')

    def read_clusters(self):
        pass

    clusters = property(read_clusters, write_clusters)

    #def write_cluster_info(self, info):
    #    logging.debug("Writing clustering info")
    ##    self._file.createGroup('/', 'cluster_info', 'Clustering Info')
    #    for (i, ci) in enumerate(info):
    #        self.write_dictionary(ci, '/cluster_info', 'ch%i' % i, \
    #                'Clustering Info')
    #    self._file.flush()

    #def read_cluster_info(self, info):
    #    self.read_dictionary(info, 'cluster_info')

    #cluster_info = property(read_cluster_info, write_cluster_info)

    def write_dictionary(self, data, root, group, description=None):
        if description is None:
            description = group
        group = self._file.createGroup(root, group, description)
        for k, v, in data.iteritems():
            self._file.createArray(group, k, v)
        self._file.flush()

    def read_dictionary(self, group):
        pass

    def close(self):
        self._file.close()
