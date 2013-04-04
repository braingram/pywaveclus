#!/usr/bin/env python

import cPickle as pickle

import tables
import tables.nodes.filenode


class SpikeStorage(object):
    def __init__(self, file, spikes_path='/spikes'):
        # table (ch<i>) filename in attrs
        # info (pickle file)
        # cluster info (ch<i>_info group per channel)
        self.file = file
        self._stdescription = None
        self._spikes_path = spikes_path
        if self._spikes_path not in self.file:
            s = self._spikes_path.split('/')
            rp = '/'.join(s[:-1])
            if (len(rp) == 0) or (rp[0] != '/'):
                rp = '/' + rp
            self.file.createGroup(rp, s[-1], createparents=True)
        self._spikes_group = file.getNode(self._spikes_path)

    def save_timerange(self, start, end):
        # group attrs
        self._spikes_group._v_attrs['timerange'] = (start, end)

    def channel_index_to_node_path(self, chi, full=False):
        if full:
            return '%s/ch%i' % (self._spikes_path, chi)
        return 'ch%i' % chi

    def create_spikes(self, chi, sis, sws):
        if not len(sws):
            return
        assert len(sis) == len(sws)
        # table
        if self._stdescription is None:
            class d(tables.IsDescription):
                time = tables.Int64Col()
                wave = tables.Float64Col(shape=(len(sws[0])))
                cluster = tables.Int8Col()
            self._stdescription = d
        nn = self.channel_index_to_node_path(chi)
        np = self.channel_index_to_node_path(chi, full=True)
        if np not in self.file:
            n = self.file.createTable(self._spikes_path, nn,
                                      self._stdescription)
        else:
            n = self.file.getNode(np)
        for (si, sw) in zip(sis, sws):
            n.row['time'] = si
            n.row['wave'] = sw
            n.row['cluster'] = -1  # stand for 'not clustered'
            n.row.append()
        n.flush()

    def load_waves(self, chi):
        # table
        n = self.file.getNode(self.channel_index_to_node_path(chi, full=True))
        return n.cols.wave[:]

    def update_spikes(self, chi, clusters):
        n = self.file.getNode(self.channel_index_to_node_path(chi, full=True))
        assert len(n.cols.cluster) == len(clusters)
        n.cols.cluster[:] = clusters

    def save_cluster_info(self, chi, info):
        # pickle dict
        nn = self.channel_index_to_node_path(chi) + '_info'
        np = self.channel_index_to_node_path(chi, full=True) + '_info'
        if np in self.file:
            self.file.removeNode(np)
        f = tables.nodes.filenode.newNode(
            self.file, where=self._spikes_path, name=nn)
        pickle.dump(info, f)
        f.close()

    def save_filename(self, chi, fn):
        nn = self.channel_index_to_node_path(chi, full=True)
        if nn not in self.file:
            raise IOError(
                "table %s must be created before writing attributes" % nn)
        n = self.file.getNode(nn)
        n.setAttr('filename', fn)

    def save_info(self, info):
        # pickle dict
        nn = '%s/info' % self._spikes_path
        if nn in self.file:
            self.file.removeNode(nn)
        f = tables.nodes.filenode.newNode(
            self.file, where=self._spikes_path, name='info')
        pickle.dump(info, f, protocol=2)
        f.close()

    def close(self):
        self.file.flush()
        self.file.close()
