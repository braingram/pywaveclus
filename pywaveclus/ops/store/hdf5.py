#!/usr/bin/env python
"""
timerange
spikes (groups with spikes [by channel], and clustering info [by channel]
general detection/filtering/clustering/etc info [pickled filenode]
"""

import cPickle as pickle

import tables
import tables.nodes.filenode


def split_path(path):
    assert not (path in ('/', ''))
    assert path[0] == '/'
    ts = path.split('/')
    if len(ts) == 2:
        return '/', ts[-1]
    return '/'.join(ts[:-1]), ts[-1]


class SpikeStorage(object):
    def __init__(self, file, spikes_path='/spikes', info_path='/spike_info'):
        # table (ch<i>) filename in attrs
        # info (pickle file)
        # cluster info (ch<i>_info group per channel)
        self.file = file
        self._stdescription = None
        self._spikes_path = spikes_path
        self._info_path = info_path
        #if self._spikes_path not in self.file:
        #    s = self._spikes_path.split('/')
        #    rp = '/'.join(s[:-1])
        #    if (len(rp) == 0) or (rp[0] != '/'):
        #        rp = '/' + rp
        #    self.file.createGroup(rp, s[-1], createparents=True)
        #self._spikes_group = file.getNode(self._spikes_path)

    def convert_channel_index(self, index):
        # convert the channel index from one numbering system to another
        return index

    def set_time_range(self, tr):
        assert isinstance(tr, (tuple, list))
        assert len(tr) == 2
        self.file.getNode(
            self._spikes_path).setAttr('timerange', (tr[0], tr[1]))

    def get_time_range(self):
        return self.file.getNode(self._spikes_path).getAttr('timerange')

    time_range = property(get_time_range, set_time_range)

    #def channel_index_to_node_path(self, chi, full=False):
    #    if full:
    #        return '%s/ch%i' % (self._spikes_path, chi)
    #    return 'ch%i' % chi

    def create_spikes(self, chi, sis, sws):
        if not len(sws):
            return
        assert len(sis) == len(sws)
        # table
        if self._stdescription is None:
            class d(tables.IsDescription):
                channel = tables.UInt8Col()
                time = tables.Int64Col()
                wave = tables.Float64Col(shape=(len(sws[0])))
                cluster = tables.Int8Col()
            self._stdescription = d
        #nn = self.channel_index_to_node_path(chi)
        #np = self.channel_index_to_node_path(chi, full=True)
        if self._spikes_path not in self.file:
            r, n = split_path(self._spikes_path)
            n = self.file.createTable(r, n, self._stdescription)
        else:
            n = self.file.getNode(self._spikes_path)
        for (si, sw) in zip(sis, sws):
            n.row['channel'] = chi
            n.row['time'] = si
            n.row['wave'] = sw
            n.row['cluster'] = -1  # stand for 'not clustered'
            n.row.append()
        n.flush()

    def load_waves(self, chi):
        return self.file.getNode(self._spikes_path).readWhere(
            'channel == chi', field='wave')
        #n = self.file.getNode(self.channel_index_to_node_path(chi, full=True))
        #return n.cols.wave[:]

    def update_spikes(self, chi, clusters):
        t = self.file.getNode(self._spikes_path)
        assert len(clusters) == len(t.getWhereList('channel == chi'))
        for (i, r) in enumerate(t.where('channel == chi')):
            r['cluster'] = clusters[i]
            r.update()
        t.flush()
        #n = self.file.getNode(self.channel_index_to_node_path(chi, full=True))
        #assert len(n.cols.cluster) == len(clusters)
        #n.cols.cluster[:] = clusters

    def set_info(self, info):
        # pickle dict
        if self._info_path in self.file:
            self.file.removeNode(self._info_path)
        r, n = split_path(self._info_path)
        f = tables.nodes.filenode.newNode(self.file, where=r, name=n)
        #nn = '%s/info' % self._spikes_path
        #if nn in self.file:
        #    self.file.removeNode(nn)
        #f = tables.nodes.filenode.newNode(
        #    self.file, where=self._spikes_path, name='info')
        pickle.dump(info, f, protocol=2)
        f.close()
        self.file.flush()

    def get_info(self):
        f = tables.nodes.filenode.openNode(self._info_path, 'r')
        d = pickle.load(f)
        f.close()
        return d

    info = property(get_info, set_info)

    def close(self):
        self.file.flush()
        self.file.close()
