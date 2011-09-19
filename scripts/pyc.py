#!/usr/bin/env python
"""
Load and cluster a single file
"""

import logging, os, sys, time
from optparse import OptionParser
from contextlib import contextmanager

import numpy as np
import pylab as pl
import scikits.audiolab as al
from scipy.signal import resample

from pywaveclus import waveletfilter, detect, waveletfeatures, cluster, template

# ------------- these are just default files because I'm lazy -------------------
filename = '/Users/graham/Repositories/coxlab/physiology_analysis/data/clip.wav'
# filename = '/Users/graham/Repositories/coxlab/physiology_analysis/data/long.wav'
# filename = '/Users/graham/Repositories/coxlab/physiology_analysis/data/input_14#01.wav'
# filename = '/Users/graham/Repositories/coxlab/physiology_analysis/data/input_3#01.wav'

parser = OptionParser(usage="usage: %prog [options] recording.wav [output_directory]")
parser.add_option("-b", "--baselineTime", dest = "baselineTime",
                    help = "number of samples at beginning of recording used to calculate spike threshold",
                    default = 441000, type='int')
parser.add_option("-B", "--butter", dest = "butter",
                    help = "use a butterworth not wavelet filter",
                    default = False, action = "store_true")
parser.add_option("-c", "--chunkSize", dest = "chunkSize",
                    help = "data is processed in chunks (to reduce mem usage). Number of samples to read per chunk",
                    default = 4410000, type='int')
parser.add_option("-d", "--detectionDirection", dest = "detectionDirection",
                    help = "pos, neg, or both: detect spikes of a particular or both directions",
                    default = 'both')
parser.add_option("-e", "--plotext", dest = "plotext",
                    help = "plot file extension",
                    default = '.png')
parser.add_option("-E", "--neo", dest = "neo",
                    help = "use non-linear energy operator for detection",
                    default = False, action = "store_true")
parser.add_option("-f", "--nfeatures", dest = "nfeatures",
                    help = "number of features to measure per waveform",
                    default = 10, type='int')
parser.add_option("-H", "--filterMax", dest = "filterMax",
                    help = "maximum wavelet decomposition level for filtering (acts as a highpass)",
                    default = 6, type='int')
parser.add_option("-l", "--lockfile", dest = "lockfile",
                    help = "use a lockfile to prevent simultaneous disc access for >1 copy of this program",
                    default = None)
parser.add_option("-L", "--filterMin", dest = "filterMin",
                    help = "minimum wavelet decomposition level for filtering (acts as a lowpass)",
                    default = 4, type='int')
parser.add_option("-m", "--mat", dest = "mat",
                    help = "save results in a mat file",
                    default = False, action = "store_true")
parser.add_option("-M", "--minwidth", dest = "minwidth",
                    help = "minimum number of super threshold points to count a spike",
                    default = 2, type='int')
parser.add_option("-n", "--nthresh", type='float',
                    help = "n standard units used in threshold calculation",
                    default = 5)
parser.add_option("-N", "--nclusters", type='int',
                    help = "number of clusters to use",
                    default = 5)
parser.add_option("-o", "--chunkOverlap", dest = "chunkOverlap",
                    help = "number of samples to overlap chunks",
                    default = 4410, type='int') # TODO better explanation
parser.add_option("-O", "--oversample", dest = "oversample",
                    help = "oversample N times waveforms for extreme finding",
                    default = 4410, type='int') # TODO better explanation
parser.add_option("-p", "--plot", dest = "plot",
                    help = "generate lots of plots",
                    default = False, action = "store_true")
parser.add_option("-r", "--ref", dest = "ref",
                    help = "detection refractory period",
                    default = 44, type='int')
parser.add_option("-s", "--slide", dest = "slide",
                    help = "use a sliding threshold (recalculated for each chunk)",
                    default = False, action = "store_true")
parser.add_option("-S", "--slop", dest = "slop",
                    help = "allow N sub-threshold points during spike detection",
                    default = 0, type='int')
parser.add_option("-t", "--timerange", dest = "timerange",
                    help = "time range (in samples, slice format (start:end]) over which to process the file",
                    default = ':')
parser.add_option("-T", "--template", dest = "template",
                    help = "perform template matching for unclustered spikes using template algorithm [none, nn, center, ml, mahal]",
                    default = 'center')
parser.add_option("-v", "--verbose", dest = "verbose",
                    help = "enable verbose reporting",
                    default = False, action = "store_true")
parser.add_option("-w", "--prew", dest = "prew",
                    help = "number of samples prior to threshold crossing to store for each waveform",
                    default = 40, type='int')
parser.add_option("-W", "--postw", dest = "postw",
                    help = "number of samples after the threshold crossing to store for each waveform",
                    default = 88, type='int')
(options, args) = parser.parse_args()

def error(string, exception=Exception):
    logging.error(string)
    raise exception, string

if options.verbose:
    logging.basicConfig(level=logging.DEBUG)

if len(args) > 2:
    parser.print_usage()
    error("Too many arguments")

if len(args) == 0:
    logging.debug("No inputfile found, using default: %s" % filename)
    args = [filename,]

filename = args[0]

if len(args) == 2:
    outdir = args[1]
else:
    outdir = os.path.dirname(os.path.abspath(filename)) + '/pyc_' + os.path.basename(filename)

# construct output directory
if not os.path.exists(outdir):
    os.makedirs(outdir)

# start logging
logging.root.addHandler(logging.FileHandler('%s/pyc.log' % outdir, mode='w'))

def chunk(n, chunksize, overlap=0):
    """
    Chunk generator
    """
    for i in xrange((n/chunksize)+1):
        if (i * chunksize) >= n:
            return
        if ((i+1) * chunksize + overlap) < n:
            yield (i*chunksize, (i+1)*chunksize + overlap)
        else:
            yield (i*chunksize, n)


logging.debug("Opening file: %s" % filename)
af = al.Sndfile(filename)
logging.debug("\tSampling Rate: %i" % af.samplerate)
samplerate = af.samplerate
logging.debug("\tN Channels   : %i" % af.channels)
logging.debug("\tN Frames     : %i" % af.nframes)

# parse options.timerange
if not (':' in options.timerange):
    error("time range [%s] must contain ':'" % options.timerange, ValueError)
tps = options.timerange.split(':')
if len(tps) > 2:
    error("invalid time range [%s], contains too many ':'" % options.timerange, ValueError)
s, e = tps
if s.strip() == '':
    frameStart = 0
else:
    frameStart = int(tps[0])
if e.strip() == '':
    frameEnd = af.nframes
else:
    frameEnd = int(tps[1])
if frameEnd > af.nframes:
    logging.warning("End of timerange[%i] was > audio nframes[%i], truncating timerange" % (frameEnd, af.nframes))
    frameEnd = af.nframes
    #error("Invalid timerange: end [%i] > nframes [%i]" % (frameEnd, af.nframes), ValueError)
if frameEnd < frameStart: error("Invalid timerange: end [%i] < start [%i]" % (frameEnd, frameStart), ValueError)
nframes = frameEnd - frameStart
logging.debug("Time Range: %i - %i = %i" % (frameStart, frameEnd, nframes))

if af.channels != 1: error("Audio file did not contain only 1 channel [%i]" % af.channels, ValueError)
if options.baselineTime > nframes:
    logging.warning("BaselineTime [%i] was > nframes [%i]" % (options.baselineTime, nframes))
    options.baselineTime = nframes
if options.chunkSize > nframes:
    logging.warning("ChunkSize [%i] was > nframes [%i]" % (options.chunkSize, nframes))
    options.chunkSize = nframes

if not(options.lockfile is None):
    # TODO have this make a directory not a file, since on UNIX directories are atomic
    @contextmanager
    def waiting_file_lock(lock_file, delay=1):
        while os.path.exists(lock_file):
            logging.info("Found lock file, waiting to recheck in %d..." % delay)
            time.sleep(delay)
        f = open(lock_file, 'w')
        f.write("1")
        f.close()
        try:
            yield
        finally:
            # if os.path.exists(lock_file):
            os.remove(lock_file) # TODO: why does this sometimes not exist? is it a matter of 'flushing' writes?
else:
    @contextmanager
    def waiting_file_lock(lock_file, delay=1):
        yield

if not options.butter:
    logging.debug("Filter low-cutoff: %.3f" % waveletfilter.level_to_cutoffs(samplerate, options.filterMax)[0])
    logging.debug("Filter high-cutoff: %.3f" % waveletfilter.level_to_cutoffs(samplerate, options.filterMin)[1])

if options.slide:
    logging.debug("Using sliding threshold")
    threshold = 0.
else:
    # find threshold
    af.seek(frameStart)
    with waiting_file_lock(options.lockfile, 1):
        if options.butter:
            f = waveletfilter.butterfilter(af.read_frames(options.baselineTime, dtype=np.int16), flow=500, fhigh=5000) # default options for now
        else:
            f = waveletfilter.waveletfilter(af.read_frames(options.baselineTime, dtyp=np.int16), minlevel = options.filterMin, maxlevel = options.filterMax)
        if options.neo:
            threshold = detect.neo_calculate_threshold(f, n = options.nthresh)
        else:
            threshold = detect.calculate_threshold(f, n = options.nthresh)
        # np.savetxt('pyc.dat', f)
    logging.debug("Found threshold: %f" % threshold)

spikeindices = None
spikewaveforms = None

# thresholds = []

maxVal = 0.
minVal = 0.

for (s, e) in chunk(nframes, options.chunkSize, options.chunkOverlap):
    af.seek(s+frameStart)
    with waiting_file_lock('/tmp/pyc.lock', 1):
        d = af.read_frames(e-s, dtype=np.int16)
    if len(d) == 0:
        logging.warning("Read 0 frames from file??")
        continue
    
    # filter
    if options.butter:
        f = waveletfilter.butterfilter(d)
    else:
        f = waveletfilter.waveletfilter(d, minlevel = options.filterMin, maxlevel = options.filterMax)
    
    maxVal = max(f.max(), maxVal)
    minVal = min(f.min(), minVal)
    
    # detect
    if options.neo:
        # check threshold
        currentThreshold = detect.neo_calculate_threshold(f, n = options.nthresh)
        if abs(currentThreshold - threshold) > threshold*.5: logging.warning("Threshold dramatically changed: %f to %f" % (threshold, currentThreshold))
        if options.slide: threshold = currentThreshold
        si, sw = detect.neo_find_spikes(f, threshold, prew = options.prew, postw = options.postw)
    else:
        currentThreshold = detect.calculate_threshold(f, n = options.nthresh)
        if abs(currentThreshold - threshold) > threshold*.5: logging.warning("Threshold dramatically changed: %f to %f" % (threshold, currentThreshold))
        if options.slide: threshold = currentThreshold
        # si, sw = detect.find_spikes(f, threshold, direction = options.detectionDirection, prew = options.prew, postw = options.postw)
        si, sw = detect.find_spikes2(f, threshold, options.detectionDirection, options.prew, options.postw, options.ref, options.minwidth, options.slop, options.oversample)
    
    # thresholds.append((currentThreshold, s))
    
    si = np.array(si)
    sw = np.array(sw)
    
    # throw out any spikes that began in overlap
    goodspikes = np.where(si < (options.chunkSize + options.prew))[0]
    
    if len(goodspikes) == 0:
        continue
    
    if spikewaveforms is None:
        spikeindices = si[goodspikes] + s
        spikewaveforms = sw[goodspikes]
    else:
        spikeindices = np.hstack((spikeindices, si[goodspikes] + s))
        # print spikewaveforms.shape, sw.shape, goodspikes, si[goodspikes].shape
        spikewaveforms = np.vstack((spikewaveforms, sw[goodspikes]))
        logging.debug("%i%% done: %i spikes" % (int((e * 100.) / nframes), len(spikeindices)))

logging.debug("Max: %f  Min: %f" % (maxVal, minVal))

# cleanup
af.close()
del f, si, sw, goodspikes

# np.savetxt('thresholds',np.array(thresholds))

if spikeindices is None: # No spikes were found
    logging.warning("No spikes were found")
    spikeindices = []
    spikewaveforms = []
    clusters = []
    clusterindices = []
    cdata = []
    ctree = []
    # sys.exit(0)
elif len(spikeindices) < options.nfeatures: # not enough spikes to cluster, put them all in cluster 0
    nspikes = len(spikeindices)
    clusters = [range(nspikes),]
    for i in range(options.nclusters): clusters.append([])
    print clusters
    clusterindices = cluster.clusters_to_indices(clusters)
    cdata = []
    ctree = []
    
    spikefeatures = waveletfeatures.wavelet_features(spikewaveforms, nfeatures = options.nfeatures)
else:
    logging.debug("Found %i spikes" % len(spikeindices))
    # measure
    spikefeatures = waveletfeatures.wavelet_features(spikewaveforms, nfeatures = options.nfeatures)
    
    # cluster
    clusters, cdata, ctree = cluster.spc(spikefeatures, quiet = (not options.verbose),\
                                        nclusters = options.nclusters, minclus = nframes / float(samplerate))
    logging.debug("Found Clusters: %s" % (str([len(c) for c in clusters])))
    
    # template match
    if options.template != 'none':
        clusters = template.match(spikewaveforms, clusters, method = options.template)
        logging.debug("Post Template Clusters: %s" % (str([len(c) for c in clusters])))
    
    clusterindices = cluster.clusters_to_indices(clusters)

# save results in output directory
outfile = '/'.join((outdir, os.path.splitext(os.path.basename(filename))[0]))
if options.mat:
    outfile += '.mat'
    logging.debug("Saving results to %s" % outfile)
    # save as mat file
    import scipy.io
    scipy.io.savemat(outfile, {'times': spikeindices,
                                'waves': spikewaveforms,
                                'clus': clusterindices,
                                'cdata': cdata,
                                'ctree': ctree})
else: # save as hdf5 file
    import tables
    outfile += '.h5'
    logging.debug("Saving results to %s" % outfile)
    
    if len(clusters) > 255:
        raise ValueError("Too many clusters [%i] to save as Int8, rewrite to Int16" % len(clusters))
    
    class description(tables.IsDescription):
        time = tables.Int32Col()
        wave = tables.Float64Col(shape=(options.prew + options.postw,))
        clu = tables.Int8Col()
    
    hdfFile = tables.openFile(outfile,"w")
    spiketable = hdfFile.createTable("/","SpikeTable",description)
    
    logging.debug("writing to hdf5 file")
    for i in xrange(len(spikeindices)):
        spiketable.row['wave'] = spikewaveforms[i]
        spiketable.row['time'] = spikeindices[i]
        spiketable.row['clu'] = clusterindices[i]
        spiketable.row.append()
    
    spcgroup = hdfFile.createGroup('/', 'SPC', 'SPC results')
    hdfFile.createArray(spcgroup,'cdata',cdata)
    hdfFile.createArray(spcgroup,'ctree',ctree)
    
    logging.debug("closing hdf5 file")
    spiketable.flush()
    hdfFile.close()

if len(spikeindices) == 0:
    logging.warning("No spikes found, no cluster data or plots")
    sys.exit(0)

# # save other stuff as txt
# np.savetxt('/'.join((outdir, 'cdata')), cdata)
# np.savetxt('/'.join((outdir, 'ctree')), ctree)



if not options.plot:
    sys.exit(0)
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
# --------------------------------   Plotting   -----------------------------------
# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------

# - waveform plot(s)
# - spike times plot
# - features plot
# - cluster info plots
# - html template with information

from mpl_toolkits.mplot3d import Axes3D
colors = ['k','b','g','r','m','y']

# figures:
#  1 : spike times
#  2 : spike waveforms
#  3 : features
#  4 : cluster information

spikewaveforms = np.array(spikewaveforms)
spikeindices = np.array(spikeindices)

# if plotRawWave:
#     pl.figure(1)
#     af.seek(0)
#     pl.plot(waveletfilter.waveletfilter(af.read_frames(nframes), minlevel=options.filterMin, maxlevel=options.filterMax))

# if options.plotext is None:
#     systype = cluster.get_os()
#     if systype == 'linux':
#         options.plotext = '.svg'
#     elif systype == 'mac':
#         options.plotext = '.pdf'
#     else:
#         options.plotext = '.svg' # TODO: what is the best ext for windows?
#     logging.debug("plot file extension set to %s" % options.plotext)

# setup 3d subplot
pl.figure(1)
pl.suptitle("Spike Times")
pl.xlabel("Time(samples)")
pl.ylabel("Amplitude")

pl.figure(2, figsize=(len(clusters)*3,6))
pl.suptitle("Spike Waveforms")
pl.xlabel("Time(samples)")
pl.ylabel("Amplitude")

pl.figure(3)
try:
    ax = pl.gcf().add_subplot(2, 2, 3, projection='3d')
    ax.set_title("Clusters: %s" % str([len(c) for c in clusters]))
    use3d = True
except:
    use3d = False

nf = options.nfeatures

for (color, cl) in zip(colors,clusters):
    if len(cl) == 0: continue
    sw = spikewaveforms[cl]
    si = spikeindices[cl]
    # plot times
    pl.figure(1)
    se = sw[:,options.prew]
    pl.scatter(si, se, c=color, edgecolors=None)
    
    # waveforms
    pl.figure(2)
    pl.subplot(2,len(colors),colors.index(color)+1)
    pl.plot(np.transpose(sw), c=color)
    pl.title("N=%i" % len(sw))
    pl.subplot(2,len(colors),colors.index(color)+1+len(colors))
    av = np.average(sw,0)
    sd = np.std(sw,0,ddof=1)
    se = sd / np.sqrt(len(sw))
    pl.fill_between(range(len(av)), av+se*1.96, av-se*1.96, color=color, alpha=0.5)
    pl.plot(av, color=color)
    
    # features
    features = spikefeatures[cl]
    if use3d:
        ax.scatter(features[:,0],features[:,1],features[:,2],c=color) # on figure 3
    
    pl.figure(3)
    for x in xrange(nf):
        for y in xrange(nf):
            if x > y:
                pl.subplot(nf,nf,x+y*nf+1)
                pl.scatter(features[:,x], features[:,y], s=10, alpha=0.5, c=color, edgecolors=None)

# set axes for features figure (3)
pl.figure(3)
pl.suptitle("Waveform Features")
for x in xrange(nf):
    for y in xrange(nf):
        if x > y:
            pl.subplot(nf,nf,x+y*nf+1)
            pl.gca().set_xticks([])
            pl.gca().set_yticks([])
            b = 1/8.
            xr = spikefeatures[:,x].max() - spikefeatures[:,x].min()
            pl.xlim(spikefeatures[:,x].min()-xr*b, spikefeatures[:,x].max()+xr*b)
            yr = spikefeatures[:,y].max() - spikefeatures[:,y].min()
            pl.ylim(spikefeatures[:,y].min()-yr*b, spikefeatures[:,y].max()+yr*b)

if ctree != []:
    pl.figure(4)
    pl.suptitle("SPC Temperatures")
    pl.plot(ctree[:,4:])
    pl.legend(range(ctree[:,4:].shape[1]))
    pl.ylabel('Cluster Population')
    pl.xlabel('Temperature Step')
    # temp = cluster.spc_find_temperature(ctree, options.nclusters, nframes / float(samplerate))
    temp = cluster.spc_find_temperature_2(ctree, options.nclusters)
    pl.axvline(temp, color = 'k')

# pl.figure(4)
# pl.imshow(ctree, interpolation='nearest')
# pl.suptitle("Clustering Tree")
# pl.xlabel("Clusters")
# pl.ylabel("Time")

# pl.figure(5)
# isi = np.diff(spikeindices) / samplerate
# pl.hist(isi)

for (i, n) in enumerate(['times', 'waves', 'features', 'spc_temp']):
    pl.figure(i+1)
    pl.savefig('/'.join((outdir, n + options.plotext)))

# output html template with inform

if options.verbose:
    pl.show()
