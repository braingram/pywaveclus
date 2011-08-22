======
Python port of WaveClus Matlab spike sorting package
======

This python package is (mostly) a port of WaveClus to python.

What is WaveClus
------

Original WaveClus package:
http://vis.caltech.edu/~rodri/Wave_clus/Wave_clus_home.htm

reported in:
R. Quian Quiroga, Z. Nadasdy and Y. Ben-Shaul. Neural Computation 16, 1661-1687; 2004.

An excellent description of spike sorting can be found here:
http://www.scholarpedia.org/article/Spike_sorting

How does pywaveclus differ?
------

WaveClus used a elliptic type bandpass filter.

pywaveclus uses wavelet filtering based on code found here:
http://www.berkelab.org/Software.html

reported in:
A. B. Wiltschko, G. J. Gage and J. D. Berke. Journal of Neuroscience Methods 173:1, 34-40; 2008.

======
Installation
======

Requirements
-----
see requirements.txt for most up to date information

- numpy >= 1.5.1
- scipy >= 0.8.0
- pywavelets >= 0.2.0
- scikits.audiolab >= 0.11.0 (only for reading wav files in pyc.py)
- tables>=2.2.1 (only for saving hdf5 files in pyc.py)
- nose >= 0.11 (only for testing)

The library version numbers may be more strict than necessary.

It may be necessary to install these requirements prior to installing as I don't know if setup.py correctly installs missing requirements.

Installing
-----
python setup.py install

Testing
-----
python setup.py test

or

import pywaveclus.tests

pywaveclus.tests.test.run()

Running
-----
To cluster a single audio file run:

pyc.py <file>

pyc.py has lots of command line options (pyc.py -h) and will produce a hdf5 or mat file containing spike times, clusters, and waveforms.

The mat file contains index matched lists (times[0] corresponds to the same spike as waves[0] and clus[0], etc...):

- times : times of the spikes (in samples/frames)
- waves : waveforms of the spikes
- clus  : clusters to which the spikes belongs (cluster 0 is unmatched)

The hdf5 file contains a main table SpikeTable that contains a row for each spike and the following columns:

- time : time of the spike (in samples/frames)
- wave : waveform of the spike
- clu  : cluster to which the spike belongs (cluster 0 is unmatched)