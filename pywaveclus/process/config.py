#!/usr/bin/env python

# return detect, filter, cluster
import ConfigParser, io, logging, optparse, os, sys

CFGDEFAULTS = """
[main]
outputdir:
timerange:
plot:
verbose:
filename:
reference:
adjacentfiles:

[reader]
lockdir:
chunksize: 441000
chunkoverlap: 44100
dtype: f8

[detect]
method: threshold
nthresh: 4.5
artifact: 45.0
baseline: 100000
pre: 40
post: 88
direction: both
minwidth: 1
slop: 0
ref: 44
oversample: 10
sliding: False

[extract]
method: simple

[filter]
method: butter
low: 500
high: 5000
samprate: 44100
order: 3

[cluster]
#method: spc
#nclusters: 5
#nnoise: 2
#nfeatures: 10
#levels: 4
#wavelet: haar
method: klustakwik
nfeatures: 6
featuretype: pca
minclusters: 3
maxclusters: 5
minspikes: 12

[template]
method: none
"""
def load(customCfgFile = None):
    config = Config()
    config.read_user_config()
    if not customCfgFile is None: config.read(customCfgFile)
    return config

class Config(ConfigParser.SafeConfigParser):
    
    sections = ['main', 'reader', 'detect', 'extract', 'filter', 'cluster', 'template']
    
    def __init__(self, *args, **kwargs):
        ConfigParser.SafeConfigParser.__init__(self, *args, **kwargs)
        # read in defaults
        self.readfp(io.BytesIO(CFGDEFAULTS))
    
    def read_user_config(self, homeDir=os.getenv('HOME')):
        filename = '/'.join((homeDir,'.pywaveclus'))
        if os.path.exists(filename):
            logging.debug("Found user cfg: %s" % filename)
            self.read(filename)
        else:
            logging.warning('No user cfg found: %s' % filename)
    
    def read_commandline(self, options = None):
        """
        usage:
            <cmd> <inputfile> <options...>
        where options is a list of key value pairs and section changes
        
        If this function encounters a section header name then subsequent
            key value pairs will effect only the new section.
        
        Example
        -------
        options = ['timerange', '0:1000']
            will set main:timerange to '0:1000'
        options = ['filter', 'method', 'butter', 'low', '500', 'template', 'method', 'center']
            will set filter:method to butter, filter:low to 500 and template:method to center
        """
        
        if options is None:
            assert len(sys.argv) > 1, "No input file"
            self.set('main','filename',sys.argv[1])
            if len(sys.argv) > 2:
                options = sys.argv[2:]
            else:
                return
        
        section = 'main'
        key = None
        for (i, option) in enumerate(options):
            if option in self.sections: # change section
                section = option
                continue
            if key is None:
                key = option
                continue
            # option is value
            val = option.strip()
            if val[0] == '"': val = val[1:]
            if val[-1] == '"': val = val[:-1]
            self.set(section, key, val)
            key = None
        if not (key is None):
            raise AttributeError("Key [%s] missing value [section:%s]" % (key, section))
    
    def pretty_print(self, func=sys.stdout.writelines):
        for section in self.sections:
            func("[%s]" % section)
            for k, v in self.items(section):
                func("%s: %s" % (k, v))
