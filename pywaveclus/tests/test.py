#!/usr/bin/env python

from ..cluster import *
from ..detect import *
from ..waveletfeatures import *
from ..waveletfilter import *

def run(regex=r"^test_"):
    """
    This dangerously looks for all callable objects in globals that match
    a regular expression, and checks if these objects execute without 
    any exceptions
    
    Parameters
    ----------
    regex : string
        Regular experession used to test if the object should be tested
    
    Returns
    -------
    ntests : int
        Number of tests run
    fails : dict
        Dictionary of failed test results
            key = function name, value = (error number, error string)
    """
    import re
    ntests = 0
    fails = {}
    for name in globals():
        if re.match(regex, name):
            f = globals()[name]
            if hasattr(f, '__call__'):
                ntests += 1
                logging.debug("---- Running test: %s ----" % name)
                try:
                    f.__call__()
                    logging.debug("---- Test Passed ----")
                except Exception as (errno, strerror):
                    logging.error("!!!! Test Failed: %s = %i : %s !!!!" % (name, errno, strerror))
                    fails[name] = (errno, strerror)
    logging.debug("================")
    logging.debug("Ran %i tests" % ntests)
    logging.debug("%i tests failed" % len(fails))
    for fail in fails:
        logging.error("\t%s failed with %i : %s" % (fail, fails[fail][0], fails[fail][1]))
    return ntests, fails

if __name__ == '__main__':
    run()
        