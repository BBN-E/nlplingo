"""
------------------------------  NOTICE  ------------------------------

This software (or technical data) was produced for the U.S. Government
under contract 2015-14120200002-002, and is subject to the Rights in
Data-General Clause 52.227-14 , ALT IV (MAY 2014) or (DEC 2007).

© 2019 The MITRE Corporation. All Rights Reserved.

----------------------------------------------------------------------
"""

# We use the Python configfile module.

import configparser, os, sys
from error import BetterDataPrepError

configDebugP   = False
configFileName = "bptools.config"

BETTER_TOOLS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Sam Bayer, author of this code originally for ML4XREQ MSR, said:
# I prefer to isolate the ConfigParser so that it doesn't raise error
# types from that package. A nice way of doing this is to make it look
# like a Python dictionary.

class BetterDataPrepConfig:

    def __init__(self, path):
        self.c = configparser.ConfigParser()
        self.c.read(path)

    def __getitem__(self, k):
        if (type(k) not in (list, tuple)) or (len(k) != 2):
            raise KeyError("key must be pair of section and option")
        try:
            return self.c.get(k[0], k[1])
        except configparser.Error as e:
            raise KeyError(str(e))

    def __setitem__(self, k, v):
        raise KeyError("read-only dictionary")

    def __delitem__(self, k):
        raise KeyError("read-only dictionary")

    def keys(self):
        allKeys = []
        for s in self.c.sections():
            allKeys += [(s, o) for o in self.c.options(s)]
        return allKeys

    def ensureMAT(self):
        # Maybe someone put it in their Python path.
        try:
            import MAT
        except ImportError:
            MAT_ROOT = None
            try:
                MAT_ROOT = self[("MAT", "MAT_ROOT")].strip()
            except KeyError:
                pass
            if not MAT_ROOT:
                raise BetterDataPrepError("No value for MAT_ROOT found in config file")        
            if os.path.isdir(os.path.join(MAT_ROOT, "src", "MAT", "lib", "mat", "python", "MAT")):
                MAT_ROOT = os.path.join(MAT_ROOT, "src", "MAT")
                if configDebugP:
                    print("config.py (debug enabled): MAT_ROOT set to {0}".format(MAT_ROOT))
            elif not os.path.isdir(os.path.join(MAT_ROOT, "lib", "mat", "python", "MAT")):
                if configDebugP:
                    print("here it is (debug enabled): {0}".format(os.path.join(MAT_ROOT, "lib", "mat", "python", "MAT")))
                raise BetterDataPrepError("MAT_ROOT is neither a CVS checkout nor a zip distribution of MAT")
            sys.path.insert(0, os.path.join(MAT_ROOT, "lib", "mat", "python"))
            try:
                import MAT
            except ImportError:
                raise BetterDataPrepError("couldn't import MAT in spite of finding a proper MAT installation")

    def ensureNLTK(self):
        try:
            import nltk
        except ImportError:
            NLTK_PATH = None
            try:
                NLTK_PATH = self[("NLTK", "NLTK_PATH")].strip()
            except KeyError:
                pass
            if not NLTK_PATH:
                raise CollabCoachError("No value for NLTK_PATH found in config file")
            sys.path.insert(0, NLTK_PATH)
            try:
                import nltk
            except ImportError:
                raise CollabCoachError("couldn't import nltk in spite of NLTK_PATH")
        # NLTK is imported. Make sure the data path is set if present.
        try:
            dataPath = self[("NLTK", "NLTK_DATA_DIR")].strip()
            if dataPath:
                if not dataPath.endswith(os.sep):
                    dataPath = dataPath + os.sep
                nltk.data.path.append(dataPath)
        except KeyError:
            pass

def getAbsolutePath(path):
    global _CONFIG, configDebugP
    if configDebugP:
        print("getAbsolutePath (debug enabled): {0} ...".format(path))
    if _CONFIG is None:
        getConfig()
    if (path == None):
        print("Cannot getAbsolutePath for None-valued path = {0}".format(path))
        return path
    if (path[0] == os.path.sep):
        print("getAbsolutePath -- path already absolute: {0}".format(path))
        return path
    rootAbs = _CONFIG[("LOCAL_PATHS", "INSTALL_ROOT")].strip()
    absPath = os.path.join(rootAbs,path)
    if configDebugP:
        print("getAbsolutePath (debug enabled); resolved absolute path to: {0}".format(absPath))
    return absPath


_CONFIG = None
    
def getConfig():
    global _CONFIG, configFileName
    if _CONFIG is None:
        configFile = os.path.join(BETTER_TOOLS_ROOT, os.path.join("config", configFileName))
        if not os.path.exists(configFile):
            raise BetterDataPrepError("Can't find config file; did you copy it from dataprep.config.in and populate it?")
        _CONFIG = BetterDataPrepConfig(configFile)
    return _CONFIG
