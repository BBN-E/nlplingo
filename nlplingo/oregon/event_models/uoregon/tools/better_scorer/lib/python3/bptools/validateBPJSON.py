"""
------------------------------  NOTICE  ------------------------------
This software (or technical data) was produced for the U.S. Government
under contract 2015-14120200002-002, and is subject to the Rights in
Data-General Clause 52.227-14 , ALT IV (MAY 2014) or (DEC 2007).

© 2019 The MITRE Corporation. All Rights Reserved.

----------------------------------------------------------------------
"""
import os, sys, re, string, codecs, json, argparse, glob
from jsonschema import validate
import jsonschema

import config
sysConfig = config.getConfig()

bpJsonSchemaName    = sysConfig[("BETTER_DATAPREP", "BPJSON_SCHEMA_NAME")].strip()
bpJsonSchemaFile    = config.getAbsolutePath(sysConfig[("BETTER_DATAPREP", "BPJSON_SCHEMA_FILE")].strip())
bpJsonVersion       = "v8f"
scriptVersion       = "v1 (2019-07-09)"
debugP            = False

def validateBPJSON (bpJsonSchemaFilename, bpJsonDataFilename):
    global debugP

    print("Loading BP JSON schema from {0}".format(bpJsonSchemaFilename))
    sys.stdout.flush()
    bpJsonSchema   = getStructFromJsonFile (bpJsonSchemaFilename)
    if debugP:
        print("bpJsonSchema:")
        print(bpJsonSchema)
    print("Loading BP JSON data file from {0}".format(bpJsonDataFilename))
    sys.stdout.flush()
    bpJsonInstance = getStructFromJsonFile (bpJsonDataFilename)
    if debugP:
        print("bpJsonInstance:")
        print(bpJsonInstance)
    print("validating ...".format(bpJsonDataFilename))
    sys.stdout.flush()

    try:
        validate(bpJsonInstance, bpJsonSchema)
        sys.stdout.write("This JSON is VALID relative to the schema in {0}\n".format(bpJsonSchemaFilename))
    except jsonschema.exceptions.ValidationError as ve:
        sys.stderr.write("This JSON is NOT VALID: {0}\n".format(str(ve)))
        sys.exit(1)

def getStructFromJsonFile (jsonFilename):
    global debugP
    if debugP:
        print("getStructFromFile {0} ...".format(jsonFilename))
    jsonFile = codecs.open(jsonFilename, 'r', encoding='UTF-8')
    jsonString = jsonFile.read()
    jsonFile.close()
    jsonStruct = json.loads(jsonString)
    return jsonStruct

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--schemafile", help="BPJSON schema file, default: {0}".format(bpJsonSchemaFile))
    argParser.add_argument("--datafile",   help="BPJSON data file to be validated")
    if len(sys.argv) == 1:
        argParser.print_help()
        sys.exit(1)
    args = argParser.parse_args()
    print("Executing {0}".format(sys.argv[0]))
    print("--schemafile: {0}".format(args.schemafile))
    print("--datafile:   {0}".format(args.datafile))
    if (args.datafile == None):
        print("Must provide --datafile argument")
        sys.exit(1)
    if (args.schemafile != None):
        bpJsonSchemaFile = args.schemafile
        print("schema file specified: {0}".format(bpJsonSchemaFile))
    sys.stdout.flush()
    validateBPJSON (bpJsonSchemaFile, args.datafile)
