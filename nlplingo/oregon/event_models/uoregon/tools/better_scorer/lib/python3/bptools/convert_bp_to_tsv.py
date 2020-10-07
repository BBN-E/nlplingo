"""
------------------------------  NOTICE  ------------------------------
This software (or technical data) was produced for the U.S. Government
under contract 2015-14120200002-002, and is subject to the Rights in
Data-General Clause 52.227-14 , ALT IV (MAY 2014) or (DEC 2007).

© 2019 The MITRE Corporation. All Rights Reserved.

----------------------------------------------------------------------
"""
import argparse, sys, os, glob, codecs, json, copy

from config import getConfig
config = getConfig()
import bpjson

eventsTable      = None
entitiesTable    = None

def convertBP2TSV (bpjsonFilename, tsvFilename):
    print("reading from {0} ...".format(bpjsonFilename))
    sys.stdout.flush()
    bpCorpus = bpjson.getStructFromJsonFile (bpjsonFilename)
    print("writing to {0} ...".format(tsvFilename))
    sys.stdout.flush()
    tsvFile  = codecs.open(tsvFilename, 'w', encoding='UTF-8')
    tsvFile.write("sent-index\tfile\tcorpus-id\tdoc-id\tsent-id\tannotator-id\tevent-id\tmaterial-verbal\thelpful-harmful\tanchor\tagents\tpatients\tsentence\n")
    corpusID = bpCorpus["corpus-id"]
    lineCount = 1
    sentCount = 0
    ## to use later: viewJson2=json.dumps(JSON,indent=2,ensure_ascii=False).encode('utf8')
    prevDocID_sentID = ""
    for entryID, entry in bpCorpus["entries"].items():
        docID  = entry["doc-id"]
        sentID = entry["sent-id"]
        annotatorID = "(Unknown)"
        thisDocID_sentID = docID + "_" + sentID
        if (thisDocID_sentID != prevDocID_sentID):
            sentCount += 1
        if (("provenance" in entry) and ("annotator-id" in entry["provenance"])):
            annotatorID = entry["provenance"]["annotator-id"]
        if (entry["segment-type"] == "sentence"):
            events       = entry["annotation-sets"]["abstract-events"]["events"]
            spanSets     = entry["annotation-sets"]["abstract-events"]["span-sets"]
            sentenceText = entry["segment-text"]
            for eventID, event in events.items():
                tsvFile.write(str(sentCount) + "\t")
                tsvFile.write(bpjsonFilename + "\t")
                tsvFile.write(corpusID + "\t")
                tsvFile.write(docID  + "\t")
                tsvFile.write(sentID + "\t")
                tsvFile.write(annotatorID + "\t")
                ## print ("eventID {0}".format(eventID))
                tsvFile.write(eventID + "\t")
                ## print ("hh: {0}".format(renderMaterialVerbal(event)))
                tsvFile.write(renderMaterialVerbal(event) + "\t")
                ## print ("hh: {0}".format(renderHelpfulHarmful(event)))
                tsvFile.write(renderHelpfulHarmful(event) + "\t")
                anchorID = event["anchors"]
                anchorString  = renderSpanSet(anchorID, spanSets)
                ## print ("anchor: {0}".format(anchorString))
                tsvFile.write(anchorString + "\t")
                for argType in ["agents", "patients"]:
                    argVals  = event[argType]
                    argCount = len(argVals)
                    if (argCount == 0):
                        tsvFile.write("\t")
                    else:
                        argStrings = [renderSpanSet(argID, spanSets) for argID in argVals]
                        for i in range(argCount):
                            argString = argStrings[i]
                            tsvFile.write(argString)
                            if (i < (argCount - 1)):
                                tsvFile.write(",")
                        tsvFile.write("\t")
                if (sentenceText == None):
                    print ("NOTE!! No sentence text found for entryID {0} docID {1} sentID {2}".format(entryID, docID, sentID))
                    tsvFile.write("<<<No Sentence Text>>>\n")
                else:
                    tsvFile.write(sentenceText.replace("\n"," ").replace("\r"," ").replace("\t"," ") + "\n")
                lineCount += 1
    tsvFile.flush()
    tsvFile.close()
    print("Finished writing {0} lines to {1}".format(lineCount, tsvFilename))

def renderHelpfulHarmful (event):
    if ("helpful-harmful" in event):
        return event["helpful-harmful"]
    else:
        return "(Empty)"

def renderMaterialVerbal (event):
    if ("material-verbal" in event):
        return event["material-verbal"]
    else:
        return "(Empty)"

def renderSpanSet (entityID, spanSets):
    if (entityID not in spanSets):
        print("ERROR: entityID {0} could not be found.".format(entityID))
        return "<Entity {0} No Spans Found>".format(entityID)
    spanSet = spanSets[entityID]
    pieces  = ["<"]
    spanCount = len(spanSet["spans"])
    for i in range(spanCount):
        span = spanSet["spans"][i]
        pieces.append(span["string"])
        if (i < (spanCount - 1)):
            pieces.append("|")
    pieces.append(">")
    entityRep = ''.join(pieces)
    return entityRep

if __name__ == "__main__":
    print("Executing {0}".format(sys.argv[0]))
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--bpjson", help="BP JSON input")
    argParser.add_argument("--tsv",    help="Simple Tab-Separated-Values format for displaying Abstract Event annotations")
    args = argParser.parse_args()
    print("--bpjson: {0}".format(args.bpjson))
    print("--tsv:    {0}".format(args.tsv))
    if len(sys.argv) == 1:
        argParser.print_help()
        sys.exit(1)
    convertBP2TSV (args.bpjson, args.tsv)
    print("Done.")
