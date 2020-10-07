"""
------------------------------  NOTICE  ------------------------------
This software (or technical data) was produced for the U.S. Government
under contract 2015-14120200002-002, and is subject to the Rights in
Data-General Clause 52.227-14 , ALT IV (MAY 2014) or (DEC 2007).

© 2019 The MITRE Corporation. All Rights Reserved.

----------------------------------------------------------------------
"""
import os, sys, re, string, codecs, json, argparse, glob
from datetime import datetime

import config
sysConfig = config.getConfig()

debugP = False
quietP = False

bpJsonSchemaName    = sysConfig[("BETTER_DATAPREP", "BPJSON_SCHEMA_NAME")].strip()
bpJsonSchemaFile    = config.getAbsolutePath(sysConfig[("BETTER_DATAPREP", "BPJSON_SCHEMA_FILE")].strip())
bpJsonVersion       = "v8f"

if (bpJsonSchemaName != bpJsonVersion):
    print("Potential problem: config file declares bpJsonSchemaName to be {0}".format(bpJsonSchemaName))
    print("but current bpjson API module is for {0}".format(bpJsonVersion))

validAnnotationRoles      = ["reference", "system"]
validAnnotatorClasses     = ["mturk", "mitre", "system", "pretagger-v1", "multiple"]
validAnnotationProcedures = ["pretagger-v1", "mturk", "mat", "system-test", "multiple"]
entryCounter              = 0

def constructCorpus (corpusID, annotatorID, annotatorClass, annotationRole, annotationProcedure):
    global bpJsonVersion, debugP
    if debugP:
        print("constructCorpus {0} {1} {2} {3} {4}".format(corpusID, annotatorID, annotatorClass, annotationRole, annotationProcedure))
    corpus = dict()
    corpus["format-type"]    = "bp-corpus"
    corpus["format-version"] = bpJsonVersion
    corpus["corpus-id"]      = corpusID
    corpus["provenance"]     = constructCorpusProvenance (annotatorClass, annotationProcedure, annotatorID, annotationRole)
    corpus["entries"]        = dict()
    return corpus

def constructMinimalCorpus (corpusID, annotationProcedure):
    global bpJsonVersion, debugP
    if debugP:
        print("constructMinimalCorpus {0} ...".format(corpusID))
    corpus = dict()
    corpus["format-type"]    = "bp-corpus"
    corpus["format-version"] = bpJsonVersion
    corpus["corpus-id"]      = corpusID
    corpus["provenance"]     = constructMinimalCorpusProvenance ("mitre", annotationProcedure, "reference")
    corpus["entries"]        = dict()
    return corpus

def constructAbstractEventEntry (entryID, docID, sentID, evtID, sentText, anchorString, agentStrings, patientStrings, provenance):
    global eventCounter, spanSetCounter
    eventCounter   = 0
    spanSetCounter = 0
    entry = {"entry-id":     entryID,
             "segment-text": sentText,
             "segment-type": "sentence",
             "provenance":   provenance,
             "doc-id":       docID,
             "sent-id":      sentID}
    anchorSpanSet  = constructSpanSet ([anchorString])
    agentSpanSet   = constructSpanSet (agentStrings)
    patientSpanSet = constructSpanSet (patientStrings)
    event          = constructEventWithID (evtID, anchorSpanSet, agentSpanSet, patientSpanSet)
    events         = {event["eventid"]: event}
    spanSets       = {anchorSpanSet["span-set-id"]:  anchorSpanSet,
                      agentSpanSet["span-set-id"]:   agentSpanSet,
                      patientSpanSet["span-set-id"]: patientSpanSet}
    abstractEvents = {"events": events, "span-sets": spanSets}
    annotSets      = {"abstract-events": abstractEvents}
    entry["annotation-sets"] = {"abstract-events": abstractEvents}
    return entry

def constructAbstractEventAnchorsEntry (entryID, docID, sentID, eventsList, spanSetsList, sentText, provenance):
    global eventCounter, spanSetCounter
    ## print "constructAbstractEventAnchorsEntry events = {0}".format(eventsList)
    eventCounter   = 0
    spanSetCounter = 0
    entry = {"entry-id":     entryID,
             "segment-text": sentText,
             "segment-type": "sentence",
             "provenance":   provenance,
             "doc-id":       docID,
             "sent-id":      sentID}
    eventsDict = {}
    spansDict  = {}
    for i in range(len(eventsList)):
        event         = eventsList[i]
        spanSet       = spanSetsList[i]
        eventsDict[event["eventid"]] = event
        spansDict[spanSet["span-set-id"]] = spanSet
    ## print "events:   {0}".format(eventsDict)
    ## print "spansets: {0}".format(spansDict)
    abstractEvents = {"events": eventsDict, "span-sets": spansDict}
    annotSets      = {"abstract-events": abstractEvents}
    entry["annotation-sets"] = {"abstract-events": abstractEvents}
    return entry

### "annotation-date": "the date on which the annotation was proposed",
### "annotation-procedure": "(inherited from corpus annotator-type) version of the pretagger, 'mturk', 'mat'",
### "annotator-class": "crowd or mitre or system",
### "annotator-id": "workerid or name of a mitre person or system",
### "hit-id": "35F6NGNVM8JBG5KLX9CLGR6V2D7T73 -optional, used if it came from mturk", 
### "source": "a file name naming the output of the pretagger, mturk or mat such as Batch_237916_batch_results.csv"
###
###   {"annotation-date":    {"type": "string"},
###    "annotator-class":    {"type": "string", "enum": ["pretagger-v1", "mturk", "mitre", "system"]},
###    "annotator-id":       {"type": "string"},
###    "hit-id":             {"type": "string"},
###    "source":             {"type": "string"},

def constructEntryProvenance (annotatorClass, annotatorID, hitID, source, annotationProcedure, annotDate):
    global debugP
    prov = {}
    if (annotatorClass != None):
        prov["annotator-class" ]     = annotatorClass
    if (annotatorID != None):
        prov["annotator-id"]         = annotatorID
    if (hitID != None):
        prov["hit-id"]               = hitID
    if (source != None):
        prov["source"]               = source
    if (annotationProcedure != None):
        prov["annotation-procedure"] = annotationProcedure
    if (annotDate != None):
        prov["annotation-date"]      = annotDate
    return prov

def constructCorpusProvenance (annotatorClass, annotationProcedure, annotatorID, annotationRole):
    global debugP
    prov = {}
    import time
    prov["corpus-creation-date"] = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if (annotatorClass != None):
        prov["annotator-class"]      = annotatorClass
    if (annotationProcedure != None):
        prov["annotation-procedure"] = annotationProcedure
    if (annotatorID != None):
        prov["annotator-id"]         = annotatorID
    if (annotationRole != None):
        prov["annotation-role"]      = annotationRole
    return prov

def constructMinimalCorpusProvenance (annotatorClass, annotationProcedure, annotationRole):
    global debugP
    prov = {}
    import time
    prov["corpus-creation-date"] = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if (annotatorClass != None):
        prov["annotator-class"]      = annotatorClass
    if (annotationProcedure != None):
        prov["annotation-procedure"] = annotationProcedure
    if (annotationRole != None):
        prov["annotation-role"]      = annotationRole
    return prov

def constructEventProvenance (eventInXml):
    global debugP
    provenance = {"event-in-xml": eventInXml}
    return provenance

def addToEntryProvenance (entry, attrName, attrValue):
    provenance = {}
    if ("provenance" in entry):
        provenance = entry["provenance"]
    else:
        entry["provenance"] = provenance
    provenance[attrName] = attrValue
    return provenance

def addToProvenance (provenance, attrName, attrValue):
    provenance[attrName] = attrValue
    return provenance

eventCounter = 0

def newEventID ():
    global eventCounter
    eventCounter += 1
    eventID = "event{0}".format(eventCounter)
    return eventID

def constructEvent (anchorSpanSet, agentSpanSet, patientSpanSet):
    eventID = newEventID()
    return constructEventWithID (eventID, anchorSpanSet, agentSpanSet, patientSpanSet)

def constructEventWithID (eventID, anchorSpanSet, agentSpanSet, patientSpanSet):
    if (not isinstance(eventID, str)):
        eventID = newEventID()
    event = {"eventid": eventID}
    event["anchors"]  = anchorSpanSet["span-set-id"]
    if (len(agentSpanSet) != 0):
        event["agents"]   = [agentSpanSet["span-set-id"]]
    else:
        event["agents"]   = []
    if (len(patientSpanSet) != 0):
        event["patients"] = [patientSpanSet["span-set-id"]]
    else:
        event["patients"] = []
    return event

spanSetCounter = 0

def constructSpanSet (spanStrings):
    global spanSetCounter
    spanSetCounter += 1
    spanSetID = "spanset-{0}".format(spanSetCounter)
    spanSet = {"span-set-id": spanSetID, "spans": []}
    for spanString in spanStrings:
        spanSet["spans"].append({"string": spanString})
    return spanSet

def constructDocEntry (docFilename, docID, docText, annotationProcedure, annotatorClass):
    global debugP
    if debugP:
        print("constructDocEntry docID = {0}".format(docID))
    doc = dict()
    doc["segment-type"]  = "document"
    doc["segment-text"]  = docText
    doc["entry-id"]      = docID
    doc["provenance"]    = constructEntryProvenance(annotatorClass, None, None, docFilename, annotationProcedure, None)
    return doc

def constructAbstractSentEntry (sentText, docID, annotationProcedure, sentStartInDoc, sentEndInDoc, annotatorClass, annotatorID):
    global debugP
    if debugP:
        print("constructAbstractSentStruct docID = {0}".format(docID))
    sent                    = dict()
    sent["entry-id"]        = "{0}_{1}-{2}".format(docID, sentStartInDoc, sentEndInDoc)
    sent["segment-type"]    = "sentence"
    sent["segment-text"]    = sentText
    sent["annotation-sets"] = {}
    sent["provenance"]      = constructEntryProvenance(annotatorClass, annotatorID, None, docID, annotationProcedure, None)
    sent["provenance"]["start-in-source"] = sentStartInDoc
    sent["provenance"]["end-in-source"]   = sentEndInDoc
    return sent


def constructNewAbstractEntry (corpus, docID, sentID, sentText):
    global debugP, entryCounter
    if debugP:
        print("constructAbstractSentStruct docID = {0} sentID = {1} ".format(docID, sentID))
    entryCounter += 1
    entry                    = dict()
    entryID                  = "{0}_{1}_{2}".format(docID, sentID, entryCounter)
    entry["entry-id"]        = entryID
    entry["segment-type"]    = "sentence"
    entry["segment-text"]    = sentText
    entry["doc-id"]          = docID
    entry["sent-id"]         = sentID
    abstractAnnots           = {"events": {}, "span-sets": {}}
    entry["annotation-sets"] = {"abstract-events": abstractAnnots}
    ## entry["provenance"]   = {}
    entryTable               = getEntries(corpus)
    entryTable[entryID]      = entry
    return entry

def addFullEventToAbstractAnnots (abstractAnnots, quadMV, quadHH, anchorString, agentStringSets, patientStringSets):
    eventID = getNewEventID (abstractAnnots["events"])
    return addFullEventToAbstractAnnots (abstractAnnots, eventID, quadMV, quadHH, anchorString, agentStringSets, patientStringSets)
    
def getNewEventID (eventTable):
    eventIndex = len(eventTable) + 1
    eventID    = "event{0}".format(eventIndex)
    while (eventID in eventTable):
        eventIndex += 1
        eventID     = "event{0}".format(eventIndex)
    return eventID

def addFullEventToAbstractAnnots (abstractAnnots, eventID, quadMV, quadHH, anchorStrings, agentStringSets, patientStringSets):
    ## print ("addFullEventToAbstractAnnots anchorStrings = {0} agentStringSets = {1} patientStringSets = {2}".format(
    ##      anchorStrings, agentStringSets, patientStringSets))
    spanSetTable = abstractAnnots["span-sets"]
    agentIDs = []
    if (len(agentStringSets) > 0):
        for agentStringSet in agentStringSets:
            spanSetID = findOrMakeSpanSet (spanSetTable, agentStringSet)
            agentIDs.append(spanSetID)
    patientIDs = []
    if (len(patientStringSets) > 0):
        for patientStringSet in patientStringSets:
            spanSetID = findOrMakeSpanSet (spanSetTable, patientStringSet)
            patientIDs.append(spanSetID)
    event = {"eventid":         eventID,
             "material-verbal": quadMV,
             "helpful-harmful": quadHH,
             "agents":          agentIDs,
             "patients":        patientIDs}
    if (len(anchorStrings) > 0):
        event["anchors"] = findOrMakeSpanSet (spanSetTable, anchorStrings)
    ## print ("event = {0}".format(event))
    eventTable          = abstractAnnots["events"]
    eventTable[eventID] = event
    return event

def findOrMakeSpanSet (spanSetTable, spanStrings):
    for spanSetID, spanSet in spanSetTable.items():
        spanSetStringsSet      = set([spanElement["string"].lower() for spanElement in spanSet["spans"]])
        proposedSpanStringsSet = set([spanString.lower for spanString in spanStrings])
        if (spanSetStringsSet == proposedSpanStringsSet):
            return spanSetID
    return makeNewSpanSet (spanSetTable, spanStrings)

def makeNewSpanSet (spanSetTable, spanStrings):
    ## print ("makeNewSpanSet {0} {1}".format(spanSetTable, spanStrings))
    spanSetIndex = len(spanSetTable) + 1
    spanSetID    = "ss-{0}".format(spanSetIndex)
    while (spanSetID in spanSetTable):
        spanSetIndex += 1
        spanSetID     = "ss-{0}".format(spanSetIndex)
    spans = []
    for spanString in set(spanStrings):
        spanElement = {"string": spanString}
        spans.append(spanElement)
    spanSetTable[spanSetID] = {"spans": spans}
    return spanSetID

def setAbstractEventsInSentEntry (sentEntry, spanSetTable, eventsTable, directionalCorefTable):
    absEvents = {"span-sets":         spanSetTable,
                 "directional-coref": directionalCorefTable,
                 "events":            eventsTable}
    sentEntry["annotation-sets"]["abstract-events"] = absEvents
    return sentEntry

def addSpanSetToEntry (entry, spanSet):
    global debugP
    if debugP:
        print("addSpanSetToEntry entry={0}  spanSet={1}".format(entry, spanSet))
    annotationSets = entry["annotation-sets"]
    if debugP:
        print("annotationSets: {0}".format(annotationSets))

    abstractEvents = annotationSets["abstract-events"]
    if debugP:
        print("abstractEvents: {0}".format(abstractEvents))
    spanSets       = abstractEvents["span-sets"]
    if debugP:
        print("spanSets:       {0}".format(spanSets))
    spanSetID      = spanSet["span-set-id"]
    if debugP:
        print("spanSetID:      {0}".format(spanSetID))
    spanSets[spanSetID] = spanSet
    return entry

def addEntry (corpus, sentEntry):
    global debugP
    if debugP:
        print("addEntry corpus-id={0} entry-id={1} segment-type={2}".format(
            corpus["corpus-id"], sentEntry["entry-id"], sentEntry["segment-type"]))
    corpus["entries"][sentEntry["entry-id"]] = sentEntry
    return corpus

def writeStructToJsonFile (dataStruct, jsonOutFilename):
    global debugP
    if debugP:
        print("converting structure into JSON string ...")
    sys.stdout.flush()
    structAsJson = json.dumps (dataStruct, indent=2, sort_keys=True, ensure_ascii=False)
    if debugP:
        print("writing JSON string to file {0} ...".format(jsonOutFilename))
    sys.stdout.flush()
    bpOut = codecs.open(jsonOutFilename, 'w', encoding='UTF-8')
    bpOut.write(structAsJson)
    bpOut.flush()
    bpOut.close()

def getStructFromJsonFile (jsonFilename):
    global debugP, quietP
    if (not quietP):
        print("getStructFromFile {0} ...".format(jsonFilename))
    jsonFile = codecs.open(jsonFilename, 'r', encoding='UTF-8')
    jsonString = jsonFile.read()
    jsonFile.close()
    jsonStruct = json.loads(jsonString)
    if (not quietP):
        if ("entries" in jsonStruct):
            print("Loaded {0} entries.".format(len(jsonStruct["entries"])))
        else:
            print("Loaded.")
    return jsonStruct

def constructSpanSetWithID (spanSetID, mentionList):
    spanSet = {"span-set-id": spanSetID,
               "spans":       mentionList}
    return spanSet

def addMentionToSpanSet (spanSet, mention):
    spanSet["spans"].append(mention)

def constructMention (span, startOffset, endOffset, synClass):
    mention = {}
    mention["string"]   = span
    mention["start"]    = startOffset
    mention["end"]      = endOffset
    mention["synclass"] = synClass
    return mention

def constructAbstractEventOnly (eventID, materialVerbalClass, helpfulHarmfulClass, agentIdList, patientIdList, anchorsID):
    event = {"eventid": eventID,
             "material-verbal": materialVerbalClass,
             "helpful-harmful": helpfulHarmfulClass,
             "agents":          agentIdList,
             "patients":        patientIdList,
             "anchors":         anchorsID}
    return event

def getEntries (corpus):
    if ("entries" in corpus):
        return corpus["entries"]
    print ("No 'entries' entry in corpus -- return empty table")
    return {}

## Alternative to previous, with explicit indication of argument type
def getCorpusEntries (corpus):
    return getEntries (corpus)

def getCorpusEntry (corpus, entryID):
    entries = getCorpusEntries(corpus)
    if (entryID not in entries):
        print("Cannot find entry-id {0} in corpus entries")
        return None
    else:
        return entries[entryID]

def getEntryID (entry):
    if ("entry-id" not in entry):
        print("Cannot find entry-id on entry")
        print("entry = {0}".format(entry))
        return "(entryid unknown)"
    else:
        return entry["entry-id"]

def getEntryAnnotatorID (entry):
    ## print "getEntryAnnotatorID on {0}".format(entry)
    if ("provenance" not in entry):
        print("Cannot find annotator-id on entry {0} -- no provenance".format(getEntryID(entry)))
        return "(annotator-id unknown)"
    else:
        provenance = entry["provenance"]
        if ("annotator-id" not in provenance):
            print("Cannot find annotator-id in provenance of entry {0}".format(getEntryID(entry)))
            return "(annotator-id unknown)"
        else:
            ## print "entryAnnotatorID = {0}".format(provenance["annotator-id"])
            return provenance["annotator-id"]

def setEntryAnnotatorID (entry, annotatorID):
    provenance = {}
    if ("provenance" in entry):
        provenance = entry["provenance"]
    provenance["annotator-id"] = annotatorID

def setEntryProcedure (entry, procedure):
    provenance = {}
    if ("provenance" in entry):
        provenance = entry["provenance"]
    provenance["procedure"] = procedure

def getEntrySegmentType (entry):
    if ("segment-type" in entry):
        return entry["segment-type"]
    else:
        return "(Unknown)"

def getEntryDocID_SentID (entry):
    return "{0}_{1}".format(getEntryDocID(entry), getEntrySentID(entry))

def getEntrySentID (entry):
    if (("segment-type" in entry) and (entry["segment-type"] == "sentence")):
        if ("sent-id" in entry):
            return entry["sent-id"]
        else:
            print("Cannot find sent-id in entry {0}".format(getEntryID (entry)))
            return "(sent-id unknown)"
    else:
        print("Entry {0} is not identified as having segment-type = sentence".format(getEntryID (entry)))
        return "(Unknown)"

def entrySegmentTypeIsSentence (entry):
    if (("segment-type" in entry) and (entry["segment-type"] == "sentence")):
        return True
    else:
        return False

def getEntrySentenceText (entry):
    if (("segment-type" in entry) and (entry["segment-type"] == "sentence")):
        if ("segment-text" in entry):
            return entry["segment-text"]
        else:
            print("Cannot find segment-text in entry {0}".format(getEntryID (entry)))
            return "(segment-text unknown)"
    else:
        print("Entry {0} is not identified as having segment-type = sentence".format(getEntryID (entry)))
        return "(Unknown)"

def getEntryDocID (entry):
    if ("doc-id" in entry):
        return entry["doc-id"]
    elif (("entry-id" in entry) and (entry["entry-id"].find("_") > 0)):
        entryID = entry["entry-id"]
        docID   = entryID[0:entryID.find("_")]
        return docID
    else:
        ## print("Cannot find doc-id for entry {0}".format(getEntryID (entry)))
        return "(doc-id unknown)"

def getEntrySentID (entry):
    if ("sent-id" in entry):
        return entry["sent-id"]
    elif (("entry-id" in entry) and (entry["entry-id"].find("_") > 0)):
        entryID = entry["entry-id"]
        sentID  = entryID[entryID.find("_")+1:]
        if (sentID.find("_") > 0):
            sentID = entryID[0:entryID.find("_")]
            return sentID
        else:
            return sentID
    else:
        print("Cannot find sent-id for entry {0}".format(getEntryID (entry)))
        return "(Unknown)"

def entryHasAbstractEventsP (entry):
    if ("annotation-sets" in entry):
        if ("abstract-events" in entry["annotation-sets"]):
            if ("events" in entry["annotation-sets"]["abstract-events"]):
                if (len(entry["annotation-sets"]["abstract-events"]["events"]) > 0):
                    return True
    return False

def getAbstractEventAnnotationSet (entry):
    return entry["annotation-sets"]["abstract-events"]

def getEntryAbstractEventTable (entry):
    eventAnnotationSet = getAbstractEventAnnotationSet (entry)
    abstractEventTable = getEventTable (eventAnnotationSet)
    return abstractEventTable

def getEntryAbstractEventsTable (entry):
    if ("annotation-sets" not in entry):
        return []
    annotSets = entry["annotation-sets"]
    if ("abstract-events" not in annotSets):
        return []
    abstractEvents = annotSets["abstract-events"]
    if ("events" not in abstractEvents):
        return []
    return abstractEvents["events"]

def setEntryAbstractEventsTable (entry, events):
    annotSets = {}
    if ("annotation-sets" in entry):
        annotSets = entry["annotation-sets"]
    abstractEvents = {}
    if ("abstract-events" in annotSets):
        abstractEvents = annotSets["abstract-events"]
    abstractEvents["events"] = events
    return events

def getEventTable (abstractEventAnnotSet):
    return abstractEventAnnotSet["events"]

def getEntrySpanSetTable (entry):
    eventAnnotationSet = getAbstractEventAnnotationSet (entry)
    abstractSpansTable = getSpanSetTable (eventAnnotationSet)
    return abstractSpansTable

def getSpanSetTable (abstractEventAnnotSet):
    return abstractEventAnnotSet["span-sets"]

def getSpanSetStrings (spanSetTable, spanSetID):
    spanSet = spanSetTable[spanSetID]
    spanStrings = []
    for span in spanSet["spans"]:
        spanStrings.append(span["string"])
    return spanStrings

def getSpanSetFirstString (spanSetTable, spanSetID):
    return getSpanSetStrings (spanSetTable, spanSetID)[0]

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

def getSentenceAlignedEntries (entryTable):
    docID_sentID_table = {}
    for entryID, entry in entryTable.items():
        docID_sentID = getEntryDocID_SentID (entry)
        if (docID_sentID in docID_sentID_table):
            docID_sentID_table[docID_sentID].append(entryID)
        else:
            docID_sentID_table[docID_sentID] = [entryID]
    return docID_sentID_table

def mergeCorporaFiles (newCorpusID, corporaFiles):
    newCorpus  = constructCorpus (newCorpusID, "none", "none", "none", "none")
    provenance = newCorpus["provenance"]
    newEntries = getEntries (newCorpus)
    for corpusFile in corporaFiles:
        print ("about to merge corpus in BP JSON file {0}".format(corpusFile))
        inCorpus = getStructFromJsonFile (corpusFile)
        if ("provenance" in inCorpus):
            updateProvenance (provenance, inCorpus["provenance"])
        corpusEntries = getEntries(inCorpus)
        for entryID, entry in corpusEntries.items():
            if (entryID in newEntries):
                print ("Ensuring entry index is unique: old entryID = {0}".format(entryID))
                entryID = getUniqueEntryID (entryID, entry, newEntries)
                print ("new entryID = {0}".format(entryID))
            newEntries[entryID] = entry
    print ("new corpus has {0} merged entries".format(len(newEntries)))
    return newCorpus

def getUniqueEntryID (entryID, entry, entryTable):
    suffix = 0
    newEntryID = entryID + "_decon-{0}".format(suffix)
    while newEntryID in entryTable:
        suffix += 1
        newEntryID = entryID + "_decon-{0}".format(suffix)
    return newEntryID

def getNewEntryID (entry, entryTable):
    suffix = 0
    newEntryID = entry["doc-id"] + "_" + entry["sent-id"] + "_" + str(suffix)
    while newEntryID in entryTable:
        suffix += 1
        newEntryID = entry["doc-id"] + "_" + entry["sent-id"] + "_" + str(suffix)
    return newEntryID

def updateProvenance (provenance, provInfoToAdd):
    for provKey, provVal in provInfoToAdd.items():
        if ((provKey not in provenance) or (provenance[provKey] in ["", "none"])):
            provenance[provKey] = provVal
        else:
            provKeyCounter = 0
            for pkey, pval in provenance.items():
                if (pkey.find(provKey) == 0):
                    provKeyCounter += 1
            provKeyIncr             = provKey + "-" + str(provKeyCounter)
            provenance[provKeyIncr] = str(provVal)
    return provenance

