"""
------------------------------  NOTICE  ------------------------------

This software (or technical data) was produced for the U.S. Government
under contract 2015-14120200002-002, and is subject to the Rights in
Data-General Clause 52.227-14 , ALT IV (MAY 2014) or (DEC 2007).

© 2019 The MITRE Corporation. All Rights Reserved.

----------------------------------------------------------------------

Script last modified: 2019-11-15 -- added Kuhn-Munkres optimization

"""
import argparse, sys, os, glob, codecs, json, copy, itertools
from datetime import datetime

from munkres import Munkres, make_cost_matrix

import config
sysConfig = config.getConfig()
import bpjson
from validateBPJSON import validateBPJSON

debugP           = False
spanMatchDebugP  = False
entryMatchDebugP = False
alignmentDebugP  = False
possAlignDebugP  = False
entrySetScoringP = False

## Scoring Policies
argMatchPolicies       = ['strict', 'loose', 'ignore']
argMatchPolicy         = 'strict'
quadClassMatchPolicies = ['strict', 'loose', 'ignore']
quadClassMatchPolicy   = 'strict'
anchorMatchPolicies    = ['strict', 'loose', 'ignore']
anchorMatchPolicy      = 'ignore'

validateP           = False
bpJsonSchemaName    = sysConfig[("BETTER_DATAPREP", "BPJSON_SCHEMA_NAME")].strip()
bpJsonSchemaFile    = config.getAbsolutePath(sysConfig[("BETTER_DATAPREP", "BPJSON_SCHEMA_FILE")].strip())

reportFilename      = None
reportFile          = None
tsvFilename         = None
tsvFile             = None
jsonFilename        = None
quietP              = False
onlySummaryP        = False
abstractKeyInAnnotationSets = "abstract-events"

def scoreSysFileAgainstRefFile (reffile, sysfile):
    global debugP, quietP, refFilename, sysFilename, refCorpus, validateP
    print("scoreSysFileAgainstRefFile {0} {1}".format(reffile, sysfile))
    print ("Execution timestamp: {0}".format(datetime.now().strftime("%Y-%m-%d_%H-%M")))
    sys.stdout.flush()
    entrySet = buildEntrySetFromSysAndRefFiles (reffile, sysfile)
    evaluateAbstractEvents (entrySet)

    print ("------------- displaying score for entry SETs -----------------------")
    print (" sys file: {0}\n ref file: {1}".format(sysfile, reffile))
    sys.stdout.flush()
    entrySet.scoreStruct.computeScore()
    entrySet.scoreStruct.displayMeasures(sys.stdout)
    if (reportFile != None):
        reportFile.write("------------- displaying score for entry SETs -----------------------\n")
        reportFile.write(" sys file: {0}\n ref file: {1}\n".format(sysfile, reffile))
        entrySet.scoreStruct.displayMeasures(reportFile)
        reportFile.flush()
        reportFile.close()
    summarizeEntrySetScore (entrySet)

def summarizeEntrySetScore (entrySet):
    global reportFile, tsvFilename
    if (tsvFilename != None):
        print("Writing TSV score report to {0}".format(tsvFilename))
        sys.stdout.flush()
        tsvFile     = codecs.open(tsvFilename, 'w', encoding='UTF-8')
        addTsvHeaders(tsvFile)
        for entryPair in entrySet.entryPairs:
            addLineToTSV (entryPair.sysEntryID, entryPair.refEntryID, entryPair.scoreStruct, tsvFile)
        addLineToTSV (entrySet.sysFilename, entrySet.refFilename, entrySet.scoreStruct, tsvFile)
        tsvFile.flush()
        tsvFile.close()
        print("Finished writing TSV score report to {0}".format(tsvFilename))
        sys.stdout.flush()

def buildEntrySetFromSysAndRefFiles (reffile, sysfile):
    global debugP, quietP, entryMatchDebugP
    print ("buildEntrySetFromSysAndRefFiles ...")
    sys.stdout.flush()
    entrySet = EntrySet(reffile, sysfile)
    filterEmptyStringsFromArguments (entrySet.refCorpus)
    filterEmptyStringsFromArguments (entrySet.sysCorpus)
    refEntries = bpjson.getEntries (entrySet.refCorpus)
    sysEntries = bpjson.getEntries (entrySet.sysCorpus)
    ## Construct EntryPairs for this EntrySet
    unpairedRefCount = 0
    unpairedSysCount = 0
    for refEntryID, refEntry in refEntries.items():
        ## if debugP:
        ##     print("ref entryID = {0}".format(refEntryID))
        if (refEntry["segment-type"] == "sentence"):
            sysEntry = findPairedEntry (refEntry, sysEntries, "ref", "sys")
            if (sysEntry == None):
                print ("unmatched REF entryID = {0} docID = {1} sentID = {2}".format(
                        refEntryID, bpjson.getEntryDocID (refEntry), bpjson.getEntrySentID (refEntry)))
                sys.stdout.flush()
                entrySet.entryPairs.append(EntryPair(refEntry, None, entrySet))
                unpairedRefCount += 1
            else:
                entryPair = EntryPair(refEntry, sysEntry, entrySet)
                entrySet.entryPairs.append(entryPair)
                if entryMatchDebugP:
                    print ("found paired entry EntryPair id={0}".format(entryPair.id))
                    sys.stdout.flush()
    for sysEntryID, sysEntry in sysEntries.items():
        ## if debugP:
        ##     print("sys entryID = {0}".format(sysEntryID))
        if (sysEntry["segment-type"] == "sentence"):
            refEntry = findPairedEntry (sysEntry, refEntries, "sys", "ref")
            if (refEntry == None):
                print ("unmatched SYS entryID = {0} docID = {1} sentID = {2}".format(
                        sysEntryID, bpjson.getEntryDocID (sysEntry), bpjson.getEntrySentID (sysEntry)))
                sys.stdout.flush()
                entrySet.entryPairs.append(EntryPair(None, sysEntry, entrySet))
                unpairedSysCount += 1
    if entryMatchDebugP:
        print ("{0} entryPairs in entrySet".format(len(entrySet.entryPairs)))
    if (not quietP): ## (not quietP):
        print(" EntrySet ref: {0}  sys: {1} paired entries: {2}".format(reffile, sysfile, len(entrySet.entryPairs)))
        print("    Unpaired entries: ref={0} sys={1}".format(unpairedRefCount, unpairedSysCount))
    sys.stdout.flush()
    return entrySet

def findPairedEntry (probeEntry, targetEntries, probeType, targetType):
    global debugP, entryMatchDebugP
    docID  = bpjson.getEntryDocID (probeEntry)
    sentID = bpjson.getEntrySentID (probeEntry)
    foundEntries = []
    if entryMatchDebugP:
        print ("Looking with probe {2} docID {0} sentID {1}".format(docID, sentID, probeType))
        sys.stdout.flush()
    for targetEntryID, targetEntry in targetEntries.items():
        if ((docID  == bpjson.getEntryDocID  (targetEntry)) and
            (sentID == bpjson.getEntrySentID (targetEntry))):
            foundEntries.append(targetEntry)
    if (len(foundEntries) > 1):
        print (" {0} {1} entries match {2} docID {3} and sentID {4}!!".format(
                len(foundEntries), targetType, probeType, docID, sentID))
        sys.stdout.flush()
    elif (len(foundEntries) == 0):
        if entryMatchDebugP:
            print ("{0} -- no paired {1} entry".format(probeType, targetType))
            sys.stdout.flush()
        return None
    return foundEntries[0]

def filterEmptyStringsFromArguments (bpCorpus):
    global debugP
    if debugP:
        print("filtering empty strings from corpus...")
        sys.stdout.flush()
    if ("entries" in bpCorpus):
        for entryId, entry in bpCorpus["entries"].items():
            if ("annotation-sets" in entry):
                annotationSets = entry["annotation-sets"]
                if ("abstract-events" in annotationSets):
                    abstractEvents = annotationSets["abstract-events"]
                    if ("events" in abstractEvents):
                        for evtId, evt in abstractEvents["events"].items():
                            for argType in ["agents", "patients"]:
                                if ("" in evt[argType]):
                                    print("resetting {0} {1} = {2}".format(evtId, argType, evt[argType]), end=' ')
                                    evt[argType] = []
                                    print("to {0}".format(evt[argType]))
                                    sys.stdout.flush()

                    
def evaluateAbstractEvents (entrySet):
    global debugP, alignmentDebugP
    if debugP:
        print("%%%%%% evaluateAbstractEvents ...")
        sys.stdout.flush()
    for entryPair in entrySet.entryPairs:
        if (not onlySummaryP):
            print ("-----------------------------------------------------------------------------")
            print ("TS={0} Evaluating entryPair id={1} doc={2} sent={3} sysid={4} refid={5}".format(
                    datetime.now().strftime("%H:%M:%S"),
                    entryPair.id, entryPair.docID, entryPair.sentID,
                    entryPair.sysEntryID, entryPair.refEntryID))
        sys.stdout.flush()
        if (entryPair.refEntry == None):
            ## Entry False Alarm
            constructFalseAlarmScoreStruct(entryPair)
            displayScoreStruct(entryPair)
        elif (entryPair.sysEntry == None):
            ## Entry Miss
            constructMissScoreStruct(entryPair)
            displayScoreStruct(entryPair)
        elif ((len(entryPair.refEventsTable) == 0) and (len(entryPair.sysEventsTable) == 0)):
            ## Entry Match with No Events to Score
            constructCorrectNoEventConfiguration (entryPair)
            displayScoreStruct(entryPair)
        else:
            ## Entry Match with Events to Evaluate
            entryPair.scoreStruct = findBestScoringAlignmentKuhnMunkres (entryPair)
            entryPair.scoreStruct.computeScore()
            displayScoreStruct (entryPair)
        if alignmentDebugP:
            print ("TS={0} Final score for entryPair id={1} docid={2} sentid={3}\n  sysEntryID={4}\n  refEntryID={5}".format(
                    datetime.now().strftime("%H:%M:%S"),
                    entryPair.id, entryPair.docID, entryPair.sentID,
                    entryPair.sysEntryID, entryPair.refEntryID))
        incrementEntrySetScore (entryPair, entrySet)

def constructFalseAlarmScoreStruct (entryPair):
    scoreStruct = entryPair.scoreStruct
    scoreStruct.eventTypeMeasures.falseAlarmCount = len(entryPair.sysEventsTable)
    scoreStruct.argMatchMeasures.falseAlarmCount  = countAllArguments (entryPair.sysEventsTable)
    scoreStruct.computeScore()
    return scoreStruct

def constructMissScoreStruct (entryPair):
    scoreStruct = entryPair.scoreStruct
    scoreStruct.eventTypeMeasures.missCount = len(entryPair.refEventsTable)
    scoreStruct.argMatchMeasures.missCount  = countAllArguments (entryPair.refEventsTable)
    scoreStruct.computeScore()
    return scoreStruct

def constructCorrectNoEventConfiguration (entryPair):
    ## No need to do anything -- no match, no false alarm, no miss -- a correct silence by system
    scoreStruct = entryPair.scoreStruct
    return scoreStruct

def scoreAlignment (entryPair, sysRefAlignment):
    global alignmentDebugP
    if alignmentDebugP:
        print ("scoreAlignment entryPair.id={0} alignment={1}".format(entryPair.id, sysRefAlignment))
        sys.stdout.flush()
    scoreStruct = ScoreStruct()
    scoreStruct.sysRefAlignment = sysRefAlignment
    sysEventIds = sysRefAlignment[0]
    refEventIds = sysRefAlignment[1]
    for i in range(len(sysEventIds)):
        sysEvtId = sysEventIds[i]
        refEvtId = refEventIds[i]
        if alignmentDebugP:
            print ("index into sys-to-ref alignment = {0} sysEvtId = {1} refEvtId = {2}".format(i, sysEvtId, refEvtId))
            sys.stdout.flush()
        if (sysEvtId == None):
            scoreStruct.eventTypeMeasures.addMiss(1.0)
            refEvent = entryPair.refEventsTable[refEvtId]
            scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, [], refEvent["agents"])
            scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, [], refEvent["patients"])
        elif (refEvtId == None):
            scoreStruct.eventTypeMeasures.addFalseAlarm(1.0)
            sysEvent = entryPair.sysEventsTable[sysEvtId]
            scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, sysEvent["agents"],   [])
            scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, sysEvent["patients"], [])
        else:
            sysEvent = entryPair.sysEventsTable[sysEvtId]
            refEvent = entryPair.refEventsTable[refEvtId]
            computeQuadClassMatchValue (entryPair, scoreStruct.eventTypeMeasures, sysEvent, refEvent)
            scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, sysEvent["agents"],   refEvent["agents"])
            scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, sysEvent["patients"], refEvent["patients"])
    scoreStruct.computeScore()
    if alignmentDebugP:
        scoreStruct.displayMeasures(sys.stdout)
    return scoreStruct

def scoreEventPair (refEvtID, sysEvtID, entryPair, scoreCache):
    global alignmentDebugP
    if alignmentDebugP:
        print ("scoreEventPair sysEvtId = {0} refEvtId = {1}".format(sysEvtID, refEvtID))
        sys.stdout.flush()
    refSysIdPair = (refEvtID, sysEvtID)
    if (refSysIdPair in scoreCache):
        if alignmentDebugP:
            print ("  cache score = {0}".format(scoreCache[refSysIdPair]))
        return scoreCache[refSysIdPair][0]
    scoreStruct = ScoreStruct()
    if (sysEvtID == None):
        scoreStruct.eventTypeMeasures.addMiss(1.0)
        refEvent = entryPair.refEventsTable[refEvtID]
        scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, [], refEvent["agents"])
        scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, [], refEvent["patients"])
    elif (refEvtID == None):
        scoreStruct.eventTypeMeasures.addFalseAlarm(1.0)
        sysEvent = entryPair.sysEventsTable[sysEvtID]
        scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, sysEvent["agents"],   [])
        scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, sysEvent["patients"], [])
    else:
        sysEvent = entryPair.sysEventsTable[sysEvtID]
        refEvent = entryPair.refEventsTable[refEvtID]
        computeQuadClassMatchValue    (entryPair, scoreStruct.eventTypeMeasures, sysEvent, refEvent)
        scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, sysEvent["agents"],   refEvent["agents"])
        scoreArgumentMentionsForEvent (entryPair, scoreStruct.argMatchMeasures, sysEvent["patients"], refEvent["patients"])
    scoreStruct.computeScore()
    scoreCache[refSysIdPair] = (scoreStruct.combinedScore, scoreStruct)
    if alignmentDebugP:
        print ("  computed score = {0}".format(scoreCache[refSysIdPair][0]))
    return scoreStruct.combinedScore


def findBestScoringAlignmentKuhnMunkres (entryPair):
    global alignmentDebugP, possAlignDebugP
    if alignmentDebugP:
        print ("findBestScoringAlignmentKuhnMunkres entryPair.id={0}".format(entryPair.id))
        sys.stdout.flush()
    sysEvtIdList = [x for x in entryPair.sysEventsTable.keys()]
    refEvtIdList = [x for x in entryPair.refEventsTable.keys()]
    if possAlignDebugP:
        print ("sysEvtIdList: {0}".format(sysEvtIdList))
        print ("refEvtIdList: {0}".format(refEvtIdList))

    ## Fill out with Nones for misses/false-alarms
    if (len(sysEvtIdList) > len(refEvtIdList)):
        if possAlignDebugP:
            print ("padding refEvtIdList...")
        refEvtIdList = refEvtIdList + [noneVal for noneVal in itertools.repeat(None,(len(sysEvtIdList) - len(refEvtIdList)))]
        if possAlignDebugP:
            print ("refEvtIdList: {0}".format(refEvtIdList))
    if (len(refEvtIdList) > len(sysEvtIdList)):
        if possAlignDebugP:
            print ("padding sysEvtIdList...")
        sysEvtIdList = sysEvtIdList + [noneVal for noneVal in itertools.repeat(None,(len(refEvtIdList) - len(sysEvtIdList)))]
        if possAlignDebugP:
            print ("sysEvtIdList: {0}".format(sysEvtIdList))
    if (not onlySummaryP):
        print ("From {0} system and {1} reference events ".format(len(sysEvtIdList), len(refEvtIdList)))
    sys.stdout.flush()
    scoreCache    = {}
    matrix        = make_cost_matrix([[scoreEventPair(refEvtID, sysEvtID, entryPair, scoreCache) for sysEvtID in sysEvtIdList] for refEvtID in refEvtIdList],
                                      lambda cost: 1.0 - cost)
    indexPairs    = Munkres().compute(matrix)
    alignScoreStr = ScoreStruct()
    ## Since we have padded the sys and ref id lists with None values, and the scoreEventPair function handles
    ## one of the event IDs being a None, then we should have a completely filled square matrix.
    sysEvtList    = []
    refEvtList    = []
    for row, column in indexPairs:
        refEvtID     = refEvtIdList[row]
        sysEvtID     = sysEvtIdList[column]
        refEvtList.append(refEvtID)
        sysEvtList.append(sysEvtID)
        refSysIdPair = (refEvtID, sysEvtID)
        evtScoreStr  = scoreCache[refSysIdPair][1]
        alignScoreStr.updateFromScoreStruct(evtScoreStr)
    alignScoreStr.computeScore()
    alignScoreStr.sysRefAlignment = [sysEvtList, refEvtList]
    if alignmentDebugP:
        alignScoreStr.displayMeasures(sys.stdout)
    return alignScoreStr

def computeQuadClassMatchValue (entryPair, eventTypeMeasures, sysEvent, refEvent):
    global quadClassMatchPolicy
    if (quadClassMatchPolicy == 'strict'):
        computeStrictQuadClassMatchValue (entryPair, eventTypeMeasures, sysEvent, refEvent)
    elif (quadClassMatchPolicy == 'loose'):
        computeLooseQuadClassMatchValue (entryPair, eventTypeMeasures, sysEvent, refEvent)
    elif (quadClassMatchPolicy == 'ignore'):
        computeFixedQuadClassMatchValue (entryPair, eventTypeMeasures, sysEvent, refEvent)
    else:
        print ("ERROR: Unrecognized quadClassMatchPolicy = {0}".format(quadClassMatchPolicy))

def computeStrictQuadClassMatchValue (entryPair, eventTypeMeasures, sysEvent, refEvent):
    if (("material-verbal" not in refEvent) and ("material-verbal" not in sysEvent) and
        ("helpful-harmful" not in refEvent) and ("helpful-harmful" not in sysEvent)):
        eventTypeMeasures.addMatch(1.0)      ## Both dimensions of QuadClass are empty in both sys and ref
    elif (("material-verbal" in refEvent) and
          ("material-verbal" in sysEvent) and
          (refEvent["material-verbal"] == sysEvent["material-verbal"]) and
          ("helpful-harmful" in refEvent) and
          ("helpful-harmful" in sysEvent) and
          (refEvent["helpful-harmful"] == sysEvent["helpful-harmful"])):
        eventTypeMeasures.addMatch(1.0)      ## Both dimensions of QuadClass match
    else:
        eventTypeMeasures.addMiss(1.0)       ## An event with this specific reference quadclass wasn't generated by system
        eventTypeMeasures.addFalseAlarm(1.0) ## An event with this specific system quadclass had no counterpart in reference

def computeLooseQuadClassMatchValue (entryPair, eventTypeMeasures, sysEvent, refEvent):
    if (("material-verbal" not in refEvent) and ("material-verbal" not in sysEvent)):
        eventTypeMeasures.addMatch(0.5)      ## Give credit if sys and ref dimension values are empty
    elif (("material-verbal" in refEvent) and ("material-verbal" in sysEvent) and
          (refEvent["material-verbal"] == sysEvent["material-verbal"])):
        eventTypeMeasures.addMatch(0.5)      ## They matched on this dimension
    else:
        eventTypeMeasures.addMiss(0.5)       ## An event with this specific reference quadclass wasn't generated by system
        eventTypeMeasures.addFalseAlarm(0.5) ## An event with this specific system quadclass had no counterpart in reference

    if (("helpful-harmful" not in refEvent) and ("helpful-harmful" not in sysEvent)):
        eventTypeMeasures.addMatch(0.5)      ## Give credit if sys and ref dimension values are empty
    elif (("helpful-harmful" in refEvent) and ("helpful-harmful" in sysEvent) and
          (refEvent["helpful-harmful"] == sysEvent["helpful-harmful"])):
        eventTypeMeasures.addMatch(0.5)      ## They matched on this dimension
    else:
        eventTypeMeasures.addMiss(0.5)       ## An event with this specific reference quadclass wasn't generated by system
        eventTypeMeasures.addFalseAlarm(0.5) ## An event with this specific system quadclass had no counterpart in reference

def computeFixedQuadClassMatchValue (entryPair, eventTypeMeasures, sysEvent, refEvent):
    eventTypeMeasures.addMatch(1.0)  ## We give credit without consulting actual quadclass values

def scoreArgumentMentionsForEvent (entryPair, argMeasures, sysEntityIDs, refEntityIDs):
    global spanMatchDebugP
    if spanMatchDebugP:
        print("scoreArgumentMentionsForEvent sys: {0} <--> ref: {1} ...".format(sysEntityIDs, refEntityIDs))
        sys.stdout.flush()
    if ((len(sysEntityIDs) == 0) and (len(refEntityIDs) == 0)):
        argMeasures.addMatch(0.0) ## no mention credit, but also no false-alarm demerits, either
        if spanMatchDebugP:
            print("empty slots match")
            sys.stdout.flush()
        return
    usedRefEntityIDs = []
    matchedRefEntityP = False
    ## Loop through all system entities
    for sysEntID in sysEntityIDs:
        if spanMatchDebugP:
            print("looking for match for sys entity {0}".format(sysEntID))
            sys.stdout.flush()
        if (sysEntID not in entryPair.sysSpanSetsTable):
            print("ERROR -- sysEntID {0} missing from its own sysSpanSetsTable!!!".format(sysEntID))
            sys.stdout.flush()
        else:
            sysEnt = entryPair.sysSpanSetsTable[sysEntID]
            ## Loop through each mention of a sys entity
            matchedRefEntityP = False
            for sysMent in sysEnt["spans"]:
                if (not matchedRefEntityP):
                    sysMentStringLower = sysMent["string"].lower()
                    if spanMatchDebugP:
                        print("sys mention: {0}".format(sysMentStringLower))
                    ## loop through all reference entities that haven't already been used to match
                    ## previous sys entities
                    for refEntID in refEntityIDs:
                        if (refEntID not in entryPair.refSpanSetsTable):
                            print("ERROR -- refEntID {0} missing from its own refSpanSetsTable???".format(refEntID))
                        else:
                            refEnt = entryPair.refSpanSetsTable[refEntID]
                            if ((not matchedRefEntityP) and (refEntID not in usedRefEntityIDs)):
                                for refMent in refEnt["spans"]:
                                    if (not matchedRefEntityP):
                                        refMentStringLower = refMent["string"].lower()
                                        if (sysMatchesRefMentionP(sysMentStringLower, refMentStringLower)):
                                            if spanMatchDebugP:
                                                print("found ref mention match: {0}".format(sysMentStringLower))
                                            matchedRefEntityP = True
                                            usedRefEntityIDs.append(refEntID)
                                        else:
                                            if spanMatchDebugP:
                                                print("does not match ref mention {0}".format(refMentStringLower))
        if (not matchedRefEntityP):
            if spanMatchDebugP:
                print ("FALSE ALARM")
            argMeasures.addFalseAlarm(1.0)
            ## argMeasures.addMiss(1.0)
            if spanMatchDebugP:
                print("could not find a matching ref entity for sys entity {0} falseAlarmCount = {1}".format(sysEntID, argMeasures.falseAlarmCount))
        else:
            if spanMatchDebugP:
                print ("MATCH")
            argMeasures.addMatch(1.0)
            if spanMatchDebugP:
                print("found a matching ref entity for sys entity {0} matchCount = {1}".format(sysEntID, argMeasures.matchCount))
    if spanMatchDebugP:
        print("Looking at missed ref entities...")
        sys.stdout.flush()
    for refEntId in refEntityIDs:
        if (refEntId not in usedRefEntityIDs):
            argMeasures.addMiss(1)
            if spanMatchDebugP:
                print ("MISS")
            if spanMatchDebugP:
                print("Missed this ref entity: {0} missCount = {1}".format(refEntId, argMeasures.missCount))
    if spanMatchDebugP:
        print("Finished scoreArgumentMentionsForEvent")
        print("new state of argMatchMeasures: {0}".format(argMeasures.renderMeasures()))
        sys.stdout.flush()
    
def sysMatchesRefMentionP (sysMentStringLower, refMentStringLower):
    global spanMatchDebugP, argMatchPolicy
    ## This code should be enriched to cover language-specific normalization, to include
    ## alternate case markings, presence/absence of determiners or other "unimportant"
    ## modifiers
    if (argMatchPolicy == 'ignore'):
        return True
    if (sysMentStringLower == refMentStringLower):
        if spanMatchDebugP:
            print("ref and sys arg strings are exact match: {0}".format(refMentStringLower))
        return True
    elif (argMatchPolicy == 'strict'):
        if spanMatchDebugP:
            print("argMatchPolicy = {0} NO EXACT MATCH: sys: {1} <--> ref: {2}".format(argMatchPolicy, sysMentStringLower, refMentStringLower))
        return False
    elif (sysMentStringLower.find(refMentStringLower) >= 0):
        if spanMatchDebugP:
            print("argMatchPolicy = {0} PARTIAL STRING MATCH succeeded: sys: <{1}> ref: <{2}>".format(argMatchPolicy,sysMentStringLower,refMentStringLower))
        return True
    elif (refMentStringLower.find(sysMentStringLower) >= 0):
        if spanMatchDebugP:
            print("argMatchPolicy = {0} PARTIAL STRING MATCH succeeded: sys: <{1}> ref: <{2}>".format(argMatchPolicy,sysMentStringLower,refMentStringLower))
        return True
    else:
        if spanMatchDebugP:
            print("argMatchPolicy = {0} NO PARTIAL MATCH".format(argMatchPolicy))
        return False

def allowableEventAlignmentP (sysEvent, refEvent):
    possAlignP = False
    if (quadClassMatchPolicy in ['ignore', 'not-required']):
        possAlignP = True
    elif (("material-verbal" not in refEvent) and ("material-verbal" not in sysEvent)):
        possAlignP = True
    elif ((quadClassMatchPolicy in ['strict', 'loose']) and
          ("material-verbal" in refEvent) and
          ("material-verbal" in sysEvent) and 
          (refEvent["material-verbal"] == sysEvent["material-verbal"]) and
          (refEvent["helpful-harmful"] == sysEvent["helpful-harmful"])):
        possAlignP = True
    elif ((quadClassMatchPolicy == 'loose') and
          ("material-verbal" in refEvent)   and
          ("material-verbal" in sysEvent)   and 
          ((refEvent["material-verbal"].find(sysEvent["material-verbal"]) >= 0) or
           (sysEvent["material-verbal"].find(refEvent["material-verbal"]) >= 0)) and
          ((refEvent["helpful-harmful"].find(sysEvent["helpful-harmful"]) >= 0) or
           (sysEvent["helpful-harmful"].find(refEvent["helpful-harmful"]) >= 0))):
        possAlignP = True
    else:
        possAlignP = False

    ## anchor matching policy below imposes ADDITIONAL constraints -- it cannot
    ## reset the variable possAlignP to True if had already been set to False by
    ## the quadClass matching logic above.  Note that we don't even bother testing
    ## if the anchorMatchPolicy is 'ignore'.
    if ((anchorMatchPolicy == 'strict') and
        (not anchorStringsMatchP(refEvent,sysEvent,refSpanSetsTable,sysSpanSetsTable))):
        possAlignP = False
    elif ((anchorMatchPolicy == 'loose') and
          (not anchorStringsPartiallyMatchP(refEvent,sysEvent,refSpanSetsTable,sysSpanSetsTable))):
        possAlignP = False

    return possAlignP

def incrementEntrySetScore (entryPair, entrySet):
    global entrySetScoringP, quiteP
    epScoreStruct = entryPair.scoreStruct
    esScoreStruct = entrySet.scoreStruct
    if entrySetScoringP:
        print ("incrementing EntrySet score {0} from EntryPair score {1}...".format(
                esScoreStruct.id, epScoreStruct.id))
        print ("entryPair score struct BEFORE: {0}".format(epScoreStruct.renderScoreStruct()))
        print ("entrySet  score struct BEFORE: {0}".format(esScoreStruct.renderScoreStruct()))
        sys.stdout.flush()

    esScoreStruct.eventTypeMeasures.addMiss      (epScoreStruct.eventTypeMeasures.missCount)
    esScoreStruct.eventTypeMeasures.addFalseAlarm(epScoreStruct.eventTypeMeasures.falseAlarmCount)
    esScoreStruct.eventTypeMeasures.addMatch     (epScoreStruct.eventTypeMeasures.matchCount)

    esScoreStruct.argMatchMeasures.addMiss       (epScoreStruct.argMatchMeasures.missCount)
    esScoreStruct.argMatchMeasures.addFalseAlarm (epScoreStruct.argMatchMeasures.falseAlarmCount)
    esScoreStruct.argMatchMeasures.addMatch      (epScoreStruct.argMatchMeasures.matchCount)

    esScoreStruct.computeScore()
    if (not quietP):
        epScoreStruct.displayMeasuresAbbrev(sys.stdout, 'entryPair')
        esScoreStruct.displayMeasuresAbbrev(sys.stdout, 'entrySet')
        sys.stdout.flush()

def countAllArguments (eventsTable):
    ## should exclude anchors, since they are NOT event "arguments"
    argCount = 0
    for evtID, event in eventsTable.items():
        for argType in ["agents", "patients"]:
            if (argType in event):
                argCount += len(event[argType])
    return argCount

def renderEventStruct (event, entitiesTable):
    global debugP
    if debugP:
        print("renderEventStruct")
    return renderAbstractEventStruct (event, entitiesTable)

def getSpanSetString (spanSetID, entitiesTable):
    global debugP
    if debugP:
        print("getSpanSetString ...")
    if (spanSetID not in entitiesTable):
        return "<Unrecognized-SpanSetID>"
    else:
        spanSet = entitiesTable[spanSetID]
        ## print "spanSetID: {0}  spanSet: {1}".format(spanSetID, spanSet)
        if (len(spanSet["spans"]) == 1):
            return spanSet["spans"][0]["string"]
        else:
            return "{0}{1}".format(spanSet["spans"][0]["string"], " (and others)")

def getSpanSetOffset (spanSetID, entitiesTable):
    if (spanSetID not in entitiesTable):
        print("getSpanSetOffset: unrecognized spanSetID ({0})".format(spanSetID))
        return -2
    else:
        spanSet = entitiesTable[spanSetID]
        if (len(spanSet["spans"]) >= 1):
            if ("start" in spanSet["spans"][0]):
                return spanSet["spans"][0]["start"]
            else:
                print("getSpanSetOffset: Span doesn't have start offset")
                return -3
        else:
            print("getSpanSetOffset: No spans for this spanSetID ({0})".format(spanSetID))
            return -1

def renderAbstractEventStruct (event, entitiesTable):
    stringPieces = ["<event id={0} ".format(event["event-id"])]
    if ("material-verbal" in event):
        stringPieces.append(event["material-verbal"])
        stringPieces.append(" ")
    if ("helpful-harmful" in event):
        stringPieces.append(event["helpful-harmful"])
        stringPieces.append(" ")
    stringPieces.append(getSpanSetString(event["anchors"], entitiesTable))
    stringPieces.append(" ")
    for arg in ["agents", "patients"]:
        if (arg in event):
            if (len(event[arg]) == 0):
                stringPieces.append("{0}=[] ".format(arg))
            else:
                for entID in event[arg]:
                    firstMention = None
                    if (entID in entitiesTable):
                        if ("spans" in entitiesTable[entID]):
                            mentions = entitiesTable[entID]["spans"]
                            firstMention = mentions[0]["string"]
                            try:
                                firstMention = codecs.encode(firstMention,'UTF-8','replace')
                                break
                            except UnicodeEncodeError:
                                firstMention = "unicode-error"
                        else:
                            firstMention = "(No mentions)"
                    else:
                        firstMention = "(No entity {0})".format(entID)
                    stringPieces.append("{0}={1}({2}) ".format(arg, entID, firstMention))
    stringPieces.append(">")
    eventRendered = ''.join(stringPieces)
    return eventRendered

def anchorStringsMatchP (refEvt, sysEvt, refSpanSetsTable, sysSpanSetsTable):
    refEvtID    = refEvt["anchors"]
    sysEvtID    = sysEvt["anchors"]
    refEvtSpans = refSpanSetsTable[refEvtID]["spans"]
    sysEvtSpans = sysSpanSetsTable[sysEvtID]["spans"]
    for refSpan in refEvtSpans:
        for sysSpan in sysEvtSpans:
            if (refSpan["string"] == sysSpan["string"]):
                ## print "These strings match! ref: {0} sys: {1}".format(refSpan["string"], sysSpan["string"])
                return True
    return False

def anchorStringsPartiallyMatchP (refEvt, sysEvt, refSpanSetsTable, sysSpanSetsTable):
    refEvtID    = refEvt["anchors"]
    sysEvtID    = sysEvt["anchors"]
    refEvtSpans = refSpanSetsTable[refEvtID]["spans"]
    sysEvtSpans = sysSpanSetsTable[sysEvtID]["spans"]
    for refSpan in refEvtSpans:
        for sysSpan in sysEvtSpans:
            if ((refSpan["string"].find(sysSpan["string"]) >= 0) or
                (sysSpan["string"].find(refSpan["string"]) >= 0)):
                return True
    return False

def addToScoredScoreStructsTable():
    global debugP, tsvFile
    if debugP:
        print("addToScoredScoreStructsTable")
    bestScoreStructsForEntries.append(bestScoreStruct)
    print("{0} bestScoreStructs in bestScoreStructsForEntries".format(len(bestScoreStructsForEntries)))
    if (tsvFile != None):
        addLineToTSV (bestScoreStruct, tsvFile)

fieldSep = "\t"

def addTsvHeaders (tsvFile):
    global fieldSep
    headers = ["System-ID", "Reference-ID",
               "evt-misses", "evt-false-alarms", "evt-matches", "evt-precision", "evt-recall", "evt-fmeasure",
               "arg-misses", "arg-false-alarms", "arg-matches", "arg-precision", "arg-recall", "arg-fmeasure", "combined-score"]
    headerCount = len(headers)
    for i in range(headerCount):
        tsvFile.write(headers[i])
        if (i < (headerCount - 1)):
            tsvFile.write(fieldSep)
    tsvFile.write("\n")

def addLineToTSV (sysEntryID, refEntryID, scoreStruct, tsvFile):
    global fieldSep
    tsvFile.write(str(sysEntryID) + fieldSep)
    tsvFile.write(str(refEntryID) + fieldSep)
    evtMsrs = scoreStruct.eventTypeMeasures
    tsvFile.write(str(evtMsrs.missCount) + fieldSep)
    tsvFile.write(str(evtMsrs.falseAlarmCount) + fieldSep)
    tsvFile.write(str(evtMsrs.matchCount) + fieldSep)
    tsvFile.write(str(evtMsrs.precision) + fieldSep)
    tsvFile.write(str(evtMsrs.recall) + fieldSep)
    tsvFile.write(str(evtMsrs.fmeasure) + fieldSep)
    argMsrs = scoreStruct.argMatchMeasures
    tsvFile.write(str(argMsrs.missCount) + fieldSep)
    tsvFile.write(str(argMsrs.falseAlarmCount) + fieldSep)
    tsvFile.write(str(argMsrs.matchCount) + fieldSep)
    tsvFile.write(str(argMsrs.precision) + fieldSep)
    tsvFile.write(str(argMsrs.recall) + fieldSep)
    tsvFile.write(str(argMsrs.fmeasure) + fieldSep)
    tsvFile.write(str(scoreStruct.combinedScore))
    tsvFile.write("\n")

def displayScoreStruct (entryPair):
    global debugP, quietP, reportFile
    if (not quietP):
        displayScoreStructAux (entryPair, entryPair.scoreStruct, sys.stdout)
    if (reportFile != None):
        displayScoreStructAux (entryPair, entryPair.scoreStruct, reportFile)

def displayScoreStructAux (entryPair, scoreStruct, outFile):
    sysEntry   = entryPair.sysEntry
    sysEntryID = entryPair.sysEntryID 
    refEntry   = entryPair.refEntry
    refEntryID = entryPair.refEntryID
    outFile.write("------------- displaying score for an entry pair -----------------------\n")
    outFile.write(" sys entry: {0}     ref entry: {1}\n".format(sysEntryID, refEntryID))
    ## we let the calling function provide the score struct, in case it hasn't yet been
    ## established as the best -- just a *possible* configuration.
    ## print(scoreStruct.renderScoreStruct())
    textSignal = "*** None available ***"
    if ((entryPair.sentenceText != None) and (entryPair.sentenceText != "")):
        textSignal = entryPair.sentenceText.replace("\n"," ").replace("\r"," ").replace("\t"," ")
    outFile.write("text-signal: <<{0}>>\n".format(textSignal))
    if (sysEntry == None):
        if (refEntry == None):
            outFile.write(" WEIRD: Neither a sys nor a ref entry available!!\n")
        else:
            outFile.write(" No sys entry available at all for this sentence!\n")
            displayRefEntryMiss (outFile, entryPair)
    elif (refEntry == None):
        outFile.write(" No ref entry available at all for this sentence!\n")
        displaySysEntryFalseAlarm (outFile, entryPair)
    elif (scoreStruct.sysRefAlignment == None):
        outFile.write(" <sysEvent None>\n")
        outFile.write(" <refEvent None>\n")
    else:
        sysEventIds = scoreStruct.sysRefAlignment[0]
        refEventIds = scoreStruct.sysRefAlignment[1]
        for i in range(len(sysEventIds)):
            sysEvtId = sysEventIds[i]
            refEvtId = refEventIds[i]
            ## print ("sysEvtId = {0} refEvtId = {1}".format(sysEvtId, refEvtId))
            if (sysEvtId == None):
                outFile.write(" <sysEvent None>\n")
            else:
                outFile.write(displayEvent(sysEvtId, entryPair.sysEventsTable, entryPair.sysSpanSetsTable, "sys"))
                emitEventArgs (outFile, sysEvtId, entryPair.sysEventsTable, entryPair.sysSpanSetsTable)
            if (refEvtId == None):
                outFile.write(" <refEvent None>\n")
            else:
                outFile.write(displayEvent(refEvtId, entryPair.refEventsTable, entryPair.refSpanSetsTable, "ref"))
                emitEventArgs (outFile, refEvtId, entryPair.refEventsTable, entryPair.refSpanSetsTable)
    outFile.write("  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
    scoreStruct.displayMeasures(outFile)
    outFile.write("------------------------------------------------------------------------\n")
    sys.stdout.flush()

def displayRefEntryMiss (outFile, entryPair):
    for evtId, event in entryPair.refEventsTable.items():
        outFile.write(" <sysEvent None>\n")
        outFile.write(displayEvent(evtId, entryPair.refEventsTable, entryPair.refSpanSetsTable, "ref"))
        emitEventArgs (outFile, evtId, entryPair.refEventsTable, entryPair.refSpanSetsTable)

def displaySysEntryFalseAlarm (outFile, entryPair):
    for evtId, event in entryPair.sysEventsTable.items():
        outFile.write(displayEvent(evtId, entryPair.sysEventsTable, entryPair.sysSpanSetsTable, "sys"))
        emitEventArgs (outFile, evtId, entryPair.sysEventsTable, entryPair.sysSpanSetsTable)
        outFile.write(" <refEvent None>\n")

def displayEvent (evtId, eventsTable, spanSetsTable, annotRole):
    ## for eid, evt in eventsTable.items():
    ##     print ("displayEvent eid={0} evt={1}".format(eid, evt))
    event    = eventsTable[evtId]
    anchorID = event["anchors"]
    anchorStrings = bpjson.getSpanSetStrings (spanSetsTable, anchorID)
    ## print ("anchorStrings = {0}".format(anchorStrings))
    if (len(anchorStrings) > 1):
        anchorStrings = "<" + '|'.join(anchorStrings) + ">"
    elif (len(anchorStrings) == 1):
        anchorStrings = "<" + anchorStrings[0] + ">"
    elif (len(anchorStrings) == 0):
        anchorStrings = "<<NONE>>"
    mv = bpjson.renderMaterialVerbal (event)
    hh = bpjson.renderHelpfulHarmful (event)
    return " <{0}Event id: {1} {2}_{3} anchor: {4}>\n".format(annotRole, evtId, mv, hh, anchorStrings)

def emitEventArgs (outFile, evtId, eventsTable, spanSetsTable):
    event = eventsTable[evtId]
    for argType in ["agents", "patients"]:
        outFile.write("  {0} ".format(argType))
        if (len(event[argType]) == 0):
            outFile.write(" []\n")
        else:
            outFile.write(" [ ")
            for argID in event[argType]:
                spanSet = spanSetsTable[argID]
                outFile.write(" <")
                spanCount = len(spanSet["spans"])
                for i in range(spanCount):
                    span = spanSet["spans"][i]
                    outFile.write(span["string"])
                    if (i < (spanCount - 1)):
                        outFile.write(",")
                outFile.write(">")
            outFile.write(" ]\n")

def computeRecPreF (missCount, falseAlarmCount, matchCount):
    measures = Measures(missCount, falseAlarmCount, matchCount)
    measures.computeMetrics()
    return measures

scoreStructGenCount = 0

class EntrySet:
    def __init__(self, reffile, sysfile):
        self.refFilename   = reffile
        self.sysFilename   = sysfile
        self.entryPairs    = []
        self.refCorpus     = bpjson.getStructFromJsonFile (reffile)
        self.sysCorpus     = bpjson.getStructFromJsonFile (sysfile)
        self.scoreStruct   = ScoreStruct()

    def renderEntrySetForScoring (self):
        if (self.refFilename != None):
            return "<EntrySet id={0} refFilename={1} sysFilename={2}>".format(
                self.id, self.refFilename, self.sysFilename)
        else:
            return "<EntrySet id={0} #source-files={1} #entryPairs={2}>".format(
                self.id, len(self.sourceFiles), len(self.entryPairsForScoring))

entryPairCounter = 0

class EntryPair:
    def __init__(self, refEntry, sysEntry, entrySet):
        global entryPairCounter
        entryPairCounter         += 1
        self.id                   = entryPairCounter
        self.entrySet             = entrySet
        self.refEntry             = refEntry
        self.sysEntry             = sysEntry

        if (refEntry != None):
            self.docID            = bpjson.getEntryDocID(refEntry)
            self.sentID           = bpjson.getEntrySentID(refEntry)
            self.docID_sentID     = bpjson.getEntryDocID_SentID (refEntry)
            self.sentenceText     = bpjson.getEntrySentenceText (refEntry)
        else:
            self.docID            = bpjson.getEntryDocID(sysEntry)
            self.sentID           = bpjson.getEntrySentID(sysEntry)
            self.docID_sentID     = bpjson.getEntryDocID_SentID (sysEntry)
            self.sentenceText     = bpjson.getEntrySentenceText (sysEntry)

        if (refEntry != None):
            self.refEventsTable   = getEventsTable  (refEntry, "ref")
            self.refSpanSetsTable = getSpanSetTable (refEntry, "ref")
            self.refEntryID       = bpjson.getEntryID(refEntry)
        else:
            self.refEventsTable   = None
            self.refSpanSetsTable = None
            self.refEntryID       = None

        if (sysEntry != None):
            self.sysEventsTable   = getEventsTable  (sysEntry, "sys")
            self.sysSpanSetsTable = getSpanSetTable (sysEntry, "sys")
            self.sysEntryID       = bpjson.getEntryID(sysEntry)
        else:
            self.sysEventsTable   = None
            self.sysSpanSetsTable = None
            self.sysEntryID       = None
        self.scoreStruct          = ScoreStruct()

    def renderEntryPairForScoring (self):
        return "<EntryPairForScoring id={0} docID_sentID={1}>".format(self.id, self.docID_sentID)

def getEventsTable (entry, entryType):
    if ("annotation-sets" not in entry):
        print("No 'annotation-sets' found in {0} entry".format(entryType))
        return {}
    annotSets = entry["annotation-sets"]
    if ("abstract-events" not in annotSets):
        print("No 'abstract-events' entry in {0} annotation-sets".format(entryType))
        return {}
    abstractEvents = annotSets["abstract-events"]
    if ("events" not in abstractEvents):
        print("No {0} events found in entry".format(entryType))
        return {}
    events = abstractEvents["events"]
    ## Ensure "event-id" slot is populated in each event structure
    for evtId, event in events.items():
        if ("event-id" not in event):
            event["event-id"] = evtId
    return events

def getSpanSetTable (entry, entryType):
    if ("annotation-sets" not in entry):
        print("No 'annotation-sets' found in {0} entry".format(entryType))
        return {}
    annotSets = entry["annotation-sets"]
    if ("abstract-events" not in annotSets):
        print("No 'abstract-events' entry in {0} annotation-sets".format(entryType))
        return {}
    abstractEvents = annotSets["abstract-events"]
    if ("span-sets" not in abstractEvents):
        print("No {0} entities found in entry".format(entryType))
        return {}
    spanSets = abstractEvents["span-sets"]
    ## Ensure "span-set-id" slot is populated in each span-set structure
    for entId, spanSet in spanSets.items():
        if ("span-set-id" not in spanSet):
            spanSet["span-set-id"] = entId
    return spanSets

scoreStructCounter = 0

class ScoreStruct:
    def __init__(self):
        global scoreStructCounter
        scoreStructCounter    += 1
        self.id                = scoreStructCounter
        self.sysRefAlignment   = None
        self.eventTypeMeasures = Measures(0.0, 0.0, 0.0)
        self.argMatchMeasures  = Measures(0.0, 0.0, 0.0)
        self.combinedScore     = 0.0

    def renderScoreStruct(self):
        return "<ScoreStruct evt miss={1} falm={2} mtch={3} arg miss={4} falm={5} mtch={6} comb={7} (id={0})>".format(
            self.id, 
            self.eventTypeMeasures.missCount,
            self.eventTypeMeasures.falseAlarmCount,
            self.eventTypeMeasures.matchCount,
            self.argMatchMeasures.missCount,
            self.argMatchMeasures.falseAlarmCount,
            self.argMatchMeasures.matchCount,
            self.combinedScore)

    def displayMeasures(self, outFile):
        for msrs,msrsType in [[self.eventTypeMeasures, "event type"],
                              [self.argMatchMeasures,  "argmt mtch"]]:
            outFile.write(" {0} missCount:         {1}".format(msrsType, msrs.missCount) + "\n")
            outFile.write(" {0} falseAlarmCount:   {1}".format(msrsType, msrs.falseAlarmCount) + "\n")
            outFile.write(" {0} matchCount:        {1}".format(msrsType, msrs.matchCount) + "\n")
            outFile.write(" {0} precision:         {1}".format(msrsType, msrs.precision) + "\n")
            outFile.write(" {0} recall:            {1}".format(msrsType, msrs.recall) + "\n")
            outFile.write(" {0} fmeasure:          {1}".format(msrsType, msrs.fmeasure) + "\n\n")
        outFile.write(" combined score:               {0}".format(self.combinedScore) + "\n")

    def displayMeasuresAbbrev (self, outFile, scoreLabel):
        for msrs,msrsType in [[self.eventTypeMeasures, "quad"],
                              [self.argMatchMeasures,  "args"]]:
            outFile.write(scoreLabel)
            outFile.write(" {0} miss={1} falm={2} mtch={3}".format(msrsType, msrs.missCount, msrs.falseAlarmCount, msrs.matchCount))
            outFile.write(" prec={1} rcll={2} fmea={3}\n".format(msrsType, msrs.precision, msrs.recall, msrs.fmeasure))
        outFile.write(scoreLabel)
        outFile.write(" combined: {0}".format(self.combinedScore) + "\n")

    def computeScore (self):
        global debugP
        if debugP:
            print("computeScore scoreStruct.id = {0} ...".format(self.id))
        self.eventTypeMeasures.computeMetrics()
        self.argMatchMeasures.computeMetrics()
        self.combinedScore = self.eventTypeMeasures.fmeasure * self.argMatchMeasures.fmeasure
        if debugP:
            print("event type measures: {0}".format(self.eventTypeMeasures.renderMeasures()))
            print("arg match measures:  {0}".format(self.argMatchMeasures.renderMeasures()))
            print("new combined score for scoreStruct {3}: evtType={0} * argMtch={1} => {2}".format(
                    self.eventTypeMeasures.fmeasure, self.argMatchMeasures.fmeasure, self.combinedScore, self.id))
            sys.stdout.flush()
        return self.combinedScore

    def updateFromScoreStruct (self, otherScoreStruct):
        global debugP
        if debugP:
            print("updateFromScoreStruct scoreStruct.id = {0} from scoreStruct.id = {1} ...".format(self.id, otherScoreStruct.id))
        self.eventTypeMeasures.updateFromOtherMeasures(otherScoreStruct.eventTypeMeasures)
        self.argMatchMeasures.updateFromOtherMeasures(otherScoreStruct.argMatchMeasures)
        self.computeScore()

class Measures:
    def __init__(self, missCountArg, falseAlarmCountArg, matchCountArg):
        self.missCount       = missCountArg
        self.falseAlarmCount = falseAlarmCountArg
        self.matchCount      = matchCountArg
        self.recall          = 0.0
        self.precision       = 0.0
        self.fmeasure        = 0.0

    def renderMeasures(self):
        return "<Measures miss={0} falm={1} mtch={2} pre={3} rec={4} fmeas={5}>".format(self.missCount, self.falseAlarmCount, self.matchCount, self.precision, self.recall, self.fmeasure)

    def displayMeasures(self, outFile):
        outFile.write(" missCount:         {0}".format(self.missCount) + "\n")
        outFile.write(" falseAlarmCount:   {0}".format(self.falseAlarmCount) + "\n")
        outFile.write(" matchCount:        {0}".format(self.matchCount) + "\n")
        outFile.write(" precision:         {0}".format(self.precision) + "\n")
        outFile.write(" recall:            {0}".format(self.recall) + "\n")
        outFile.write(" fmeasure:          {0}".format(self.fmeasure) + "\n")

        return "<Measures miss={0} falm={1} mtch={2} pre={3} rec={4} fmeas={5}>".format(self.missCount, self.falseAlarmCount, self.matchCount, self.precision, self.recall, self.fmeasure)

    def addMiss (self,increment):
        self.missCount += increment

    def addFalseAlarm (self,increment):
        self.falseAlarmCount += increment
    
    def addMatch (self,increment):
        self.matchCount += increment

    def computeMetrics (self):
        if ((self.matchCount + self.falseAlarmCount + self.missCount) == 0):
            ## Give 100% credit if no false alarms or misses are encountered -- either 
            ## the system was perfect or there were no items in the reference set
            self.precision = 1.0
            self.recall    = 1.0
            self.fmeasure  = 1.0
        else:
            if ((self.matchCount + self.falseAlarmCount) > 0.0):
                self.precision = float(self.matchCount) / float(self.matchCount + self.falseAlarmCount)
            if ((self.matchCount + self.missCount) > 0.0):
                self.recall    = float(self.matchCount) / float(self.matchCount + self.missCount)
            if ((self.precision + self.recall) != 0.0):
                self.fmeasure = 2.0 * ((self.precision * self.recall) / (self.precision + self.recall))

    def updateFromOtherMeasures (self, otherMeasuresObj):
        if debugP:
            print("updating Measures object from another Measures object ...")
            print(self.renderMeasures())
            print(otherMeasuresObj.renderMeasures())
        self.missCount       += otherMeasuresObj.missCount
        self.falseAlarmCount += otherMeasuresObj.falseAlarmCount
        self.matchCount      += otherMeasuresObj.matchCount
        self.computeMetrics()
        if debugP:
            print("merged scorer: {0}".format(self.renderMeasures()))

def writeOutputLine (line):
    global quietP, reportFile
    if (not quietP):
        sys.stdout.write(line)
        sys.stdout.write("\n")
        sys.stdout.flush()
    if (reportFile != None):
        reportFile.write(line)
        reportFile.write("\n")

if __name__ == "__main__":
    print("Executing {0}".format(sys.argv[0]))
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--reffile", help="A single BP JSON file containing reference annotations (cannot be used with --bpjson, --bpjsondir, --refid or --sysid)")
    argParser.add_argument("--sysfile", help="A single BP JSON file containing  system   annotations (cannot be used with --bpjson, --bpjsondir, --refid or --sysid)")
    argParser.add_argument("--txtrpt",  help="Score report in text format, with alignment details")
    argParser.add_argument("--tsvrpt",  help="Score report in TSV format")
    argParser.add_argument("--quadmatchpolicy", help="'strict' (both dimenions must match), 'loose' (match on at least one dimension required), 'not-required' (zero, one or both may match), 'ignore' (not consulted or weighted). Default={0}".format(quadClassMatchPolicy))
    argParser.add_argument("--argmatchpolicy", help="'strict' (exact string match required), 'loose' (some overlap on args accepted, weighted), 'ignore' (not consulted, not weighted). default={0}".format(argMatchPolicy))
    argParser.add_argument("--anchormatchpolicy", help="'strict' (exact string match required for event alignment), 'loose' (some overlap on anchor accepted for alignment), 'ignore' (ignored when exploring event alignments). Default={0}".format(anchorMatchPolicy))
    argParser.add_argument("--quiet",   help="If present, don't print detailed alignment and scoring information to stdout (default: {0})".format(quietP),
                           action="store_true")
    argParser.add_argument("--summaryonly",   help="If present, don't print ANY entry-level scoring information to stdout (default: {0})".format(onlySummaryP),
                           action="store_true")
    argParser.add_argument("--debug",   help="If present, turn on debugging messages to stdout (default: {0})".format(debugP),
                           action="store_true")

    args = argParser.parse_args()
    print("--reffile:     {0}".format(args.reffile))
    print("--sysfile:     {0}".format(args.sysfile))
    print("--txtrpt:      {0}".format(args.txtrpt))
    print("--tsvrpt:      {0}".format(args.tsvrpt))
    print("--argmatchpolicy     {0}".format(args.argmatchpolicy))
    print("--quadmatchpolicy    {0}".format(args.quadmatchpolicy))
    print("--anchormatchpolicy  {0}".format(args.anchormatchpolicy))
    print("--quiet:       {0}".format(args.quiet))
    print("--debug:       {0}".format(args.debug))
    if len(sys.argv) == 1:
        argParser.print_help()
        sys.exit(1)
    args = argParser.parse_args()

    if args.debug:
        print("Turning debugging on")
        debugP = True

    if args.quiet:
        print("Running scorer in 'quiet' mode")
        quietP = True

    if args.summaryonly:
        print("Running scorer in 'SUMMARY ONLY' mode")
        onlySummaryP = True
        print("Running scorer in 'quiet' mode (entailed by summary only mode)")
        quietP = True

    if (args.txtrpt != None):
        reportFilename = args.txtrpt
        reportFile     = codecs.open(reportFilename, 'w', encoding='UTF-8')
        print("Writing detailed text score report to {0}".format(reportFilename))

    if (args.tsvrpt != None):
        tsvFilename = args.tsvrpt
        print("Will write TSV-formatted summary score report to {0}".format(tsvFilename))

    if (args.argmatchpolicy != None):
        if (args.argmatchpolicy not in argMatchPolicies):
            print("unrecognized --argmatchpolicy value {0}".format(args.argmatchpolicy))
            argParser.print_help()
            sys.exit(0)
        else:
            argMatchPolicy = args.argmatchpolicy
            print("setting anchorMatchPolicy = {0}".format(argMatchPolicy))
    else:
        print("anchorMatchPolicy = {0}".format(argMatchPolicy))

    if (args.quadmatchpolicy != None):
        if (args.quadmatchpolicy not in quadClassMatchPolicies):
            print("unrecognized --quadmatchpolicy value {0}".format(args.quadmatchpolicy))
            argParser.print_help()
            sys.exit(0)
        else:
            quadClassMatchPolicy = args.quadmatchpolicy
            print("setting quadClassMatchPolicy = {0}".format(quadClassMatchPolicy))
    else:
        print("quadClassMatchPolicy = {0}".format(quadClassMatchPolicy))

    if (args.anchormatchpolicy != None):
        if (args.anchormatchpolicy not in anchorMatchPolicies):
            print("unrecognized --anchormatchpolicy value {0}".format(args.anchormatchpolicy))
            argParser.print_help()
            sys.exit(0)
        else:
            anchorMatchPolicy = args.anchormatchpolicy
            print("setting anchorMatchPolicy = {0}".format(anchorMatchPolicy))
    else:
        print("anchorMatchPolicy = {0}".format(anchorMatchPolicy))
            
    scoreSysFileAgainstRefFile (args.reffile, args.sysfile)

    print("Done.")
