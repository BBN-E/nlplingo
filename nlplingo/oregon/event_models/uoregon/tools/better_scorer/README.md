# better_tande_tools

This project contains scripts and utilities that will be provided to
IARPA BETTER Program performer teams.

------------------------------  NOTICE  ------------------------------

This software (or technical data) was produced for the U.S. Government
under contract 2015-14120200002-002, and is subject to the Rights in
Data-General Clause 52.227-14 , ALT IV (MAY 2014) or (DEC 2007).

© 2019 The MITRE Corporation. All Rights Reserved.

----------------------------------------------------------------------

The Python scripts in this package assume version 3.6+ of Python.

Some of the scripts depend on the following additional Python modules
that are NOT covered by the license above.

   `jsonschema` -- https://pypi.org/project/jsonschema/
                 Install with pip or other ways documented at the
		 above web site.

   `munkres` -- https://pypi.org/project/munkres/
              This module is included in this code, and so does not
	      need to be separately installed.

----------------------------------------------------------------------
Running the Abstract Event scorer.

## Setup

1. Make sure you are using Python3.6 or later.
2. Make sure you install the dependencies listed in `requirements.txt`.
    - If using `pip`, simply ingest it with `pip install -r requirements.txt`.
    - If not using `pip`, it's fairly easy to look up the packages with your
      package manager, or install them manually from `PyPI`.

## Invocation

For most purposes you only need two arguments, --reffile and
--sysfile, each of which must be in BP JSON format.  For example:

```bash
unix$ python lib/python3/bptools/scoreAbstractEvents.py  \
             --reffile sampledata/smalltest.ref.bp.json  \
	     --sysfile sampledata/smalltest.sys5.bp.json 
```

A text file of the scoring report can be captured by using the
--txtrpt argument.  A compact tab-separated-values (TSV) version of
the scoring results can be captured using the --tsvrpt argument.

Using the --quite argument will reduce the verbosity of the output
printed to the terminal (stdout) during execution.  Using the
--summaryonly will limit the verbosity of the terminal printout even
further, to just the final results of scoring the two files.

This version of the scorer does NOT yet implement all the string
matching options discussed in the evaluation plan, such as being
forgiving of small variations in punctuation, initial
determiners/articles, etc. -- those will be added in subsequent
versions. 

The default matching policies should be used, so for most contexts
none of the matching policy arguments (--quadmatchpothat,
--argmatchpolicty, --anchormatchpolicy) should be specified when
invoking.

Below is the documentation for all the arguments available from the
command line.  An example of the scrorer being invoked on some sample
data can be found in sampledata/smalltest.score.out.txt

```bash
unix$ python lib/python3/bptools/scoreAbstractEvents.py -h
Executing lib/python3/bptools/scoreAbstractEvents.py
usage: scoreAbstractEvents.py [-h] [--reffile REFFILE] [--sysfile SYSFILE] [--txtrpt TXTRPT]
                              [--tsvrpt TSVRPT] [--quadmatchpolicy QUADMATCHPOLICY]
                              [--argmatchpolicy ARGMATCHPOLICY]
                              [--anchormatchpolicy ANCHORMATCHPOLICY] [--quiet] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  --reffile REFFILE     A single BP JSON file containing reference annotations (cannot be
                        used with --bpjson, --bpjsondir, --refid or --sysid)
  --sysfile SYSFILE     A single BP JSON file containing system annotations (cannot be used
                        with --bpjson, --bpjsondir, --refid or --sysid)
  --txtrpt TXTRPT       Score report in text format, with alignment details
  --tsvrpt TSVRPT       Score report in TSV format
  --quadmatchpolicy QUADMATCHPOLICY
                        'strict' (both dimenions must match), 'loose' (match on at least one
                        dimension required), 'not-required' (zero, one or both may match),
                        'ignore' (not consulted or weighted). Default=strict
  --argmatchpolicy ARGMATCHPOLICY
                        'strict' (exact string match required), 'loose' (some overlap on
                        args accepted, weighted), 'ignore' (not consulted, not weighted).
                        default=strict
  --anchormatchpolicy ANCHORMATCHPOLICY
                        'strict' (exact string match required for event alignment), 'loose'
                        (some overlap on anchor accepted for alignment), 'ignore' (ignored
                        when exploring event alignments). Default=ignore
  --quiet               If present, don't bother printing scoring information to stdout
                        (default: False)
  --debug               If present, turn on debugging messages to stdout (default: False)
```