# The Lexicons

This directory contains various lexicons. Every word in the lexicon is tagged with the stanford POS tagger. 

# How to use this data

All the utilities and methods that can be used to create/use this data are located in the lexicons.py module.

## List of files and directories

* metafile.tsv - the list of all the lexicons that are to be used by generate_features.py.
Note that not all of the present lexicons are actually used

* ALL.tsv - the ultimate lexicon that for every word that has been seen at least once in any of the lexicons stores
the information about its presence in any lexicon. 1 - means present, 0 - means absent. This file is created and 
used by lexicons.py

* SAT - a list of 3500 SAT words. The list also includes those words that are not SAT words proper,
are listed in the book as cognates to the original 3500. Finally, some words can represent different
parts of speech in which case both parts of speech are present.  The total number of entries is 
therefore 3885

* OpenOffice - a tagged version of US English dictionary found on https://extensions.openoffice.org/en/project/english-dictionaries-apache-openoffice

* Ogden - various lists of words by Ogden

* GSL

* Fry

* Oxford

* Swadesh

* WordFrequency_info - https://www.wordfrequency.info/

* README.md - this file
