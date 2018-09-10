# Text databases

This directory contains multiple text databases as well as some tools for their processing. 

## List of databases
* NewsCrawl (http://www.statmt.org/wmt11/translation-task.html -> http://www.statmt.org/wmt11/training-monolingual.tgz)
* SubIMDB (http://ghpaetzold.github.io/subimdb/ -> http://www.quest.dcs.shef.ac.uk/subimdb/SubIMDB_All_Compiled.tar.gz),
* Subtlex (http://subtlexus.lexique.org/ -> http://subtlexus.lexique.org/corpus/Subtlex%20US.rar),
* UMBC_webbase (https://ebiquity.umbc.edu/resource/html/id/351 -> http://swoogle.umbc.edu/umbc_corpus.tar.gz),
* Kauchak Wikipedia (http://www.cs.pomona.edu/~dkauchak/simplification/data.v2/document-aligned.v2.tar.gz)

Each of these folders contains three types of files. There are original files 
as obtained from the web. These can come in various formats. 
(For Kauchak Wikipedia, the files were splitted so that each file represents
a separate article. So, there are two versions of this databse: KW_mult_files and KW_two_files). 
(For NewsCrawl, only files in english were used, others were discarded)
There are .possf2 files which are the output of the coreNLP tagging
algorithm. Finally, there are .clean files, which are versions of .possf2 files
with html tags being removed. Each directory also contains a metafile.txt file
created with find $(pwd) -name '*.txt.possf2.cleaned' > metafile.txt command.

## Other files

* metafile.txt - List of all of the other metafile.txt files (one for each database)

* Kauchak_original.tar.gz - the original files for this database (with article
and paragpah information being preserved)

* stanford-corenlp-full-2018-02-27 - stanford coreNLP tools used for tagging

* process.sh & process.py - scripts that allow a new database to be tagged 
if teh need arises. The bash script calls teh python script

* process_small_files.sh - a version of process.sh that should only be used
when it is certain that no file in the database is larger than approx. 20 MB

* utils.py - some additional utilities (including the option of building
a vocabulary)

* voc.tsv - sample vocabulary files

* README.md - this file
