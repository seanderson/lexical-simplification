#!/usr/bin/env bash

set -e

# checking the parameters

if [ $# -ne 2 ]
    then
    echo "This script takes two arguments"
    echo "The first is the name of the folder in which the files to be tagged are located"
    echo "This folder must be located inside /home/nlp/wpred/text_databases/"
    echo "The second parameter is either caseless or casefull - depending on which model to use"
    exit 1
    else
       if [ $2 != "caseless" ] && [ $2 != "casefull" ]
       then
          echo "The second parameter is either caseless or casefull - depending on which model to use"
          exit 1
       fi
fi

path="/home/nlp/wpred/text_databases/$1"
metafile="$path/metafile.txt"
splitfile="$path/splitfile.txt"

find $path -type f \( -name '*' ! -name '*metafile.txt' \) | sort > $metafile

if [ $2 == "caseless" ]
then
   ./stanford-corenlp-full-2018-02-27/corenlp.sh -annotators tokenize,ssplit,pos -pos.model edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger -filelist "$path/metafile.txt" -outputFormat text -outputDirectory "$path"
else
   if [ $2 == "casefull" ]
   then
      ./stanford-corenlp-full-2018-02-27/corenlp.sh -annotators tokenize,ssplit,pos -filelist "$path/metafile.txt" -outputFormat text -outputDirectory "$path"
   fi
fi

echo "merging the files back together, processing them with the python script and deleting the files that are no longer relevant"

cat $metafile | while read line
do
   find $path -maxdepth 1 -type f -path "$line?out"  | sort | python process.py
   mv "$line.out" "$line.possf2"
done
rm "$path/metafile.txt"