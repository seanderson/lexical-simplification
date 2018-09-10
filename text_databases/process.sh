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

echo "splitting large files..."

find $path -type f \( -name '*' ! -name '*metafile.txt' \)> $metafile
cat $metafile | while read line
do
   newname="$line.split"
   split -C 20M -d -a 3 $line $newname
done
find $path -type f -name '*\.split*' | sort > $splitfile

echo "tagging the files..."
# Caseless model should be used for databases like SubIMDB

if [ $2 == "caseless" ]
then
   ./stanford-corenlp-full-2018-02-27/corenlp.sh -annotators tokenize,ssplit,pos -pos.model edu/stanford/nlp/models/pos-tagger/english-caseless-left3words-distsim.tagger -filelist "$path/splitfile.txt" -outputFormat text -outputDirectory "$path"
else
   if [ $2 == "casefull" ]
   then
      ./stanford-corenlp-full-2018-02-27/corenlp.sh -annotators tokenize,ssplit,pos -filelist "$path/splitfile.txt" -outputFormat text -outputDirectory "$path"
   fi
fi

echo "merging the files back together, processing them with the python script and deleting the files that are no longer relevant"

cat $metafile | while read line
do
   newname="$line.split"
   finalname="$line.possf2"
   find $path -maxdepth 1 -type f -path "$newname????out"  | sort | python process.py
   find $path -maxdepth 1 -type f -path "$newname????out"  | sort | xargs cat > $finalname
   rm "$line.split"*
done
rm "$path/metafile.txt"
rm "$path/splitfile.txt"