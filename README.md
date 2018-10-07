# Lexical simplification tools

## List of files and directories

* newsela - the directory that contains texts from Newsela corpus in various formats. 
The same directory houses data about alignments. See README.md inside the directory for details. 

* googleNGrams - Web 1T 5-gram Version 1 2006 (Google Inc.)

* stanford-parser-full-2015-12-09 - is exactly what it says

* stanford-postagger - June 2018 Stanford postagger. Note: this is not the tagger used to tag the text_databases.
This tagger is used to tag the datasets.

* lexicons - various lexicons to be used by generate_features.py. See README.md inside the directory for details.

* datasets - the three main datasets used for the CWI and LEXSIMP tasks:
    1. native - the data obtained from the alignments of the newsela corpus
    2. kriz - courtesy of Reno Kriz (see Simplification Using Paraphrases and Context-based Lexical Substitution)
    3. cwi - data from the Complex Word Identification Shared Task 2018
    
* text_databases - various text databases that are used to create wor2vec representations of the words. See README.md
inside the directory for details.

* src - All the code except for the code for database processing and Stanford Corenlp utilities. See README.md
inside the directory for details.

* word2vecmodels - models obtained from text_databases

* ghpaetzold-MorphAdornerToolkit-44bb87d - the MorphAdorner tool used by generate_features.py via lexenstein

* .gitignore

* README.md - this file

## Sentence alignment
The repository contains a modified version of a sentence alignment algorithm described by Paetzold and Specia (2016). 
The algorithm can be used to align articles from the Newsela corpus. 
To run the algorithm, use align_all or align_particular methods of the *align.py* module.

*euclidian.py* is a supplementary module used solely within *align.py*. 

*compare.py* provides a set of tools that allow evaluation of the algorithm and execution of the grid search over some of its parameters.

*alignutils.py* is a set of utilities that can be used to extract the alignments in human readable format.

## Complex Word Identification

*generate_features.py* is a module for generating various word- and sentence-level lexical features.
The list of available features: wordnet synonym count, wordnet synset count, word syllable count,
sentence average word syllable count, sentence length (in words), word embedding values, hit frequency, vowel count,
google n-gram count (no smoothing is currently implemented), lexicon feature (assesses whether a word is in a particular set of lexicons)

*lexicons.py* is a model for creating a lexicon file (which is then used by generate features) out of multiple lists of words (SAT, Ogden, Fry, etc.)

*featureClassification.py* - a model that uses keras neural networks and/or scikit learn SVM to solve the CWI problem.

*classpaths.py* is a module that stores paths to various executables, language models, etc.
This file might have to be modified, if the code is moved to a different machine
