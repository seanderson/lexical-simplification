# Lexical siplification tools

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
