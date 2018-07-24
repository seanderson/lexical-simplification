# Lexical siplification tools

## Sentence alignment
The repository contains a modified version of a sentence alignment algorithm described by Paetzold and Specia (2016). 
The algorithm can be used to align articles from the Newsela corpus. 
To run the algorithm, use align_all or align_particular methods of the *align.py* module.

*euclidian.py* is a supplementary module used solely within *align.py*. 

*compare.py* provides a set of tools that allow evaluation of the algorithm and executing grid search over some of its parameters.

*alignutils.py* is a set of utilities taht can be used to extract the alignemnts in human readable format.

## Complex Word Identification

*generate_features.py* is a module that allows one to assess the values of various word-level and sentence-level lexical features,
such as the number of synonyms, the number of senses, etc.

*classpaths.py* is a module that stores pthas to various executables, language models, etc. 
This file might have to be modified, if the code is moved to a different machine
