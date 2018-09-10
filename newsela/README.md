# The NEWSELA corpus 

## How to use this data

* alignutils.py - is a module that can extract the alignment data in a more user-friendly format
* align.py - is a module for generating alignments
* compare.py - is a module for comparing manual and automatic alignments

## List of files and directories

* articles - the directory that contains the articles from the Newsela corpus. 
Each article is presented in three formats :
    1. .txt - the original text
    2. .tok - the text processed by the Stanford tokenizer (see tokenize_and_parse.py). 
    Most python modules use the .tok files.
    3. .prs - [UNUSED] - Parse trees for the .tok files created by the Stanford parser (see tokenize_and_parse.py)
    
* articles_metadata.csv - the metadata file that describes files in the articles/ directory

* manual - contains the results of the manual alignments. This directories contains three subfolders:
    1. paragraphs - each file in this folder contains the results of the paragraph alignment of two articles. 
    Every line (except for the first one) represents one alignment. It contains the list of the paragraphs from the 
    first article separated by commas, and then, after two \t symbols, the same list for the second article. 
    All indexes are 1-based.
    2. sentences: each file in this folder contains the results of the sentence alignment of two articles. 
    Every line (except for the first one) represents one block of alignments. Each alignment in the block is separated 
    from the others with a /t symbol. Each alignment consists of two sentences, the data about which is separated with
    a comma. Each sentence is a pair of two values (paragraph and sentence indexes) separated from each other with 
    a semicolon. 
    3. xmls - xml files that are convenient-to-read versions of tokenized articles.
    
* alignments - the folder with automatically generated alignments. Its structure mimcs that of the folder with the
manual alignments

* README.md - this file
