# Contains all the path variables that need to be configured before running
# the program on a new computer

import sys

USERDIR = '/home/nlp'
if sys.platform == 'darwin':
    USERDIR = '/Users/alexanderfedchin'
BASEDIR = USERDIR + '/newsela'
METAFILE = BASEDIR + '/articles_metadata.csv'
PARSERDIR = BASEDIR + '/stanford-parser-full-2015-12-09/'
OUTDIR_SENTENCES = BASEDIR+'/output/sentences/'
OUTDIR_PARAGRAPHS = BASEDIR+'/output/paragraphs/'
OUTDIR_NGRAMS = BASEDIR+'/output/ngrams/'
OUTDIR_PRECALCULATED = OUTDIR_NGRAMS+'ngramsByFile/'
OUTDIR_TO_DELETE = OUTDIR_NGRAMS+'toDelete/'
OUTDIR_TOK_NGRAMS = OUTDIR_NGRAMS+'tokenizedForNgrams/'
OUTDIR_PERPLEX = OUTDIR_NGRAMS+'perplexity/'
MANUAL_SENTENCES = BASEDIR+'/manual/sentences/new_format/'
MANUAL_PARAGRAPHS = BASEDIR+'/manual/paragraphs/'

PARSERPROG = 'custom/Parser'
TOKENIZERPROG = 'custom/Tokenizer'

MODELS = 'edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'

CLASSPATH = ':'.join(['.',PARSERDIR,PARSERDIR + 'stanford-parser.jar', PARSERDIR + 'stanford-parser-3.6.0-models.jar', PARSERDIR + 'slf4j-api.jar'])

PREDICTIONS = BASEDIR + "/predictions/"
NN_MODELS = BASEDIR + "/models/"
DEFAULT_MODEL_NAME = PREDICTIONS + "Best02-srn-63-3.01-probs.h5"
nnetFile = BASEDIR + "/data/test/NoOverlapRawTest.pbz2"
indexFile = BASEDIR + "/data/test/NoOverlapRawTest.idx"

CHRIS_PAPER_FILE = USERDIR + \
                   "/corpora/newsela_complex/Newsela_Complex_Words_Dataset.txt"

MORPH_ADORNER_TOOLKIT = BASEDIR + "/ghpaetzold-MorphAdornerToolkit-44bb87d/"    # can be downloaded from
    # http://ghpaetzold.github.io/MorphAdornerToolkit/
NEWSELA_COMPLEX = USERDIR + "/corpora/newsela_complex/"
