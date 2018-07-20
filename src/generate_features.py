from lexenstein.features import *


fe = FeatureEstimator()
fe.addLengthFeature('Complexity')  # word length
fe.addSynonymCountFeature('Simplicity')  # WordNet synonyms
fe.addWordVectorValues('embeddings_model.bin', 1300, 'Simplicity')