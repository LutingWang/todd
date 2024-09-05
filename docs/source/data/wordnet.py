import nltk
from nltk.corpus import wordnet
from nltk.corpus.reader import Synset

nltk.data.path.append('data/wordnet')
synsets: list[Synset] = wordnet.synsets('dog', 'n')
for synset in synsets:
    print(synset)
    print(synset.definition())
    print()
