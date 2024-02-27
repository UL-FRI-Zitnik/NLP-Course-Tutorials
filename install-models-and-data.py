import nltk

# pre-download nltk models and data
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")
nltk.download("reuters")
nltk.download("gutenberg")
nltk.download("wordnet")
nltk.download("tagsets")
nltk.download('punkt')
nltk.download('stopwords')

import classla

# pre-download standard models for Slovenian
classla.download('sl') 

exit(0)