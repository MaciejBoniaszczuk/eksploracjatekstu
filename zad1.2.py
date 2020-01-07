import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer

stem = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

file = open("articles/computer.txt", "rt")
contents = file.read()
file.close()

#COMPUTER.TXT

#TOKENIZACJA
tokenized_word = word_tokenize(contents)

#STOPWORDS
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append(',')
stopwords.append('.')
stopwords.append('[')
stopwords.append(']')
stopwords.append('(')
stopwords.append(')')
stopwords.append('The')
stopwords.append('A')
stopwords.append('In')
print(stopwords)
filtered_stopwords_computer = []
for w in tokenized_word:
    if w not in stopwords:
        filtered_stopwords_computer.append(w)

#LEMATYZACJA
lemmet_words_computer = []
for w in filtered_stopwords_computer:
    lemmet_words_computer.append(lem.lemmatize(w))

#WYKRES
fdist = FreqDist(lemmet_words_computer)
fdist.plot(30, cumulative=False)
plt.show()
