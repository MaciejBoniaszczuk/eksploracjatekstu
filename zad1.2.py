import nltk
import sklearn
import numpy as np
import nltk, string

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from scipy import spatial
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer

stem = PorterStemmer()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

file1 = open("articles/computer.txt", "rt")
file2 = open("articles/genetic-algorithm.txt", "rt")
file3 = open("articles/language.txt", "rt")
file4 = open("articles/life.txt", "rt")
file5 = open("articles/natural-computing.txt", "rt")
file6 = open("articles/programming-language.txt", "rt")
# contents = file.read()
# file.close()

def tylkotekst(file):
    text = file.read()
    file.close()
    return text



def analiza_tekstu(file):
    text = file.read()
    file.close()

    #TOKENIZACJA
    tokenized_word = word_tokenize(text)

    #STOPWORDS
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append(',')
    stopwords.append('.')
    stopwords.append('[')
    stopwords.append(']')
    stopwords.append('{')
    stopwords.append('}')
    stopwords.append('"')
    stopwords.append(';')
    stopwords.append(':')
    stopwords.append('(')
    stopwords.append(')')
    stopwords.append('The')
    stopwords.append('A')
    stopwords.append('In')

    filtered_stopwords_text = []
    for w in tokenized_word:
        if w not in stopwords:
            filtered_stopwords_text.append(w)

    #LEMATYZACJA
    lemmet_words_text = []
    for w in filtered_stopwords_text:
        lemmet_words_text.append(lem.lemmatize(w))

    #WYKRES
    # fdist = FreqDist(lemmet_words_text)
    # fdist.plot(30, cumulative=False,title='30 najczestszych slow z '+file.name)
    # plt.show()





# analiza_tekstu(file1)
# analiza_tekstu(file2)
# analiza_tekstu(file3)
# analiza_tekstu(file4)
# analiza_tekstu(file5)
# analiza_tekstu(file6)


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]



print(cosine_sim(tylkotekst(file1), tylkotekst(file4)))
print(cosine_sim(tylkotekst(file2), tylkotekst(file6)))
print(cosine_sim('Ala ma kota', 'Ala ma psa'))
print(cosine_sim('Ala ma kota', 'Ala ma kota'))
print(cosine_sim('Ala ma kota', 'Ala ma psa'))



