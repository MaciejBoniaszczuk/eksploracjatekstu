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

tokenized_text = sent_tokenize(contents)
# print("TOKENIZED TEXT ------------------------------------------------------------")
# print(tokenized_text)

tokenized_word = word_tokenize(contents)
print("TOKENIZED WORD ------------------------------------------------------------")
print(tokenized_word)

fdist = FreqDist(tokenized_word)
print(fdist)
print(fdist.most_common(20))
fdist.plot(30, cumulative=False)
plt.show()

stop_words = set(stopwords.words("english"))

# Removing Stopwords
tokenized_sent = word_tokenize(contents)
filtered_sent = []
for w in tokenized_sent:
    if w not in stop_words:
        filtered_sent.append(w)
print("REMOVING STOPWORDS")
# print("Tokenized Sentence:",tokenized_sent)
print("Filterd Sentence:", filtered_sent)

fdist = FreqDist(filtered_sent)
print(fdist.most_common(20))
fdist.plot(30, cumulative=False)
plt.show()

# Stemming

ps = PorterStemmer()

stemmed_words = []
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))
print("STEMMING")
# print("Filtered Sentence:",filtered_sent)
print("Stemmed Sentence:", stemmed_words)

fdist = FreqDist(stemmed_words)
print(fdist.most_common(20))
fdist.plot(30, cumulative=False)
plt.show()

lemmet_words = []
for w in stemmed_words:
    lemmet_words.append(lem.lemmatize(w))
print("Lemmatized Words:", lemmet_words)

fdist = FreqDist(lemmet_words)
print(fdist.most_common(20))
fdist.plot(30, cumulative=False)
plt.show()
