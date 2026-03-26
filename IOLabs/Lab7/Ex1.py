import nltk
from nltk import WordNetLemmatizer
# nltk.download('all')
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud

bbc_text_file = open("BBC Text.txt", 'r')
article = bbc_text_file.read()
tokens = nltk.word_tokenize(article.lower())

# Results of first tokenization
print(tokens)
print(len(tokens))

filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
# Results of second tokenization - erasing stop-words
print(filtered_tokens)
print(len(filtered_tokens))

# print(stopwords.words("english"))
my_stop_words = ['.', ',', '(', ')', "'s", "``", "''"]
new_stop = stopwords.words('english')
for word in my_stop_words:
    new_stop.append(word)
second_filtered_tokens = [token for token in tokens if token not in new_stop]
# Results of third tokenization - erasing , . " etx.
print(second_filtered_tokens)
print(len(second_filtered_tokens))


lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in second_filtered_tokens]
# Results of fourth tokenization - using lemmatizer
print(lemmatized_tokens)
print(len(lemmatized_tokens))

words_vector = nltk.FreqDist(lemmatized_tokens).most_common()
print(words_vector)
ten_most_common = nltk.FreqDist(lemmatized_tokens).most_common(10)
print(ten_most_common)

words = []
quantity = []
for result in ten_most_common:
   words.append(result[0])
   quantity.append(result[1])

plt.bar(words, quantity)
plt.xlabel("Words")
plt.ylabel("Quantity")
plt.show()

cloud_of_words = WordCloud(background_color="white", repeat=True)
cloud_of_words.generate(article.lower())
plt.axis("off")
plt.imshow(cloud_of_words, interpolation="bilinear")
plt.show()
