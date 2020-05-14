import numpy as np
import pandas as pd

df = pd.read_csv("emails.csv")

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
for i in range(5727):
    df['text'][i] = re.sub('[^a-zA-Z]',' ',df['text'][i])
    df['text'][i] = df['text'][i].lower()
    df['text'][i] = df['text'][i].split()[1:]
    ps = PorterStemmer()
    df['text'][i] = [ps.stem(word) for word in df['text'][i] if not word in stopwords.words('english')]
    df['text'][i] = ' '.join(df['text'][i])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(df['text']).toarray()
y = df.iloc[:,1]

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x,y)

y_pred = classifier.predict(x)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)
print(cm)