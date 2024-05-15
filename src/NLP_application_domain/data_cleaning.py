import re
from nltk.corpus import stopwords

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    join_list = []
    for w in text.split():
        if w not in STOPWORDS:
            if w == 'television':
                w = 'tv'
            join_list.append(w)
    text = ' '.join(join_list)
    return text

def clean_data(X):
    return [text_prepare(x) for x in X]
