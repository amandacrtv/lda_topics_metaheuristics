import emoji
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import dok_matrix
import spacy
from gensim import corpora, models
from gensim.models import Phrases
from gensim.models.phrases import Phraser

from spacy.lang.en.stop_words import STOP_WORDS

lemma_keys = {"pyt": "pytest", "datum": "data"}
bigram_keys = {"machine learn": "machine learning", "deep learn": "deep learning"}

def stopwords_from_txt():
    path = './stopwords.txt'
    file = open(path, 'r', encoding="utf8")
    txt_stopwords = [line.rstrip('\n') for line in file]
    file.close()
    return set(txt_stopwords)

def remove_non_ASCII(text):
    return text.encode().decode('ascii', 'replace').replace(u'\ufffd', ' ')

def remove_frozen_symbols(text):
    symbols = '()[]{\}'
    for i in range(len(symbols)):
        text = text.replace(symbols[i], ' ')
        text = text.replace('  ', ' ')
    return text

def remove_url(text):
    return re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)

def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u' ', text)

def remove_single_characters(text):
    return [token for token in text if len(token) > 1]

def remove_numbers(text):
    return [token for token in text if not token.isnumeric()]

def only_remove_URL_emoji(text):
    text = text.lower()
    text = remove_frozen_symbols(text)
    text = remove_url(text)
    text = remove_emoji(text)
    text = remove_non_ASCII(text)
    text = text.replace('  ', ' ')
    return text

def remove_punctuation(text):
    symbols = "!\",'#$%&*+-./:;<=>?@\^_`|~’\n"
    for i in range(len(symbols)):
        text = text.replace(symbols[i], ' ')
        text = text.replace('  ', ' ')
    return text

def preprocess_document_text(document):
    return [remove_punctuation(only_remove_URL_emoji(str(text).lower())) for text in document]

def filter_stopwords(document, stopwords):
    return [' '.join([word for word in text.split() if word not in stopwords]) for text in document]

def get_documents_lemmatization(documents):
    sp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
    filtered_documents = []
    for doc in sp.pipe(documents):
        phrase = ''
        for token in doc:
            lemma = token.lemma_ if token.lemma_ not in lemma_keys.keys() else lemma_keys[token.lemma_]
            phrase += lemma + ' '
        filtered_documents.append(phrase)
    return filtered_documents
    
def process_after_lemma(documents):
    documents = [only_remove_URL_emoji(text) for text in documents]

    filtered_stopwords = stopwords_from_txt()
    # filtered_stopwords = STOP_WORDS
    # filtered_stopwords |= set(['based', 'base'])
    
    documents = filter_stopwords(documents, filtered_stopwords)

    documents = get_documents_lemmatization(documents)

    documents = [remove_punctuation(text) for text in documents]
    
    return filter_stopwords(documents, filtered_stopwords)

def fix_keys(documents):
    _keys = {"machine_learn": "machine_learning", "deep_learn": "deep_learning"}
    return [[_keys[word] if word in _keys.keys() else word for word in doc] for doc in documents]

def get_documents_bigram(documents, minc=5):
    documents = [[word for word in doc.split()] for doc in documents]
    bigrams = Phrases(documents, min_count=minc)
    bigram_mod = Phraser(bigrams)
    data_words_bigrams = fix_keys([bigram_mod[doc] for doc in documents])
    documents = [' '.join(remove_numbers(remove_single_characters(doc))) for doc in data_words_bigrams]
    return documents

def get_document_ngram_dict(document, rangen, wspace = True):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, rangen))

    ngram_vector = vectorizer.fit_transform(document)
    feature_names = vectorizer.get_feature_names()
    dok_matrix_ngram = dok_matrix(ngram_vector)
    ngram_dict = {}

    for k in dok_matrix_ngram.keys():
        ngram_dict[feature_names[k[1]]] = ngram_dict.setdefault(feature_names[k[1]], 0) + 1
    
    for key, value in bigram_keys.items():
        if key in ngram_dict.keys() and value in ngram_dict.keys():
            ngram_dict[value] += ngram_dict[key]
            ngram_dict.pop(key)
        elif key in ngram_dict.keys():
            ngram_dict[value] = ngram_dict.pop(key)

    if not wspace:
        oth = {}
        for key, item in ngram_dict.items():
            oth[key.replace(' ', '_')] = item
        return oth
    
    return ngram_dict

def get_document_tfidf_dict(document):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(document)
    feature_names = vectorizer.get_feature_names()
    dok_matrix_tfidf = dok_matrix(vectors)
    return {feature_names[k[1]]: v  for k, v in dok_matrix_tfidf.items()}

def filter_by_individual_TF_IDF(documents, min_value=0.2, max_value=0.8):
    docs = [doc.split() for doc in documents]

    dictionary = corpora.Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = models.TfidfModel(corpus, id2word=dictionary)
    new_docs = []

    for bow in corpus:
        current_doc_vals = []
        current_doc_ids = []
        for id, value in tfidf[bow]:
            current_doc_ids.append(id)
            current_doc_vals.append(value)
        if any(val < min_value for val in current_doc_vals) or any(val > max_value for val in current_doc_vals):
            words_filtered = []
            for i, val in enumerate(current_doc_vals):
                if val > min_value and val < max_value:
                    words_filtered.append(current_doc_ids[i])
            new_docs.append(' '.join([dictionary[id] for id in words_filtered]))
        else:
            new_docs.append(' '.join([dictionary[id] for id in current_doc_ids]))
    
    return new_docs

    
def filter_by_global_TF_IDF(descriptions_lemma, descriptions_ngram, min_value, max_value):
    TFIDF_dict = get_document_tfidf_dict(descriptions_lemma)
    TFIDF_keys = [k for k, v in TFIDF_dict.items() if v <= max_value and v > min_value]

    final = {}

    for k, v in descriptions_ngram.items():
        split_key = k.split()
        if len(split_key) > 1:
            for splited_key in split_key:
                if splited_key in TFIDF_keys:
                    final[k] = v
                    break
        elif k in TFIDF_keys:
            final[k] = v

    return final
    #return {k: v for k, v in descriptions_ngram_dict.items() if k in TFIDF_keys}

def get_terms_filtered_by_TF_IDF(descriptions, stopwords, min_value, max_value):
    descriptions_lemmatization = get_documents_lemmatization(filter_stopwords(preprocess_document_text(descriptions), stopwords))
    descriptions_ngram_dict = get_document_ngram_dict(descriptions_lemmatization)
    return filter_by_global_TF_IDF(descriptions_lemmatization, descriptions_ngram_dict, min_value, max_value)

