from wordcloud import WordCloud, STOPWORDS
from operator import itemgetter
from document_processor import *
import matplotlib.pyplot as plt

filter_terms = {'python', 'repo', 'repository', 'jupyter', 'github', 'notebook', 'python3', 'ipynb', 'py', 'source', 'code', 'use', 'base', 'et', 'al', 'etc', 'like', 'implementation', 'non', 'contain', 'implement', 'likely'}

def sort_dict_asc(dict_to_sort):
    return {k: v for k, v in sorted(dict_to_sort.items(), key=itemgetter(1))}

def diff_dict(first_dict, second_dict):
    return {k : first_dict[k] for k in set(first_dict) - set(second_dict)}

def intersection_dict(first_dict, second_dict):
    return {k: first_dict[k] + second_dict[k] for k in first_dict.keys() & second_dict.keys()}

def simple_wordcloud(texts, popfirst, filename, filter, max_words=100):
    stopwords = set(STOPWORDS)
    stopwords.update(filter)
    wordcloud_class = WordCloud(background_color='white', width=1600,                            
                        height=800, max_words=max_words, stopwords=stopwords, normalize_plurals=False, colormap=plt.get_cmap('viridis'))

    all_texts = " ".join(str(text).lower() for text in texts)
    freq = sort_dict_asc(wordcloud_class.process_text(all_texts))
    print("terms dict len: {0}".format(len(freq)))
    
    for _ in range(popfirst):
        freq.popitem()

    wordcloud_img = wordcloud_class.generate_from_frequencies(freq)
    fig, ax = plt.subplots(figsize=(16,8))            
    ax.imshow(wordcloud_img, interpolation='bilinear')       
    ax.set_axis_off()
    plt.imshow(wordcloud_img)                 
    wordcloud_img.to_file(filename)

    words = ''
    for _ in range(max_words):
        it = freq.popitem()
        words += it[0] + ', '
    print(words)

def simple_intersection_wordcloud(first_texts, second_texts, popfirst, filename, filter, max_words=100):
    stopwords = set(STOPWORDS)
    stopwords.update(filter)
    wordcloud_class = WordCloud(background_color='white', width=1600,                            
                        height=800, max_words=max_words, stopwords=stopwords, normalize_plurals=False, colormap=plt.get_cmap('viridis'))

    all_first_texts = " ".join(text.lower() for text in first_texts)
    all_second_texts = " ".join(text.lower() for text in second_texts)

    freq_first = sort_dict_asc(wordcloud_class.process_text(all_first_texts))
    freq_second = sort_dict_asc(wordcloud_class.process_text(all_second_texts))

    final_freq = sort_dict_asc(intersection_dict(freq_first, freq_second))
    print("terms dict len: {0}".format(len(final_freq)))

    for _ in range(popfirst):
        final_freq.popitem()

    wordcloud_img = wordcloud_class.generate_from_frequencies(final_freq)

    fig, ax = plt.subplots(figsize=(16,8))            
    ax.imshow(wordcloud_img, interpolation='bilinear')       
    ax.set_axis_off()
    plt.imshow(wordcloud_img)                 
    wordcloud_img.to_file(filename)

    words = ''
    for _ in range(max_words):
        it = final_freq.popitem()
        words += it[0] + ', '
    print(words)

def generate_wordcloud(terms, filename, max_words=100):
    wordcloud_class = WordCloud(background_color='white', width=1600,                            
                        height=800, max_words=max_words, colormap=plt.get_cmap('viridis'))

    wordcloud_img = wordcloud_class.generate_from_frequencies(terms)
    fig, ax = plt.subplots(figsize=(16,8))            
    ax.imshow(wordcloud_img, interpolation='bilinear')       
    ax.set_axis_off()
    plt.imshow(wordcloud_img)                 
    wordcloud_img.to_file(filename)
    words = ''
    for _ in range(max_words):
        it = terms.popitem()
        words += it[0] + ', '
    print(words)

def processed_wordcloud(texts, pop_first, filename):
    terms = get_document_ngram_dict(get_documents_lemmatization(filter_stopwords(preprocess_document_text(texts), filter_terms)))
    print("terms dict len: {0}".format(len(terms)))

    for _ in range(pop_first):
        terms.popitem()

    generate_wordcloud(terms, filename)

def tfidf_wordcloud(texts, pop_first, filename, min_value=0.3, max_value=0.8):
    terms = sort_dict_asc(get_terms_filtered_by_TF_IDF(texts, filter_terms, min_value, max_value))
    print("terms dict len: {0}".format(len(terms)))

    for _ in range(pop_first):
        terms.popitem()

    generate_wordcloud(terms, filename)

def intersection_wordcloud_TF_IDF(first_texts, second_texts, pop_first, filename, min_value=0.3, max_value=0.8):
    lemma_first = get_documents_lemmatization(filter_stopwords(preprocess_document_text(first_texts), filter_terms))
    lemma_second = get_documents_lemmatization(filter_stopwords(preprocess_document_text(second_texts), filter_terms))

    ngram_first = get_document_ngram_dict(lemma_first)
    ngram_second = get_document_ngram_dict(lemma_second)

    print("first terms dict len: {0}".format(len(ngram_first)))
    print("second terms dict len: {0}".format(len(ngram_second)))

    intersection = intersection_dict(ngram_first, ngram_second)
    joined_lemma = lemma_first + lemma_second
    filtred_intersection = sort_dict_asc(filter_by_global_TF_IDF(joined_lemma, intersection, min_value, max_value))
    print("intersection terms dict len: {0}".format(len(filtred_intersection)))

    for _ in range(pop_first):
        filtred_intersection.popitem()

    generate_wordcloud(filtred_intersection, filename)

def difference_wordcloud_TF_IDF(first_texts, second_texts, pop_first, filename, min_value=0.3, max_value=0.8):
    lemma_first = get_documents_lemmatization(filter_stopwords(preprocess_document_text(first_texts), filter_terms))
    lemma_second = get_documents_lemmatization(filter_stopwords(preprocess_document_text(second_texts), filter_terms))

    ngram_first = get_document_ngram_dict(lemma_first)
    ngram_second = get_document_ngram_dict(lemma_second)

    print("first terms dict len: {0}".format(len(ngram_first)))
    print("second terms dict len: {0}".format(len(ngram_second)))
    difference = sort_dict_asc(filter_by_global_TF_IDF(lemma_first, diff_dict(ngram_first, ngram_second), min_value, max_value))
    print("difference terms dict len: {0}".format(len(difference)))

    for _ in range(pop_first):
        difference.popitem()

    generate_wordcloud(difference, filename)