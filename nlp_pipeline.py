from text_classifier import classify_lang, filter_by_row
from document_processor import get_documents_bigram, process_after_lemma, filter_by_individual_TF_IDF
import numpy as np
import time
from datetime import timedelta
import pandas as pd 

language = 'python'

year = '2020'

minc = 10

path = 'yearly_dataset/5_stars/'

filename = f"{language}_{year}.csv"

classified_lang = 'en_'

prefix = 'cleaned_all_more_'

cols = ['id', 'full_name', 'description', 'updated_at']

filter_words = ['course', 'courses', 'assignment', 'assignments', 'homework', 'homeworks', 'tutorial', 'tutorials', 'workshop', 'workshops', 'exercise', 'exercises', 'lesson', 'lessons', 'udacity', 'bootcamp', 'bootcamps', 'kaggle', 'coursera', 'udemy', 'competition', 'competitions', 'hackathon', 'hackathons', 'hacktoberfest', 'challenge', 'challenges', 'hands-on', 'practice', 'practices', 'material', 'materials', 'class', 'classroom', 'introduction', 'hackerrank', 'euler', 'solution', 'solutions', 'leetcode']

def classify_lang_preprocess_texts():
    print('LANGUAGE CLASSIFICATION')
    df = pd.read_csv(path+filename, usecols=cols)

    print(f"\nDocument {filename} - {df.shape}")

    df = classify_lang(df, 'description', filename)

    df = filter_by_row(df, 'lang', 'en')

    print(f"after filter by lang: {df.shape}\n")

    df.to_csv(path + classified_lang + filename, index=False) 

def filter_repos_course(name, save_name, column='description'):
    print('FILTER NOT SOFTWARE')
    df = pd.read_csv(path+name, usecols=cols.append(column))

    print(f"\nDocument {name} - {df.shape}")

    final = []
    for text in df[column]:
        aux = []
        for word in str(text).lower().split():
            if word in filter_words:
                aux.append(word)
                break
        final.append(" ".join(aux))

    df['not_software'] = final

    print(df.groupby(['not_software']).size())

    df = df[df['not_software'] == '']

    print(f"After cleaning - {df.shape}\n")

    df = df.drop('not_software', 1)

    df.to_csv(path + save_name, index=False) 

def lemmatization_stopword_characters_removal():
    print('LEMMATIZATION, STOPWORD REMOVAL, PUNCTUATION REMOVAL')
    df = pd.read_csv(path+classified_lang+prefix+filename, usecols=cols)

    print(f"\nDocument {classified_lang+prefix+filename} - {df.shape}")

    df['cleaned'] = process_after_lemma(df['description'])

    df = df[df['cleaned'].str.strip().astype(bool)]

    print(f"After cleaning lemma and stopwords {classified_lang+prefix+filename} - {df.shape}\n")

    df.to_csv(path + classified_lang+prefix+filename, index=False) 

def find_bigrams_lemma():
   print('GROUP BIGRAMS')
   df = pd.read_csv(path+classified_lang+prefix+filename, usecols=cols.append('cleaned'))

   print(f"\nDocument {classified_lang+prefix+filename} - {df.shape}")

   df['cleaned'] = get_documents_bigram(df['cleaned'], minc)

   df = df[df['cleaned'].str.strip().astype(bool)]

   print(f"after grouping bigrams filtering single digits {df.shape}\n")

   df.to_csv(path + classified_lang+prefix+filename, index=False)

def filter_by_tf_idf():
    print('FILTER BY TF-IDF')
    df = pd.read_csv(path+classified_lang+prefix+filename, usecols=cols.append('cleaned'))
    print(f"\nDocument {classified_lang+prefix+filename} - {df.shape}")

    df['filtered'] = filter_by_individual_TF_IDF(df['cleaned'])

    df = df[df['filtered'].str.strip().astype(bool)]

    print(f"after filter by low/high TF-IDF: {df.shape}\n")
    df.to_csv(path + classified_lang+prefix+filename, index=False)

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]

def filter_outliers():
    print('FILTER OUTLIERS')
    df = pd.read_csv(path+classified_lang+prefix+filename, usecols=cols.extend(['cleaned', 'filtered']))

    print(f"\nDocument {classified_lang+prefix+filename} - {df.shape}")

    df['count_words'] = np.array([len(doc.split()) for doc in df['filtered']])
    print('Count words before removing outliers')
    print(df.groupby(['count_words']).size())
    # df['count_words'] = reject_outliers(df['count_words'])
    df = df.loc[(df['count_words'] > 1) & (df['count_words'] < 9)]
    # df = df.loc[(df['count_words'] > 3) & (df['count_words'] < 7)]
    print('Count words after removing outliers')
    print(df.groupby(['count_words']).size())

    df = df[df['count_words'].notna()]
    df = df.drop('count_words', 1)

    print(f"After cleaning outliers: {df.shape}\n")
    df.to_csv(path + classified_lang+prefix+filename, index=False)

def pipeline_repos():

    t1 = time.time()

    classify_lang_preprocess_texts()

    filter_repos_course(classified_lang+filename, classified_lang+prefix+filename, column='description')

    lemmatization_stopword_characters_removal()

    filter_repos_course(classified_lang+prefix+filename, classified_lang+prefix+filename, column='cleaned')

    find_bigrams_lemma()

    filter_by_tf_idf()

    filter_outliers()

    t2 = time.time()

    print(f'\n\nTotal time for {filename} : {timedelta(seconds=(t2 - t1))}\n\n')

