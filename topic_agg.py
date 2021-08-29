import pandas as pd
from operator import itemgetter
from more_itertools import take
import gensim
import os
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud

cols_df = ['full_name', 'description', 'filtered']
prefix_df = 'en_cleaned_all_more'
language = 'jupyter_notebook'
year = '2020'
main_folder = './Experiments_more_5stars/'
ds_folder = '.\\yearly_dataset\\5_stars\\'
domain_legend = ['Classification', 'Prediction', 'Others']

#jn 2019
# bad_topics = [0, 4, 6]
# domains = {'others': [12, 16, 23, 25, 29], 'prediction': [1, 2, 9, 10, 22], 'classification': [3, 5, 7, 8, 11, 13, 14, 15, 17, 18, 19, 20, 21, 24, 26, 27, 28]}


#py 2019
# bad_topics = [1, 18, 24, 28, 29]
# domains = {'ml_related': [2,4, 6, 10, 11, 12, 15, 19, 20, 23], 'tool': [0, 3, 5, 7, 8, 9, 13, 14, 16, 17, 21, 22, 25, 26, 27]}
# domains = {'others': [18, 24, 28, 29], 'ml_related': [2,4, 6, 10, 11, 12, 15, 19, 20, 23], 'tool': [0, 3, 5, 7, 8, 9, 13, 14, 16, 17, 21, 22, 25, 26, 27]}

#jn 2020
bad_topics = [7, 11, 16]
domains = {'others': [4, 5, 18, 20, 22], 'prediction': [1, 2, 24, 27], 'classification': [0, 3, 6, 8, 9, 10, 12, 13, 14, 15, 17, 19, 21, 23, 25, 26, 28]}


#py 2020
# bad_topics = [0, 1, 9, 17, 18, 19, 20, 29]
# domains = {'ml_related': [2, 6, 10, 11, 16, 21, 22, 27, 28], 'tool': [3, 4, 5, 7, 8, 12, 13, 14, 15, 23, 24, 25, 26]}
# domains = {'others': [0, 1, 9, 17, 18, 19, 20, 29], 'ml_related': [2, 6, 10, 11, 16, 21, 22, 27, 28], 'tool': [3, 4, 5, 7, 8, 12, 13, 14, 15, 23, 24, 25, 26]}


foldermodels = f'{main_folder}{language}/{year}/models/'
file = f'{ds_folder}{prefix_df}_{language}_{year}.csv'
file_topics = f'{ds_folder}topic_{prefix_df}_{language}_{year}.csv'


def select_top_topic():

    df = pd.read_csv(file, usecols=cols_df)

    print(f'\n\nDocument {file} - {df.shape}')

    modelfile = ''
    for f in os.listdir(foldermodels):
        if ('.model' in f):
            filetype = f.split('.')[-1]
            if filetype not in ['state','id2word','npy']:
                modelfile = f

    model_lda = gensim.models.ldamodel.LdaModel.load(foldermodels+modelfile)
    curr_dict = gensim.corpora.Dictionary.load(foldermodels+'MultiCore.dict')

    for i, row in df.iterrows():
        docProbs = model_lda[[curr_dict.doc2bow(row['filtered'].split())]]
        topics = {}
        for p in docProbs[0]:
            topics[int(p[0])] = float(round(p[1], 3))

        topics = {k: v for k, v in sorted(topics.items(), key=itemgetter(1), reverse=True)}
        uniques = len(set([*topics.values()]))
        top_key = take(2, topics.keys())

        if uniques > 1:
            if top_key[0] in bad_topics:
                if uniques > 2 and top_key[1] not in bad_topics:
                    df.at[i, 'top_topic'] = top_key[1]
            else:
                df.at[i, 'top_topic'] = top_key[0]

    df = df[df['top_topic'].notna()]

    print(f'\n\nAfter removind bad topics {file} - {df.shape}')
    
    df.to_csv(file_topics, index=False)

def aggr_topic_into_domain():
    df = pd.read_csv(file_topics, usecols=cols_df.append('top_topic'))

    print(f'\n\nDocument {file} - {df.shape}')
    for i, row in df.iterrows():
        for key, item in domains.items():
            if row['top_topic'] in item:
                df.at[i, 'domain'] = key
    
    df.to_csv(file_topics, index=False)

def barplot_domain():
    years = [2019, 2020]
    dfs = []
    for y in years:
        df = pd.read_csv(f'{ds_folder}topic_{prefix_df}_{language}_{y}.csv', usecols=['domain', 'top_topic'])
        # df = df.rename({'domain': 'Domain', 'top_topic': 'Topic'}, axis=1)
        df = df.rename({'domain': 'Domínio', 'top_topic': 'Tópico'}, axis=1)
        # df['Year'] = y
        df['Ano'] = y
        dfs.append(df)
    dfs = pd.concat(dfs)
    # x, y = 'Year', 'Domain'
    x, y = 'Ano', 'Domínio'

    # data = dfs.groupby(x)[y].value_counts(normalize=True).mul(100).rename('Repositories(%)').reset_index()
    data = dfs.groupby(x)[y].value_counts(normalize=True).mul(100).rename('Repositórios(%)').reset_index()
    # data['Domain'] = data['Domain'].map({'classification': 'Classification', 'prediction': 'Prediction', 'others': 'Others', 'ml_related': 'AI & ML', 'tool': 'Tools'})
    data['Domínio'] = data['Domínio'].map({'classification': 'Classificação', 'prediction': 'Predição', 'others': 'Outros', 'ml_related': 'AI & ML', 'tool': 'Ferramentas'})
    sns.set_theme(style="whitegrid", font_scale=2)
    muted=["#4878CF", "#6ACC65", "#D65F5F",
           "#B47CC7", "#C4AD66", "#77BEDB"]
    a4_dims = (11.7, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    # plot = sns.catplot(
    #     data=data, x=x, y='Repositories(%)', hue=y, kind='bar', palette=muted[-3:]
    # ).set(title=language.replace('_', ' ').title())
    plot = sns.catplot(
        data=data, x=x, y='Repositórios(%)', hue=y, kind='bar', palette=muted
    ).set(title=language.replace('_', ' ').title())
    # [-3:]
    for p in plot.ax.patches:
      txt = str(p.get_height().round(1)) 
      txt_x = p.get_x() 
      txt_y = p.get_height()
      plot.ax.text(txt_x,txt_y,txt, size=18)
    #   percentage = str(p.get_height().round(2)) + '%'
    #   x = p.get_x() + p.get_width() / 2 - 0.05
    #   y = p.get_y() + p.get_height()
    #   ax.annotate(percentage, (x, y), size = 12)
    # plt.rcParams["axes.labelsize"] = 1
    plt.savefig(f"catplots_{language}_pt.svg", format="svg", bbox_inches='tight')

def top_words_by_domain(domains, foldermodels):

    modelfile = ''
    for f in os.listdir(foldermodels):
        if ('.model' in f):
            filetype = f.split('.')[-1]
            if filetype not in ['state','id2word','npy']:
                modelfile = f

    model_lda = gensim.models.ldamodel.LdaModel.load(foldermodels+modelfile)

    dict_words_domain = {}

    for domain, topics in domains.items():
        list_dict = []
        for topic in topics:
            words = dict(model_lda.show_topic(topic, 20))
            list_dict.append(words)
        total = sum((Counter(dict(x)) for x in list_dict), Counter())
        dict_words_domain[domain] = {k: v for k, v in sorted(total.items(), key=itemgetter(1), reverse=True)}
    
    return dict_words_domain

def gen_wordcloud(dict_words, language, year):
    plt.figure()
    j = int(np.ceil(len(dict_words)/3))
    i = 0
    for domain, words in dict_words.items():
        i += 1
        plt.subplot(j, 3, i).set_title("Domain " + str(domain))
        plt.plot()
        wordc_img = WordCloud(background_color='white', width=800, height=400, max_words=200).generate_from_frequencies(words)
        plt.imshow(wordc_img)
        svg = wordc_img.to_svg(embed_font=True)
        file_svg = open(f'{language}_{year}_{domain}.svg', 'w', encoding='utf-8')
        file_svg.write(svg)
        file_svg.close()
        plt.axis("off")
    plt.show()


# select_top_topic()
# aggr_topic_into_domain()

# dict_words = top_words_by_domain(domains, foldermodels)

# gen_wordcloud(dict_words, language, year)

# barplot_domain()