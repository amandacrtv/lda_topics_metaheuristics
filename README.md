# lda_topics_metaheuristics

NLP pipeline, Topic Classification and hyperparameter tuning algorithms developed for the research 
"How COVID-19 Impacted Data Science: a Topic Retrieval and Analysis from GitHub Projects' Descriptions" _(paper under process for publication, 
presented on [Brazilian Symposium On Databases 2021 (SBBD)](https://sbbd.org.br/2021/))_

This work compares topics of interest from Data Science projects and their evolution over the COVID-19 pandemic period by analyzing Jupyter Notebook 
and Python GitHub projects from a year before and during the pandemic. We employ various state-of-art algorithms to find topics based on the repositories descriptions, 
and compare their performance for tuning the topic classification model hyperparameters for better accuracy.

The research dataset is also available on Zenodo: [Greed:  Github repositories and descriptions](https://www.doi.org/10.5281/zenodo.5138079)

## Libraries and Algorithms
* [pylang](https://pypi.org/project/pylang/)
* [spaCy](https://spacy.io/)
* [gensim](https://radimrehurek.com/gensim/index.html)
* Term Frequency and Inverse Document Frequency (TF-IDF)
* Latent Dirichlet Allocation (LDA)
* Differential Evolution-ary (DE) and its Self Adaptive version (SADE)
* Genetic Algorithm (GA)
* Particle Swarm Optimization (PSO) and its Generational version (GPSO) 
* Simulated Annealing (SA)

## Implementation

The `Topic Aggregation_5stars.ipynb` file is a Jupyter Notebook document that presents steps while aggregating topics into domains.

More details on the research paper _(paper under process for publication)_
