# lda_topics_metaheuristics

NLP pipeline, Topic Classification and multicore hyperparameter tuning algorithms developed for the research 
[__"How COVID-19 Impacted Data Science: a Topic Retrieval and Analysis from GitHub Projects' Descriptions"__](https://sol.sbc.org.br/index.php/sbbd/article/view/17893) (presented at the [Brazilian Symposium On Databases 2021 (SBBD)](https://sbbd.org.br/2021/))

This work compares topics of interest from Data Science projects and their evolution over the COVID-19 pandemic period by analyzing Jupyter Notebook 
and Python GitHub projects from a year before and during the pandemic. We employ various state-of-art algorithms to find topics based on the repositories descriptions, 
and compare their performance for tuning the topic classification model hyperparameters for better accuracy.

The research dataset is also available on Zenodo: [Greed:  Github repositories and descriptions](https://www.doi.org/10.5281/zenodo.5138079)

## Libraries and Algorithms
* [pylang](https://pypi.org/project/pylang/)
* [spaCy](https://spacy.io/)
* [gensim](https://radimrehurek.com/gensim/index.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/)
* [seaborn](https://seaborn.pydata.org/)
* [numpy](https://numpy.org/)
* Term Frequency and Inverse Document Frequency (TF-IDF)
* Latent Dirichlet Allocation (LDA)
* Differential Evolutionary (DE) and its Self Adaptive version (SADE)
* Genetic Algorithm (GA)
* Particle Swarm Optimization (PSO) and its Generational version (GPSO) 
* Simulated Annealing (SA)

## Implementation

The `Topic Aggregation_5stars.ipynb` file is a Jupyter Notebook document that presents steps while aggregating topics into domains.

More details in the [research paper](https://sol.sbc.org.br/index.php/sbbd/article/view/17893)
