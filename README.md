## NLP Transformers with Python

Transformer models are the de-facto standard in modern NLP. They have proven themselves as the most expressive, powerful models for language by a large margin, beating all major language-based benchmarks time and time again. This repo covers content from a Udemy course, which you can access [here](https://ibm-learning.udemy.com/course/nlp-with-transformers/learn/lecture/26466724#overview).

This repo consists of projects using key NLP frameworks such as:

- HuggingFace's Transformers
- TensorFlow 2
- PyTorch
- spaCy
- NLTK
- Flair

Projects on NLP use-cases:

- Language classification/sentiment analysis
- Named entity recognition (NER)
- Question and Answering
- Similarity/comparative learning

### Project1: Language classification/sentiment analysis

The project1 is a sentimental analysis classifier using Transformers and TensforFlow. This projects runs through the standard steps required to build a NLP model, namely, :

- Data preprocessing: Getting data from the Kaggle API, transform data to preprare it for sentimental analysis and tokenization
- Tf input pipeline: Build dataset (shuffle, batch, split data) for tensorflow
- Modellng and training: Initialise BERT model and define the architecture (the inpout layers, mask layer, embedding layers, output layer shapes, max pooling, activation layer, etc), set up the optimizer, loss function and evaluation metric. Train and save the model
- Getting predictions: Load the trained model, tokenise test data and make predictions

### Project2: Named entity recognition (NER) + sentiment analysis

The Named Entity Recognition (NER) folder covers an introduction to NER using spaCy. The main idea here is to use NER to identify and classify named entities in text into predefined categories such as persons, organizations, quantities, etc. This is crucial in understanding specific pieces of information from unstructured text. I followed the follwoing approach in this project:

- Extract stock information from a investing subreddit. This is done by sending a reqquets to the Reddit API. Communication with the API was authenticated using an auth bearer token. The POST method was used to extract the subreddit. The data is then stored in a pandas dataframe

- spacy's 'en_core_web_sm' model is initialized and used to extract ORG  entities. Some pre-processing is done to remove govn organizations (non-stock orgs).

- Next, sentiment analysis is performed on the reddit threads containing infromation about each ORG entity. Finally, the average sentiment score, average positive/negative score per organization is calculated.
