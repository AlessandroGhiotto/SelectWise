# **NLP-project**

---

<div align='center'>
<font size="+2">

Text Mining and Natural Language Processing  
2023-2024

<b>SelectWise</b>

Alessandro Ghiotto 513944

</font>
</div>

---

## Dataset

<https://huggingface.co/datasets/allenai/qasc>

Each sample consist of a question, 8 multiple choices and two facts which gives information about the related question.

## Notebooks

### Notebook 1 - Count based methods:

1. Look at the Dataset
2. Representing documents with TF-IDF weighting:
   - Normal tf-idf
   - tf-idf truncated with SVD
   - tf-idf based retrieval from train set
3. n-gram LM based classification

### Notebook 2 - Word Embeddings:

1. Representation by means of static word embeddings:
   - Word2Vec
   - GloVE
   - FastText
   - Doc2Vec
2. Other ways of combining word embeddings:
   - Remove duplicated words
   - Word embeddings weighted by their idf score
3. Other ways of choosing the answer:
   - Siamese Neural Network
   - Feed Forward Neural Network

### Notebook 3 - Transformer Encoder Only:

1. BERT:
   - Binary classification - NextSentencePrediction
   - Multiclass classification - MultipleChoice
2. Different ways of tuning a pretrained models:
   - Linear probing
   - Mixed method

### Notebook 4 - LLM Prompting:

- Zero-Shot Prompting
- Zero-Shot Chain of Thought Prompting
- Few-Shot Prompting
- RAG inspired Few-Shot Prompting

### Notebook 5 - Evaluation on the Test Set:

- BERT - combined method
- DeciLM - few-shot prompting

## Models

In the `./Models` folder I have saved the models trained in the notebooks.

## requirements.txt

In the `requirements.txt` file are present all the libraries used in the environment in which I have run the notebooks (output of the command `pip freeze > requirements.txt`)
