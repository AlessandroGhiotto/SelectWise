# SelectWise

Natural Language Processing project on multiple choice question answering on the [QASC dataset](https://huggingface.co/datasets/allenai/qasc).

Each item consists of a question, eight multiple choices (from 'A' to 'H') and two facts that provide information
about the question, the task being to select the correct answer. The multiple-choice task can be seen as a
single-label, multi-class classification between the eight possible alternatives.

## Description

### Models

The models can be downloaded at the followin Google Drive link:

https://drive.google.com/file/d/1C_AeDhgTwvPTqQOXj01beEMnr6Kkqf2_/view?usp=sharing

Unzip the file and move the 'models' folder into the 'SelectWise' folder. In the `./models` folder I have saved the models trained in the notebooks. This models can be loaded and used for evaluation.

### Images

In the `./imgs` folder I have saved the images which are opened in the notebooks.

## Notebooks

The notebooks can be just run sequentially cell by cell. The dataset is loaded with `load_dataset("allenai/qasc")` thanks to the hugging face library, so it's not necessary to download the dataset.

In the notebooks the models are loaded with the following path: `'../models/model_name'`

In the notebooks the images are inserted in this way: `![picture](../imgs/img_name)`

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

## Dependencies

- `requirements.txt` for pip
- `requirements.yml` for anaconda

## References

```bibtex
 @inproceedings{Khot2019QASC,
   title={{QASC}: A Dataset for Question Answering via Sentence Composition},
   author={Tushar Khot and Peter Clark and Michal Guerquin and Peter Alexander Jansen and Ashish Sabharwal},
   booktitle={AAAI},
   year={2019},
  }
```
