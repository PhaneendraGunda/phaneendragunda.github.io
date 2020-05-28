---
layout: post
title: Machine Learning evaluation metrics
category:
        - nlp
        - metrics

thumbnail: ml_metrics.jpg
---



https://colab.research.google.com/github/huggingface/nlp/blob/master/notebooks/Overview.ipynb



Here are the metrics to evaluate the different tasks,

1. [BLEU](#BLEU)
2. [ROUGE](#ROUGE)
3. [SQUAD](#SQUAD)
4. [SEQEVAL](#SEQEVAL)
5. [GLUE](#GLUE)



### BLEU

* **BLEU** - bilingual evaluation understudy
* It is an algorithm for evaluating the quality of text which has been <u>machine-translated</u> from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" – this is the central idea behind BLEU. BLEU was one of the first metrics to claim a high correlation with human judgements of quality and remains one of the most popular automated and inexpensive metrics.



### ROUGE

- **ROUGE**- Recall-Oriented Understudy for Gisting Evaluation
- It is a set of metrics and a software package used for evaluating <u>automatic summarization and machine translation</u> software in natural language processing. The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.
- The following five evaluation metrics are available.
  - **ROUGE-N**: Overlap of N-gram between the system and reference summaries.
    - ROUGE-1 refers to the overlap of ***unigram*** *(each word)* between the system and reference summaries.
    - ROUGE-2 refers to the overlap of ***bigrams*** between the system and reference summaries.
  - **ROUGE-L**: Longest Common Subsequence based statistics. Longest common subsequence problem takes into account sentence level structure similarity naturally and identifies longest co-occurring in sequence n-grams automatically.
  - **ROUGE-W**: Weighted LCS-based statistics that favors consecutive LCSes .
  - **ROUGE-S**: Skip-bigram based co-occurrence statistics. Skip-bigram is any pair of words in their sentence order.
  - **ROUGE-SU**: Skip-bigram plus unigram-based co-occurrence statistics.



### SQUAD

- **SQUAD** -  <u>Stanford Question Answering Dataset</u>
- It is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or *span*, from the corresponding reading passage, or the question might be unanswerable.
- [Dataset](https://rajpurkar.github.io/SQuAD-explorer/)





### SEQEVAL

- **Seqeval**- Sequence labeling evaluation
- To evaluate tasks such as <u>Named Entity Eecognition (NER)</u>, <u>Part of Speech Tagging (POS)</u>, <u>semantic role tagging</u> etc.



```python
>>> from seqeval.metrics import accuracy_score
>>> from seqeval.metrics import classification_report
>>> from seqeval.metrics import f1_score
>>> 
>>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
>>>
>>> f1_score(y_true, y_pred)
0.50
>>> accuracy_score(y_true, y_pred)
0.80
>>> classification_report(y_true, y_pred)
             precision    recall  f1-score   support

       MISC       0.00      0.00      0.00         1
        PER       1.00      1.00      1.00         1

  micro avg       0.50      0.50      0.50         2
  macro avg       0.50      0.50      0.50         2
```





### GLUE

-  **GLUE** - <u>General Language Understanding Evaluation</u>
-  For evaluting NLU tasks including <u>Question Answering</u>, <u>Sentiment Analysis</u>, <u>Textual Entailment</u> etc..



### GLEU