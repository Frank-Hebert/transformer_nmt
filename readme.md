# Transfer Learning on Low-Ressource Language Pairs for NMT

**By Yassine Kassis, Francois Hébert, Yu Chen**

## Description

The main objective of this project was to reproduce the transfer learning method used by Kocmi \& Bojar and to study in depth how the amount of training of the pre-trained model affects the final performance. The technique consists of training a model with a high-resource language translation pair first, then towards the end, resume the training with a low-resource pair. The existing research shows that this transfer learning method boosts the model's performance in translating the low-resource language pair comparing to a model just trained under the low-resource language pair alone \cite{anki}. In order to reproduce this approach, the team used public-available Python libraries such as BPE and Pandas for preprocessing the data and the Pytorch transformer as our model. We used data from manythings.org. At the end, our model showed an increase in BLEU score for the low-resource pair trained under transfer learning than without it. Also, we found that the parent should be trained for more than one epoch, but stopped before convergence for the best results.

## Content

**transfer_learning_nmt.ipynb** : Main notebook to reproduce our results. 

**preprocess.ipynb** : Notebook to preprocess the original datasets from http://www.manythings.org/anki/ and split the languages in different .txt files.

**baseline.py** : Script used to calculate the baseline for the child language pair

**parent.py** : Script to pretrain the parent language pair. It will save the model at each epoch and also saved the "best model" at convergence.

**child.py** : Script to load the parent model and fine-tune on the child language pair. 

## Usage

Simply clone this repo and run **transfer_learning_nmt.ipynb** on google colab. Don't forget to pick a GPU in the notebook setting.



## Data and References

Data from:
http://www.manythings.org/anki/


References:

How to make transformer from scratch
https://www.youtube.com/watch?v=U0s0f995w14&feature=youtu.be&ab_channel=AladdinPersson

How to train NMT with Pytorch transformer
https://www.youtube.com/watch?v=M6adRGJe5cQ&ab_channel=AladdinPersson
