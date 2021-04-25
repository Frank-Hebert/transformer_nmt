Yassine Kassis, Francois Hébert, Yu Chen
IFT 6010 Project on transfer learning


In this file we will explain the steps that we should follow 
to have the results obtained for the English-Lithuanian example.

First of all, we need to do the preprocessing for the dataset that we downloaded from 
manythings.org. And by calling preprocess.ipynb on the dataset.txt file it'll result two text
files for each of the languages pairs.

Now, we can run notebook.ipynb that will install all the needed librarys and run the code to
give us the results that we're looking for.
It'll start by runnig baseline.py, this one will give us the baseline results for our model.
then parent.py will give us the parent model that's trained until convergence but it'll save 
all the parent's models after each epoch and finally the child.py where we apply the transfer
learning and have the final results that we are looking for.



Data from:
http://www.manythings.org/anki/


References:
How to make transformer from scratch
https://www.youtube.com/watch?v=U0s0f995w14&feature=youtu.be&ab_channel=AladdinPersson

How to train NMT with Pytorch transformer
https://www.youtube.com/watch?v=M6adRGJe5cQ&ab_channel=AladdinPersson