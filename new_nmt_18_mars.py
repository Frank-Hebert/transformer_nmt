import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import pandas as pd
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Dataset, Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from tokenizers import Tokenizer
from tokenizers import CharBPETokenizer
# from tokenizers.models import BPE
# from tokenizers.pre_tokenizers import Whitespace
# from tokenizers.trainers import BpeTrainer

# tokenizer = Tokenizer(BPE())
# tokenizer.pre_tokenizer = Whitespace()

# trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
# tokenizer.train(files=["wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"], trainer=trainer)
# output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
# print(output.tokens)

eng_tokenizer = CharBPETokenizer()

# Then train it!
eng_tokenizer.train([ "english.txt" ])

encoded = eng_tokenizer.encode("I can feel the magic, can you?")
print(encoded.tokens)
lt_tokenizer = CharBPETokenizer()

# Then train it!
lt_tokenizer.train([ "lithuanian.txt" ])
print(lt_tokenizer)

# sys.exit(0)


spacy_eng = spacy.load("en_core_web_sm")
# spacy_fr = spacy.load("fr_core_news_sm")
spacy_lit = spacy.load("lt_core_news_sm")

english_txt = open('english.txt', encoding='utf-8').read().split('\n')
lithuanian_txt = open('lithuanian.txt', encoding='utf-8').read().split('\n')

#Create dataframe
raw_data = {'English': [line for line in english_txt],
            'Lithuanian': [line for line in lithuanian_txt]}

df = pd.DataFrame(raw_data, columns=['English', 'Lithuanian'])

#Split the data
test_size = 0.05
valid_size = 0.15 / (1-test_size) 

train_temp, test = train_test_split(df, test_size=test_size)
train, valid = train_test_split(train_temp, test_size=valid_size)

#Save to .json files
train.to_json('train.json', orient='records', lines=True)
test.to_json('test.json', orient='records', lines=True)
valid.to_json('valid.json', orient='records', lines=True)

#Tokenizers
def tokenize_eng(text):
    # return [tok.text for tok in spacy_eng.tokenizer(text)]
    return [tok for tok in eng_tokenizer.encode(text).tokens]

def tokenize_lit(text):
#   return [tok.text for tok in spacy_lit.tokenizer(text)]
    return [tok for tok in lt_tokenizer.encode(text).tokens]


#Create Fields
english = Field(sequential=True, use_vocab=True, tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
lithuanian = Field(sequential=True, use_vocab=True, tokenize=tokenize_lit, lower=True, init_token="<sos>", eos_token="<eos>")


fields = {'Lithuanian' : ('src', lithuanian), 'English': ('trg', english)}

#Tabular Dataset
train_data, valid_data, test_data = TabularDataset.splits(
    path='',
    train='train.json',
    validation='valid.json',
    test='test.json',
    format='json',
    fields=fields)


#Create Vocab
english.build_vocab(train_data, max_size=10000, min_freq=2)
lithuanian.build_vocab(train_data, max_size=10000, min_freq=2)

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
    ):
        super(Transformer, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = False
save_model = True

# Training hyperparameters
num_epochs = 20
learning_rate = 3e-4
batch_size = 32

# Model hyperparameters
src_vocab_size = len(lithuanian.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 256
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "Mano vardas Yassine."


for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, lithuanian, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target[:-1, :])

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)

# running on entire test data takes a while
# score = bleu(test_data[1:100], model, lithuanian, english, device)
# print(f"Bleu score {score * 100:.2f}")