import torch
import torch.nn as nn
import torch.optim as optim
import spacy
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Dataset, Field, BucketIterator, TabularDataset
from torchtext.data.metrics import bleu_score
from sklearn.model_selection import train_test_split
import numpy as np
import sys
from tokenizers import Tokenizer
from tokenizers import CharBPETokenizer

def translate_sentence(model, sentence, german, english, device, max_length=50):
    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    
    if type(sentence) == str:
        tokens = [tok for tok in lt_tokenizer.encode(sentence).tokens]
    else:
        tokens = [tok for tok in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]

def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    

tokenizer = CharBPETokenizer()
tokenizer.train(["english_lt.txt", "lithuanian.txt" ])
# tokenizer.train([ "english_fr.txt", "english_lt.txt", "french.txt", "lithuanian.txt" ])

eng_tokenizer = CharBPETokenizer()
eng_tokenizer.train([ "english_lt.txt" ])
lt_tokenizer = CharBPETokenizer()
lt_tokenizer.train([ "lithuanian.txt" ])

english_txt = open('english_lt.txt', encoding='utf-8').read().split('\n')
lithuanian_txt = open('lithuanian.txt', encoding='utf-8').read().split('\n')

#Create dataframe
raw_data = {'English': [line for line in english_txt],
            'Lithuanian': [line for line in lithuanian_txt]}

df = pd.DataFrame(raw_data, columns=['English', 'Lithuanian'])

df = df.drop_duplicates("English")
print(len(df.index))
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
#   return [tok.text for tok in spacy_eng.tokenizer(text)]
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
num_epochs = 100
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
step_valid = 0

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

#early_stopping = EarlyStopping(patience=patience, verbose=True)



# You look just adorable tonight.
sentence = "Šįvakar tu atrodai tiesiog žavingai."
validLoss = [np.inf]

early_stop = 0
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
        

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
    model.eval()
    valid_losses = []

    for batch_idx, batch in enumerate(valid_iterator):
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


        loss = criterion(output, target)
        valid_losses.append(loss.item())

        writer.add_scalar("valid loss", loss, global_step=step_valid)
        step_valid += 1

    mean_valid_loss = sum(valid_losses) / len(valid_losses)
    mean_train_loss = sum(losses) / len(losses)

    scheduler.step(mean_valid_loss)
    print('train loss = ',mean_train_loss)
    print('valid loss = ',mean_valid_loss)
    print(validLoss[-1])
    if validLoss[-1]<mean_valid_loss:
      
      early_stop+=1
    else:
      checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
      save_checkpoint(checkpoint,'ro_eng.pth.tar')
      early_stop = 0
    if early_stop ==4:
      print('overfitting')
      break
    validLoss.append(mean_valid_loss)



# running on entire test data takes a while
load_checkpoint(torch.load("ro_eng.pth.tar"), model, optimizer)
score = bleu(test_data, model, lithuanian, english, device)
print(f"Bleu score {score * 100:.2f}")
