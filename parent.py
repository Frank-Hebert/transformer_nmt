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
import matplotlib.pyplot as plt

# function to translate a sentence from source to target using the latest trained model
def translate_sentence(model, sentence, german, english, device, max_length=50):
    # Create tokens using spacy and everything in lower case (which is what our vocab is)

    if type(sentence) == str:
        tokens = [tok for tok in tokenizer.encode(sentence).tokens]
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

# Function to compute the blue score of the child pair using the test set and latest model
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

# function to save model parameters
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# function to load pre-saved model parameters
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

# Common Tokenizers training
tokenizer = CharBPETokenizer()
tokenizer.train(["english_fr.txt", "english_lt.txt", "french.txt", "lithuanian.txt"])

# Data loading
english_lt = open("english_lt.txt", encoding="utf-8").read().split("\n")
lithuanian = open("lithuanian.txt", encoding="utf-8").read().split("\n")
english_fr = open("english_fr.txt", encoding="utf-8").read().split("\n")
french = open("french.txt", encoding="utf-8").read().split("\n")

# Create dataframe
raw_data_child = {
    "English": [line for line in english_lt],
    "Lithuanian": [line for line in lithuanian],
}
raw_data_parent = {
    "English": [line for line in english_fr],
    "French": [line for line in french],
}

df_child = pd.DataFrame(raw_data_child, columns=["English", "Lithuanian"])
df_parent = pd.DataFrame(raw_data_parent, columns=["English", "French"])


# Split the data
test_size = 0.05
valid_size = 0.15 / (1 - test_size)

train_temp, test_parent = train_test_split(df_parent, test_size=test_size)
train_parent, valid_parent = train_test_split(train_temp, test_size=valid_size)


# Save to .json files
train_parent.to_json("train_parent.json", orient="records", lines=True)
test_parent.to_json("test_parent.json", orient="records", lines=True)
valid_parent.to_json("valid_parent.json", orient="records", lines=True)

train_temp, test_child = train_test_split(df_child, test_size=test_size)
train_child, valid_child = train_test_split(train_temp, test_size=valid_size)


# Save to .json files
train_child.to_json("train_child.json", orient="records", lines=True)
test_child.to_json("test_child.json", orient="records", lines=True)
valid_child.to_json("valid_child.json", orient="records", lines=True)

# Tokenizers
def tokenize_global(text):
    return [tok for tok in tokenizer.encode(text).tokens]


# Create a common Field
common = Field(
    sequential=True,
    use_vocab=True,
    tokenize=tokenize_global,
    lower=True,
    init_token="<sos>",
    eos_token="<eos>",
)

fields_child = {"Lithuanian": ("src", common), "English": ("trg", common)}
fields_parent = {"French": ("src", common), "English": ("trg", common)}

# Convert parent into Tabular Dataset
train_data_parent, valid_data_parent, test_data_parent = TabularDataset.splits(
    path="",
    train="train_parent.json",
    validation="valid_parent.json",
    test="test_parent.json",
    format="json",
    fields=fields_parent,
)

# Convert child into Tabular Dataset
train_data_child, valid_data_child, test_data_child = TabularDataset.splits(
    path="",
    train="train_child.json",
    validation="valid_child.json",
    test="test_child.json",
    format="json",
    fields=fields_child,
)

# Create a common vocab
common.build_vocab(train_data_child, train_data_parent, max_size=10000, min_freq=2)

# Prebuild transformer class from pytorch
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
src_vocab_size = len(common.vocab)
trg_vocab_size = len(common.vocab)
embedding_size = 256
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = common.vocab.stoi["<pad>"]

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0
step_valid = 0

# build into torchtext iterators
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data_parent, valid_data_parent, test_data_parent),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)

# Initiate transformer model
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

pad_idx = common.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


# Sentence in Luthanian to translate, the true meaning is: You look just adorable tonight.
sentence = "Il fait super beau aujourd'hui et c'était la fête de Frank hier."
validLoss = [np.inf]

early_stop = 0
# iterate through each epoch
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, common, common, device, max_length=50
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

# compute losses
    mean_valid_loss = sum(valid_losses) / len(valid_losses)
    mean_train_loss = sum(losses) / len(losses)

    scheduler.step(mean_valid_loss)
    print("train loss = ", mean_train_loss)
    print("valid loss = ", mean_valid_loss)
    print(validLoss[-1])
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    save_checkpoint(checkpoint, f"./model/fr_en_lt_epoch_{epoch}.pth.tar")
    # If valid loss is smaller than the mean loss
    if min(validLoss) < mean_valid_loss:

        early_stop += 1
    else:
        save_checkpoint(checkpoint, f"./model/fr_en_lt_best.pth.tar")
        early_stop = 0
    # Check if converges for 4 consecutives epochs
    if early_stop == 4:
        print("overfitting")
        print(valid_losses)
        plt.plot(valid_losses)
        plt.show()
        break
    validLoss.append(mean_valid_loss)

print(f"validloss : {validLoss}")
# running on entire test data takes a while
load_checkpoint(torch.load("./model/fr_en_lt_best.pth.tar"), model, optimizer)
# Compute BLEU score
score = bleu(test_data_parent, model, common, common, device)
print(f"Bleu score {score * 100:.2f}")