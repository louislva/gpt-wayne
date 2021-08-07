import pickle
import numpy as np
import random
import os
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from transformers import GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import GPT2Tokenizer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# By order of size:
# gpt-2
# gpt-2-medium
# gpt-2-large
# gpt-2-xl

MODEL_VERSION = "gpt2"

model = GPT2LMHeadModel.from_pretrained(MODEL_VERSION)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_VERSION)
model.to(DEVICE)


class SongDataset(Dataset):
    def __init__(self, data_src, sequence_length):
        self.sequence_length = sequence_length

        with open(data_src, "rb") as f:
            self.original_data = pickle.load(f)

        self.shuffle()

    def shuffle(self):
        # Shift the dataset every time we shuffle, so the sequences aren't always cut in the same place
        offset = random.randrange(0, self.sequence_length)
        self.data = self.original_data[offset:] + self.original_data[:offset]

        # Pad in the end
        self.data = self.data + (
            [tokenizer.eos_token_id]
            * (self.sequence_length - (len(self.data) % self.sequence_length))
        )

        # Cut into sequences with length = self.sequence_length
        self.data = np.array(self.data).reshape(-1, self.sequence_length)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]).long()


# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

BATCH_SIZE = 16

train_dataset = SongDataset("../data/dataset/songs-train.pkl", 120)
test_dataset = SongDataset("../data/dataset/songs-test.pkl", 120)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.0)


def evaluate(model):
    model.eval()
    with torch.no_grad():
        generation_output = model.generate(
            torch.tensor([[tokenizer.eos_token_id]]).to(DEVICE),
            return_dict_in_generate=True,
            output_scores=True,
        )
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        total_loss = 0

        for i, x in tqdm(
            enumerate(test_dataloader), total=test_dataset.__len__() // BATCH_SIZE
        ):
            x = x.to(DEVICE)
            y = model.forward(input_ids=x).logits

            loss = loss_fn(y.view((-1, 50257)), x.view((-1,)))
            total_loss += loss.item()

        return total_loss, tokenizer.decode(generation_output.sequences[0])


batches_per_epoch = train_dataset.__len__() // BATCH_SIZE
test_loss, test_generation = evaluate(model)
writer.add_scalar("Loss/test", test_loss, 0 * batches_per_epoch)
writer.add_text("Generation", test_generation, 0 * batches_per_epoch)

for epoch in range(100):
    print("EPOCH", epoch)
    model.train()
    train_dataset.shuffle()
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    for i, y in tqdm(enumerate(train_dataloader), total=batches_per_epoch):
        y = y.to(DEVICE)
        x = torch.cat(
            [
                torch.full((BATCH_SIZE, 1), tokenizer.eos_token_id).long().to(DEVICE),
                y[:, :-1],
            ],
            dim=1,
        )
        y_ = model.forward(input_ids=x).logits

        loss = loss_fn(y_.view((-1, 50257)), y.view((-1,)))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar("Loss/train", loss.item(), epoch * batches_per_epoch + i)

    test_loss, test_generation = evaluate(model)

    checkpoint_path = (
        "./checkpoints/" + writer.get_logdir()[5:] + "/epoch-" + str(epoch)
    )
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    model.save_pretrained(checkpoint_path)

    writer.add_scalar("Loss/test", test_loss, (epoch + 1) * batches_per_epoch)
    writer.add_text("Generation", test_generation, (epoch + 1) * batches_per_epoch)
