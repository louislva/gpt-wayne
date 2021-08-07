from transformers import GPT2Tokenizer
from pickle import dump

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

full_file = open("./dataset/lil_wayne_genius_dataset.txt", "r").read()
tokens = tokenizer.encode(full_file)

dump(tokens, open("./dataset/lil_wayne_genius_dataset_tokenized.pkl", "wb"))
