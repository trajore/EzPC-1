import numpy as np
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import struct
import sys

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


def tokenize(tokenizer, text_a, text_b=None):
    tokens_a = ["[CLS]"] + tokenizer.tokenize(text_a) + ["[SEP]"]
    tokens_b = (tokenizer.tokenize(text_b) + ["[SEP]"]) if text_b else []

    tokens = tokens_a + tokens_b
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(tokens_a) + [1] * len(tokens_b)

    return tokens, input_ids, segment_ids


f = None


def set_file(filename):
    global f
    f = open(filename, "wb")


def close_file():
    global f
    f.close()
    f = None


def dumpvec(x):
    global f
    if f == None:
        return
    assert len(x.shape) == 1
    CI = x.shape[0]
    for ci in range(CI):
        f.write(struct.pack("f", x[ci].item()))


def dumpmat(w):
    global f
    if f == None:
        return
    assert len(w.shape) == 2
    CI = w.shape[0]
    CO = w.shape[1]
    for ci in range(CI):
        for co in range(CO):
            f.write(struct.pack("f", w[ci][co].item()))


def main():
    # load dataset
    dataset = load_dataset("sst2", split="validation")
    sd = torch.load("model_sst2.pth", map_location=torch.device("cpu"))

    wse = sd["bert.embeddings.token_type_embeddings.weight"]
    wte = sd["bert.embeddings.word_embeddings.weight"]
    wpe = sd["bert.embeddings.position_embeddings.weight"]

    max_len = 512
    n_head = 12

    og = open("datasets/sst2/labels.txt", "w")
    for label in dataset["label"]:
        og.write(str(label) + "\n")
    og.close()
    i = 0
    for text, label in tqdm(zip(dataset["sentence"], dataset["label"])):
        _, input_ids, segment_ids = tokenize(tokenizer, text)
        input_ids, segment_ids = input_ids[:max_len], segment_ids[:max_len]
        x = wte[input_ids] + wpe[range(len(input_ids))] + wse[segment_ids]
        # print(x.shape)
        set_file("datasets/sst2/" + str(i) + ".dat")
        dumpmat(x)
        close_file()
        # print(label)
        i += 1


if __name__ == "__main__":
    import fire

    fire.Fire(main)
