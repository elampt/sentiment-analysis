#!/usr/bin/env python3
import http.server
import json
import numpy as np
import pandas as pd
from urllib.parse import urlparse, parse_qs

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import unicodedata
import re
from transformers import BertModel, BertTokenizer
import random


class ATIS(Dataset):
    def __init__(self, df, translit_prob=0, shuffle_prob=0, max_token_len=100):
        super(ATIS, self).__init__()

        self.df = df
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.max_token_len = max_token_len
        self.translit_prob = translit_prob
        self.shuffle_prob = shuffle_prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        query = self.df.iloc[index, 0]
        query = self.query_preprocessing(query)

        if random.random() < self.shuffle_prob:
            query_list = query.split()
            if len(query_list) < 10:
                random.shuffle(query_list)
                query = " ".join(query_list)

        tokens = self.tokenizer.tokenize(query)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        if len(tokens) < self.max_token_len:
            tokens = tokens + ["[PAD]" for i in range(self.max_token_len - len(tokens))]
        else:
            tokens = tokens[: self.max_token_len - 1] + ["[SEP]"]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        token_id_tensor = torch.tensor(token_ids)
        attention_mask_tensor = (token_id_tensor != 0).long()

        return token_id_tensor, attention_mask_tensor

    def unicodeToAscii(self, s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.lower().strip())
        s = re.sub(r"([.!?])", r"", s)
        s = re.sub(r"\s+", r" ", s).strip()
        return s

    def query_preprocessing(self, query_text):
        q = self.normalizeString(query_text)

        return q


class INTENT_CLASSIFIER(nn.Module):
    def __init__(self, freeze_bert=True):
        super(INTENT_CLASSIFIER, self).__init__()

        self.bert_layers = BertModel.from_pretrained(
            "bert-base-multilingual-cased", return_dict=False
        )
        self.linear1 = nn.Linear(768, 300)
        self.linear11 = nn.Linear(300, 8)
        self.linear2 = nn.Linear(8, 2)
        self.dropout = nn.Dropout(0.5)

        if freeze_bert:
            for param in self.bert_layers.parameters():
                param.requires_grad = False

    def forward(self, token_ids, atten_mask):
        """Both argument are of shape: batch_size, max_seq_len"""
        _, CLS = self.bert_layers(token_ids, attention_mask=atten_mask)
        logits = self.dropout(self.linear1(CLS))
        logits = self.dropout(self.linear11(logits))
        logits = self.linear2(logits)

        return logits


class RequestHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        CLASSES = ["Negative", "Positive"]
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        parsed_data = json.loads(post_data.decode())
        # corpus is a list of strings
        sentences = parsed_data["sentences"]
        sent_df = pd.DataFrame({"sentences": sentences})
        # creating instance of datset class
        test_set = ATIS(sent_df, max_token_len=120, translit_prob=0, shuffle_prob=0)
        t_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False)
        bert_model.eval()
        y_pred = []
        with torch.no_grad():
            for data in t_loader:
                tokens, masks = data
                tokens = tokens.to(device)
                masks = masks.to(device)
                outputs = bert_model(tokens, masks)
                _, predicted = torch.max(outputs.data, 1)
                y_pred += predicted.detach().cpu().numpy().tolist()
        y_pred = list(map(lambda x: CLASSES[x], y_pred))
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        # Send the response JSON
        self.wfile.write(json.dumps(y_pred).encode())


bert_model = torch.load("./models/best_model.pth", map_location=torch.device("cpu"))
device = "cuda" if torch.cuda.is_available() else "cpu"
if __name__ == "__main__":
    PORT = 42070
    server_address = ("", PORT)
    httpd = http.server.HTTPServer(server_address, RequestHandler)
    print("Server listening on port", PORT)
    httpd.serve_forever()
