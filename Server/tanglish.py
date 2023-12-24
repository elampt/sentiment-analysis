#!/usr/bin/env python3
from flask import Flask, jsonify, request, Response
from nltk import word_tokenize, sent_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from ai4bharat.transliteration import XlitEngine
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model
import re
import string
import pickle
from numpy import argmax
import requests

transliteration_model = XlitEngine("ta", beam_width=20, rescore=True)
with open("./models/pos_crf_model.pkl", "rb") as f:
    pos_model = pickle.load(f)
with open("./models/ner_crf_model.pkl", "rb") as f:
    ner_model = pickle.load(f)
with open("./models/lstm_tokenizer.pkl", "rb") as f:
    lstm_tokenizer = pickle.load(f)
lstm_model = load_model("./models/llm_lstm_model.keras/")

app = Flask(__name__)


def conv_uchar_to_space(str_):
    return "".join([i if ord(i) < 128 else " " for i in str_])


def remove_punctuation(str_: str):
    return str_.translate(str.maketrans("", "", string.punctuation))


def preprocess(str_, do_word_tokenize=True):
    str_ = sent_tokenize(str_)
    for ind, val in enumerate(str_):
        tmp = conv_uchar_to_space(val)
        tmp = tmp.lower()
        tmp = re.sub(r"\d+", "", tmp)
        tmp = remove_punctuation(tmp)
        if do_word_tokenize:
            tmp = word_tokenize(tmp)
        str_[ind] = tmp
    return str_


def ngram(str_):
    str_ = preprocess(str_)
    return padded_everygram_pipeline(2, str_)


def transliterate(str_):
    return transliteration_model.translit_sentence(str_, lang_code="ta")


def extract_crf_features(sentence, index):
    return {
        "word": sentence[index],
        "is_first": index == 0,
        "is_last": index == len(sentence) - 1,
        "prefix-1": sentence[index][0],
        "prefix-2": sentence[index][:2],
        "prefix-3": sentence[index][:3],
        "prefix-3": sentence[index][:4],
        "suffix-1": sentence[index][-1],
        "suffix-2": sentence[index][-2:],
        "suffix-3": sentence[index][-3:],
        "suffix-3": sentence[index][-4:],
        "prev_word": "" if index == 0 else sentence[index - 1],
        "next_word": "" if index < len(sentence) else sentence[index + 1],
        "has_hyphen": "-" in sentence[index],
        "is_numeric": sentence[index].isdigit(),
    }


def transform_to_crf_dataset(corpus: str):
    sentences = sent_tokenize(corpus)
    sentences = [word_tokenize(sentence) for sentence in sentences]
    X = []
    for sentence in sentences:
        tmp = []
        for i in range(len(sentence)):
            tmp.append(extract_crf_features(sentence, i))
        X.append(tmp)
    return X


def pos(corpus: str):
    X = transform_to_crf_dataset(corpus)
    return pos_model.predict(X)


def extract_ner_features(sentence, index):
    word = sentence[index]
    return {
        "word": word,
        "is_first": index == 0,
        "is_last": index == len(sentence) - 1,
        "prefix-1": "" if not word else word[0],
        "prefix-2": word[:2],
        "prefix-3": word[:3],
        "prefix-4": word[:4],
        "suffix-1": "" if not word else word[-1],
        "suffix-2": word[-2:],
        "suffix-3": word[-3:],
        "suffix-4": word[-4:],
        "prev_word": "" if index == 0 else sentence[index - 1][0],
        "next_word": "" if index == len(sentence) - 1 else sentence[index + 1][0],
        "has_hyphen": "-" in word,
        "is_numeric": word.isdigit(),
    }


def transform_to_ner_dataset(corpus: str):
    sentences = sent_tokenize(corpus)
    sentences = [word_tokenize(sentence) for sentence in sentences]
    X = []
    for sentence in sentences:
        tmp = []
        for i in range(len(sentence)):
            tmp.append(extract_ner_features(sentence, i))
        X.append(tmp)
    return X


def ner(corpus: str):
    X = transform_to_ner_dataset(corpus)
    return ner_model.predict(X)


def transform_to_lstm_dataset(corpus: list):
    corpus = lstm_tokenizer.texts_to_sequences(corpus)
    corpus = pad_sequences(corpus)
    return corpus


def sa_lstm(corpus: list):
    CLASSES = ["Mixed_feelings", "Negative", "Positive", "unknown_state"]
    X = transform_to_lstm_dataset(corpus)
    y_pred = lstm_model.predict(X)
    return list(map(lambda x: CLASSES[argmax(x)], y_pred))


def sa_bert(corpus: list):
    API_URL = "http://127.0.0.1:42070/"
    request_body = {"sentences": corpus}
    res = requests.post(API_URL, json=request_body)
    return res.json()


@app.route("/preprocess/", methods=["POST"])
def handle_preprocess():
    corpus = request.get_json()
    corpus = corpus["text"]
    corpus = preprocess(corpus)
    return jsonify(corpus)


@app.route("/ngram/", methods=["POST"])
def handle_ngram():
    req = request.get_json()
    corpus = req["text"]
    train, vocab = ngram(corpus)
    ret = {}
    ret["train"] = [list(x) for x in train]
    ret["vocab"] = [x for x in vocab]
    return jsonify(ret)


@app.route("/transliterate/", methods=["POST"])
def handle_transliterate():
    corpus = request.get_json()["text"]
    return jsonify(transliterate(corpus))


@app.route("/pos/", methods=["POST"])
def handle_pos():
    corpus = request.get_json()["text"]
    corpus = transliterate(corpus)
    return jsonify(pos(corpus))


@app.route("/ner/", methods=["POST"])
def handle_ner():
    corpus = request.get_json()["text"]
    corpus = transliterate(corpus)
    return jsonify(ner(corpus))


@app.route("/saLSTM/", methods=["POST"])
def handle_sa_lstm():
    corpus = request.get_json()["text"]
    corpus = preprocess(corpus, do_word_tokenize=False)
    return jsonify(sa_lstm(corpus))


@app.route("/saBERT/", methods=["POST"])
def handle_sa_bert():
    corpus = request.get_json()["text"]
    corpus = preprocess(corpus, do_word_tokenize=False)
    return jsonify(sa_bert(corpus))
