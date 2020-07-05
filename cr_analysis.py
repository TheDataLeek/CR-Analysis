#!/usr/bin/env python3.8

# # Analyzing Critical Role
import argparse
import sys
import numpy as np
import pandas as pd
import pathlib
from bs4 import BeautifulSoup
import tqdm
import json
import matplotlib.pyplot as plt
from sklearn import preprocessing as skl_pre
from sklearn import model_selection
import re
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pprint import pprint as pp
import gc


curdir = pathlib.Path()
output_file = curdir / "cr.json"
processed_file = curdir / "processed.json"
model_dir = curdir / 'models'
script_file = curdir / 'output_script.txt'


SEQUENCE_SIZE = 20

USE_EP = lambda i: 70 <= i <= 80


def main():
    args = get_args()

    author_map = generate_author_map()
    json_formatted = parse_html(author_map)

    json_formatted = [
        obj for obj in json_formatted if USE_EP(obj["episode"]) and obj["season"] == 2
    ]

    json_formatted = extract_rp(json_formatted)

    df = pd.read_json("cr.json")
    df.head()

    if args.process or not processed_file.exists():
        print("processing data")
        main_cast_probs = (
            df.groupby("author").text.count().sort_values(ascending=False)[:8]
        )
        main_cast = list(main_cast_probs.index)
        main_cast_probs = main_cast_probs.values / sum(main_cast_probs.values)

        main_cast_json = [obj for obj in json_formatted if obj["author"] in main_cast]

        new_text, word_index, index_word = preprocess(main_cast_json)

        processed_file.write_text(
            json.dumps(
                {"text": new_text, "word_index": word_index, "index_word": index_word}
            )
        )
    else:
        print("Loading previously processed data")
        obj = json.loads(processed_file.read_text())
        new_text, word_index, index_word = (
            obj["text"],
            obj["word_index"],
            obj["index_word"],
        )

    # how_long_is_avg_sentence = get_avg_sentence_length(new_text)

    index_word = {int(k): v for k, v in index_word.items()}

    pp(new_text[0])

    print(f"{len(word_index)=}")

    print(f"{len(new_text)=}")

    if args.train:
        feature_train, feature_test, label_train, label_test = generate_features_and_labels(
            new_text, word_index
        )

        build_model(
            feature_train, feature_test, label_train, label_test, word_index,
        )

    model = model_dir / "model.h5"
    if not model.exists():
        raise FileNotFoundError
    model = load_model(model)

    seed = 'matt>'

    null_input = np.ones(SEQUENCE_SIZE) * word_index["."]
    seq = []
    for word in tokenize_sentence(seed):
        if word in word_index:
            seq.append(word_index[word])

    seq = seq[-SEQUENCE_SIZE:]
    null_input[-len(seq):] = seq
    start = np.array([null_input], dtype=np.float64)

    response = []
    for i in range(1_000):
        preds = model.predict(start)[0].astype(np.float64)
        preds = preds / sum(preds)  # normalize
        probas = np.random.multinomial(1, preds, 1)[0]

        next_idx = np.argmax(probas)
        response.append(next_idx)

        start = np.array([[*start[0], next_idx][-SEQUENCE_SIZE:]], dtype=np.float64)

    script_file.write_text(cleanup(seed, " ".join(index_word[s] for s in response)))


def extract_rp(json_formatted):
    cur_ep = None
    seen_theme_yet = False
    music_note = '♪'

    filtered = []
    for message in json_formatted:
        if message['ep'] != cur_ep:
            seen_theme_yet = False

        cur_ep = message['ep']

        if music_note in message['text'] and 'roll the dice' in message['text'].lower():
            seen_theme_yet = True
        elif seen_theme_yet:
            filtered.append(message)

    return filtered


def get_avg_sentence_length(json_formatted):
    num_words = 0
    num_sentences = 0
    current_sentence_word_count = 0
    just_text = ' '.join(t['tokenized'] for t in json_formatted).split(' ')
    data = []
    for word in just_text:
        if word in '.?!>':
            data.append(current_sentence_word_count)
            current_sentence_word_count = 0
            num_sentences += 1
        else:
            current_sentence_word_count += 1
            num_words += 1

    print(f"{num_words / num_sentences=}")

    hist, bins = np.histogram(data, bins=range(0, (5 * 15) + 1, 5), density=True)
    plt.figure(figsize=(12, 6))
    plt.bar(bins[1:] - 2.5, hist)
    plt.savefig('word_dist.png')



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False, help='Retrain model? (False)')
    parser.add_argument('--process', action='store_true', default=False, help='Reprocess data? (False)')
    args = parser.parse_args()
    return args


def generate_markov_probs(main_cast, json_formatted):
    nodes = main_cast
    node_lookup = {nodes[i]: i for i in range(len(nodes))}
    chain = np.zeros((len(nodes), len(nodes)), dtype=np.float64)

    previous = None
    for phrase in json_formatted:
        author = phrase["author"]
        if author not in nodes:
            continue
        if previous is None:
            previous = author
            continue

        chain[node_lookup[previous], node_lookup[author]] += 1
        previous = author
    chain = skl_pre.normalize(chain, norm="l1")

    # In[247]:

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(chain, cmap="viridis")
    plt.colorbar(im)
    ax.set_xticklabels(["a"] + main_cast)
    ax.set_yticklabels(["a"] + main_cast)
    ax.set_title('Probabilities of "Who speaks next"')
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    plt.show()


def strip_chars(s):
    return s.replace("→", "")


def generate_author_map():
    author_map = {}
    for row in [
        [name.strip() for name in line.split(",")]
        for line in (curdir / "author_map.txt").read_text().split("\n")
    ]:
        actual_name = row[0]
        author_map[actual_name] = actual_name
        if len(row) > 1:
            for other in row[1:]:
                author_map[other] = actual_name
    return author_map


def clean_author(s):
    return s.strip().replace("# ", "").lower().capitalize()


def parse_html(author_map):
    if output_file.exists():
        formatted = json.loads(output_file.read_text())
    else:
        parsed = []
        files = list((curdir / "html").glob("cr2*.html"))
        for file in tqdm.tqdm(files):
            soup = BeautifulSoup(file.read_text())
            text = soup.get_text()
            lines = text.split("\n")
            parsed_lines = []
            author = None
            curline = ""
            for line in lines:
                author_line = line.startswith("# ")

                if not line:
                    continue
                if author is None and not author_line:
                    continue
                if author_line and not curline:
                    author = line
                    continue
                if not author_line:
                    curline += line
                    continue
                if author_line and curline:
                    if "," in author:
                        for a in author.split(","):
                            parsed_lines.append(
                                {
                                    "author": author_map[clean_author(a)],
                                    "text": curline,
                                    "file": str(file.name),
                                    "ep": str(file.stem),
                                }
                            )
                    else:
                        parsed_lines.append(
                            {
                                "author": author_map[clean_author(author)],
                                "text": curline,
                                "file": str(file.name),
                                "ep": str(file.stem),
                            }
                        )
                    author = line
                    curline = ""
                    continue
            parsed += parsed_lines
        formatted = [
            {
                "author": o["author"],
                "text": strip_chars(o["text"]),
                "file": o["file"],
                "ep": o["ep"],
                "season": int(o["ep"].split("-")[0][2:]),
                "episode": int(o["ep"].split("-")[1]),
                "raw": f"{o['author']}> {strip_chars(o['text'])}",
            }
            for o in parsed
        ]
        formatted.sort(key=lambda o: (o["season"], o["episode"]))
        output_file.write_text(json.dumps(formatted))

    return formatted


def tokenize_sentence(message: str):
    message = message.lower()
    message = re.subn(r"\([a-z]+ [a-z]+\)", "", message)[0]  # remove things like (laughter)
    chars_to_remove = '"#$%*+-<=@[\\]^_`{|}~/♪'  # remove all unneeded chars
    message = "".join(c for c in message if c not in chars_to_remove)
    message = re.subn(r"(&|,|\.|\?|!)", r" \1 ", message)[0]  # surround important punctuation with spaces
    message = re.subn(r"  ", " ", message)[0]  # clean up double spaces
    words = [w for w in re.split(r"\s", message) if w != ""]  # remove empty tokens now
    return words


def preprocess(text):
    just_content = [t["raw"].lower() for t in text]

    index = 1
    word_index = {}
    sequences = []
    for message in just_content:
        words = tokenize_sentence(message)
        for word in words:
            if word not in word_index:
                word_index[word] = index
                index += 1
        sequences.append([word_index[word] for word in words])

    index_word = {v: k for k, v in word_index.items()}

    for i, seq in enumerate(sequences):
        text[i]["sequence"] = seq
        text[i]["tokenized"] = " ".join(index_word[s] for s in seq)

    return text, word_index, index_word


def generate_features_and_labels(text, word_index):
    features = []
    labels = []

    full_sequence = [item for message in text for item in message["sequence"]]
    print(f"{len(full_sequence)=}")

    for i in range(SEQUENCE_SIZE, len(full_sequence)):
        extract = full_sequence[i - SEQUENCE_SIZE : i + 1]

        features.append(extract[:-1])  # train on first N
        labels.append(extract[-1])  # label is last

    features = np.array(features, dtype=np.uint16)

    # # one-hot encode (switch to binary representation) for words
    # num_words = len(word_index) + 1
    # label_array = np.zeros((len(features), num_words), dtype=np.bool_)
    # for i, label in enumerate(labels):
    #     label_array[i, label] = True

    label_array = np.array(labels, dtype=np.uint16)

    print(f"{features.shape=}")
    print(f"{label_array.shape=}")

    (
        feature_train,
        feature_test,
        label_train,
        label_test,
    ) = model_selection.train_test_split(
        features, label_array, test_size=0.1, shuffle=True
    )

    # lets not leave these hanging huh?
    gc.enable()
    del features, label_array
    gc.collect()

    return feature_train, feature_test, label_train, label_test


def build_model(
    feature_train, feature_test, label_train, label_test, word_index,
):
    num_words = len(word_index) + 1

    dim = 100

    # set up model
    model = Sequential()  # Build model one layer at a time
    weights = None
    model.add(
        Embedding(  # maps each input word to 100-dim vector
            input_dim=num_words,  # how many words can input
            input_length=SEQUENCE_SIZE,  # timestep length
            output_dim=dim,  # output vector
            weights=weights,
            trainable=True,  # update embeddings
        )
    )
    model.add(
        Bidirectional(
            LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1),
            input_shape=(SEQUENCE_SIZE, dim),
        )
    )
    # model.add(
    #     Bidirectional(
    #         LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
    #     )
    # )
    model.add(
        Bidirectional(
            LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)
        )
    )
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.1))  # input is rate that things are zeroed
    model.add(Dense(num_words, activation="softmax"))
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    model.summary()

    # train
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5),
        ModelCheckpoint(
            "./models/model.h5", save_best_only=True, save_weights_only=False
        ),
    ]

    history = model.fit(
        feature_train,
        label_train,
        batch_size=2048,
        epochs=150,
        callbacks=callbacks,
        validation_data=(feature_test, label_test),
    )


def cleanup(seed, input_string: str) -> str:
    output_string = re.subn(r"([a-z]+>)", r"\n\1", input_string)[0]
    output_string = f"{seed} {output_string}"

    output_string = re.subn(r" ([,\.\!\?])", r"\1", output_string)[0]
    output_string = re.subn(r" i([ ,\.\!\?'])", r" I\1", output_string)[0]

    to_upper = lambda match: match.group(1).upper()
    output_string = re.subn(r"^(\w)", to_upper, output_string)[0]
    to_upper = lambda match: f"{match.group(1)}{match.group(2).upper()}"
    output_string = re.subn(r"([.\!\?] )(\w)", to_upper, output_string)[0]

    return output_string


if __name__ == "__main__":
    sys.exit(main())
