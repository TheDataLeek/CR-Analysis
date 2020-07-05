#!/usr/bin/env python3.8

# # Analyzing Critical Role
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


curdir = pathlib.Path()
output_file = curdir / "cr.json"
processed_file = curdir / "processed.json"


SEQUENCE_SIZE = 100


def main():
    author_map = generate_author_map()
    json_formatted = parse_html(author_map)

    df = pd.read_json("cr.json")
    df.head()

    if not processed_file.exists():
        print('processing data')
        main_cast_probs = (
            df.groupby("author").text.count().sort_values(ascending=False)[:8]
        )
        main_cast = list(main_cast_probs.index)
        main_cast_probs = main_cast_probs.values / sum(main_cast_probs.values)

        main_cast_json = [obj for obj in json_formatted if obj["author"] in main_cast]

        new_text, word_index, index_word = preprocess(main_cast_json)

        processed_file.write_text(
            json.dumps(
                {"text": new_text, "word_index": word_index, "index_word": index_word,}
            )
        )
    else:
        print('Loading previously processed data')
        obj = json.loads(processed_file.read_text())
        new_text, word_index, index_word = obj['text'], obj['word_index'], obj['index_word']

    pp(new_text[0])

    print(len(new_text))


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
                "raw": f"{o['author']}> {strip_chars(o['text'])}"
            }
            for o in parsed
        ]
        formatted.sort(key=lambda o: (o["season"], o["episode"]))
        output_file.write_text(json.dumps(formatted))

    return formatted


def tokenize_sentence(message: str):
    message = message.lower()
    message = re.subn(r"\([a-z]+\)", "", message)[0]
    chars_to_remove = '"#$%()*+-<=@[\\]^_`{|}~/'
    message = "".join(c for c in message if c not in chars_to_remove)
    message = re.subn(r"(&|,|\.|\?|!)", r" \1 ", message)[0]
    message = re.subn(r"  ", " ", message)[0]
    words = [w for w in re.split(r"\s", message) if w != ""]
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

    for message in text:
        sequence = message["sequence"]
        if len(sequence) < SEQUENCE_SIZE:
            continue

        for i in range(SEQUENCE_SIZE, len(sequence)):
            extract = sequence[i - SEQUENCE_SIZE : i + 1]

            features.append(extract[:-1])  # train on first 4
            labels.append(extract[-1])  # label is last

    features = np.array(features)

    # one-hot encode (switch to binary representation) for words
    num_words = len(word_index) + 1
    label_array = np.zeros((len(features), num_words), dtype=np.int8)
    for i, label in enumerate(labels):
        label_array[i, label] = 1

    print(f"Feature Dimensions: {features.shape}")
    print(f"Label Dimensions: {label_array.shape}")

    (
        feature_train,
        feature_test,
        label_train,
        label_test,
    ) = model_selection.train_test_split(
        features, label_array, test_size=0.1, shuffle=True
    )

    return feature_train, feature_test, label_train, label_test


def build_model(
    feature_train, feature_test, label_train, label_test, word_index,
):
    num_words = len(word_index) + 1

    dim = 25

    # set up model
    model = Sequential()  # Build model one layer at a time
    weights = None
    model.add(
        Embedding(  # maps each input word to 100-dim vector
            input_dim=num_words,  # how many words can input
            input_length=TIMESTEP,  # timestep length
            output_dim=dim,  # output vector
            weights=weights,
            trainable=True,  # update embeddings
        )
    )
    model.add(
        Bidirectional(
            LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1),
            input_shape=(TIMESTEP, dim),
        )
    )
    model.add(
        Bidirectional(
            LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)
        )
    )
    model.add(
        Bidirectional(
            LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)
        )
    )
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.25))  # input is rate that things are zeroed
    model.add(Dense(num_words, activation="softmax"))
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
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


def cleanup(input_string: str) -> str:
    output_string = re.subn(r" ([,\.\!\?])", r"\1", input_string)[0]
    output_string = re.subn(r" i([ ,\.\!\?'])", r" I\1", output_string)[0]

    to_upper = lambda match: match.group(1).upper()
    output_string = re.subn(r"^(\w)", to_upper, output_string)[0]
    to_upper = lambda match: f"{match.group(1)}{match.group(2).upper()}"
    output_string = re.subn(r"([.\!\?] )(\w)", to_upper, output_string)[0]

    if output_string[-1] not in ",.!?":
        output_string += random.choice(",.!?")

    return output_string


if __name__ == "__main__":
    sys.exit(main())
