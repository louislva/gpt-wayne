import jsonlines
from transformers import GPT2Tokenizer
import pickle
import random

MODEL_VERSION = "gpt2"
TEST_SPLIT = 0.1

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_VERSION)


def get_all_songs(source_path):
    reader = jsonlines.Reader(open(source_path, "r"))

    selected_songs = []

    for song in reader:
        selected_songs.append(song)

    random.shuffle(selected_songs)

    return selected_songs


def write_dataset(songs, output):
    dataset = ""

    for song in songs:
        processed_song_title = " ".join(
            [word.title() for word in song["song"].split("-")[:-1]]
        )

        dataset += (
            "===\nArtist & title: "
            + processed_song_title
            + "\n---\n\n"
            + song["lyrics"].strip()
            + "\n\n\n"
        )

    with open(output, "wb") as f:
        tokens = tokenizer.encode(dataset)
        pickle.dump(tokens, f)


if __name__ == "__main__":
    songs = get_all_songs("./dataset/lyrics.jl")

    songs_train = songs[int(len(songs) * TEST_SPLIT) :]
    songs_test = songs[: int(len(songs) * TEST_SPLIT)]

    write_dataset(songs_train, "./dataset/songs-train.pkl")
    write_dataset(songs_test, "./dataset/songs-test.pkl")
