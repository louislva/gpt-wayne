import jsonlines


reader = jsonlines.Reader(open('./dataset/lyrics.jl', 'r'))

selected_songs = []

for song in reader:
    if song["song"].startswith("Lil-wayne"):
        selected_songs.append(song)
    elif "Lil Wayne]" in song["lyrics"]:
        selected_songs.append(song)

dataset = ""

for song in selected_songs:
    processed_song_title = " ".join([word.title()
                                     for word in song["song"].split("-")[:-1]])

    dataset += "===\nArtist & title: " + processed_song_title + \
        '\n---\n\n' + song["lyrics"].strip() + "\n\n\n"


with open("./dataset/lil_wayne_genius_dataset.txt", "w") as f:
    f.write(dataset)
