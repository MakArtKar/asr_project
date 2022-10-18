import os


def main():
    os.makedirs("./data/datasets/librispeech", exist_ok=True)
    os.system("wget https://us.openslr.org/resources/11/librispeech-vocab.txt -P ./data/datasets/librispeech")

    os.makedirs("./data/decoders/", exist_ok=True)
    os.system("wget https://us.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz -P ./data/decoders")
    os.system("gzip -d ./data/decoders/3-gram.pruned.1e-7.arpa.gz")

    with open("./data/decoders/3-gram.pruned.1e-7.arpa", "r") as f_in, \
            open("./data/decoders/processed-3-gram.pruned.1e-7.arpa", "w") as f_out:
        for line in f_in:
            f_out.write(line.lower())


if __name__ == "__main__":
    main()
