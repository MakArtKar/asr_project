import os


links = {
    "librispeech vocab": "https://us.openslr.org/resources/11/librispeech-vocab.txt",
    "LM": "https://us.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz",
}


def main():
    os.makedirs("./data/datasets/librispeech", exist_ok=True)
    os.system(f"wget {links['librispeech vocab']} -P ./data/datasets/librispeech")

    os.makedirs("./data/decoders/", exist_ok=True)
    os.system(f"wget {'LM'} -P ./data/decoders")
    os.system("gzip -d ./data/decoders/3-gram.pruned.1e-7.arpa.gz")

    with open("./data/decoders/3-gram.pruned.1e-7.arpa", "r") as f_in, \
            open("./data/decoders/processed-3-gram.pruned.1e-7.arpa", "w") as f_out:
        for line in f_in:
            f_out.write(line.lower())


if __name__ == "__main__":
    main()
