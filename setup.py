import argparse
import os


links = {
    "librispeech vocab": "https://us.openslr.org/resources/11/librispeech-vocab.txt",
    "LM": "https://us.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz",
}


def setup_librispeech_vocab():
    os.makedirs("./data/datasets/librispeech", exist_ok=True)
    os.system("rm -f ./data/datasets/librispeech/librispeech-vocab.txt")
    os.system(f"wget {links['librispeech vocab']} -P ./data/datasets/librispeech")


def setup_lm():
    os.makedirs("./data/decoders/", exist_ok=True)
    os.system(f"wget {'LM'} -P ./data/decoders")
    os.system("gzip -d ./data/decoders/3-gram.pruned.1e-7.arpa.gz")

    with open("./data/decoders/3-gram.pruned.1e-7.arpa", "r") as f_in, \
            open("./data/decoders/processed-3-gram.pruned.1e-7.arpa", "w") as f_out:
        for line in f_in:
            f_out.write(line.lower())


def setup_common_voice(token):
    from datasets import load_dataset
    load_dataset('mozilla-foundation/common_voice_11_0', 'en', use_auth_token=token)


def main(args):
    if args.librispeech_vocab:
        setup_librispeech_vocab()
    if args.lm:
        setup_lm()
    if args.common_voice:
        setup_common_voice(args.common_voice)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--lm', action="store_true", default=False, help="Download lm")
    parser.add_argument('--librispeech_vocab', action="store_true", default=False, help="Download librispeech vocab")
    parser.add_argument('--common_voice', type=str, default="", help="Download Common Voice dataset")
    args = parser.parse_args()
    main(args)
