import glob
import os


def imdb_dataset(directory="data/aclImdb/",
                 train=False,
                 test=False,
                 train_directory="train",
                 test_directory="test",
                 need_vocabs=False):
    sentiments = ['pos', 'neg']
    x = []
    splits = [
        dir_ for (requested, dir_) in [(train, train_directory), (test, test_directory)]
        if requested
    ]
    for split_directory in splits:
        full_path = os.path.join(directory, split_directory)
        examples = []
        for sentiment in sentiments:
            for filename in glob.iglob(os.path.join(full_path, sentiment, "*.txt")):
                with open(filename, 'r', encoding="utf-8") as f:
                    textnew = f.readline()
                examples.append({
                    'text': textnew,
                    'sentiment': '0' if sentiment == 'neg' else '1',
                })
        x.append(examples)
    vocab = []
    if need_vocabs:
        with open(os.path.join(directory, "imdb.vocab")) as f:
            vocab = [lines.strip() for lines in f.readlines()]
    if len(x) == 1:
        return x[0] if not need_vocabs else x[0], vocab
    else:
        return tuple(x) if not need_vocabs else tuple(x), vocab


def amazon_or_yelp_dataset(
        directory="data/amazon/",
        train=False,
        dev=False,
        test=False,
        train_directory="train",
        dev_directory="dev",
        test_directory="test"
):
    sentiments = ['0', '1']
    x = []
    splits = [
        dir_ for (requested, dir_) in [(train, train_directory), (test, test_directory), (dev, dev_directory)]
        if requested
    ]
    for split_directory in splits:
        full_path = directory + f"sentiment.{split_directory}"
        examples = []
        for sentiment in sentiments:
            filename = full_path + f".{sentiment}"
            with open(filename, 'r', encoding="utf-8") as f:
                textnew = f.readlines()
                for lines in textnew:
                    examples.append({
                        'text': lines,
                        'sentiment': sentiment,
                    })
        x.append(examples)
    if len(x) == 1:
        return x[0]
    else:
        return tuple(x)


def yahoo_dataset(
        directory="data/filter_yahoo/",
        train=False,
        dev=False,
        test=False,
        train_directory="train",
        dev_directory="valid",
        test_directory="test"
):
    x = []
    splits = [
        dir_ for (requested, dir_) in [(train, train_directory), (test, test_directory), (dev, dev_directory)]
        if requested
    ]

    for split_directory in splits:
        full_path = os.path.join(directory, split_directory, f"{split_directory}.txt")
        examples = []
        with open(full_path, 'r', encoding="utf-8") as f:
            textnew = f.readlines()
            for lines in textnew:
                temp = eval(lines.strip())
                examples.append({
                    "text": temp['review'],
                    "sentiment": temp["score"],
                })
        x.append(examples)
    if len(x) == 1:
        return x[0]
    else:
        return tuple(x)
