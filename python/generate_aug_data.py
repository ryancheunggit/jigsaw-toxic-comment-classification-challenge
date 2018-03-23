"""Based on https://github.com/PavelOstyakov/toxic/blob/master/tools/extend_dataset.py"""

from joblib import Parallel, delayed
from textblob import TextBlob
from textblob.translate import NotTranslated
import argparse
import os
import pandas as pd

NAN_WORD = "_NAN_"

LANGUAGE_CODES = [
        "sq", "am", "ar", "hy", "az", "eu", "bn", "bs", "bg", "ca", "ceb",
        "zh-CN *", "zh-TW", "co", "hr", "cs", "da", "nl", "eo", "et", "fi",
        "fr", "fy", "gl", "ka", "de", "el", "gu", "ht", "ha", "haw", "iw",
        "hi", "hmn", "hu", "is", "ig", "id", "ga", "it", "ja", "jw", "kn",
        "kk", "km", "ko", "ku", "lo", "lv", "lt", "lb", "mk", "mg", "ms",
        "ml", "mt", "mi", "mr", "mn", "ne", "no", "ny", "ps", "fa", "pl",
        "pt", "pa", "ro", "ru", "sm", "gd", "sr", "st", "sn", "sd", "si",
        "sk", "sl", "so", "es", "sw", "sv", "tl", "tg", "ta", "te", "th",
        "tr", "uk", "ur", "uz", "vi", "cy", "xh", "yi", "yo", "zu"
    ]

def translate(comment, language):
    if hasattr(comment, "decode"):
        comment = comment.decode("utf-8")
    text = TextBlob(comment)
    try:
        text = text.translate(to=language)
        text = text.translate(to="en")
    except NotTranslated:
        pass
    return str(text)


def main():
    parser = argparse.ArgumentParser("Script for extending train dataset")
    parser.add_argument("--languages", nargs="+", default=LANGUAGE_CODES)
    args = parser.parse_args()

    train_data = pd.read_csv("../input/train.csv")
    comments_list = train_data["comment_text"].fillna(NAN_WORD).values
    try:
        os.system("mkdir ../input/ext")
    except:
        pass

    parallel = Parallel(16, backend="threading", verbose=5)
    for i, language in enumerate(args.languages):
        print('Translate comments using "{0}" language {1}/{2}'.format(language, i+1, len(args.languages)))
        print("Total : {}".format(len(comments_list)))
        translated_data = parallel(delayed(translate)(comment, language) for comment in comments_list)
        # translated_data = [translate(comment, language) for comment in tqdm(comments_list)]
        train_data["comment_text"] = translated_data
        train_data.to_csv("../input/ext/train_{}.csv".format(language.lower()), index=False)

if __name__ == "__main__":
    main()
