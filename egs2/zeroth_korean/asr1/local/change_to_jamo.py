import os
import re
import sys
import jamo
import logging
import argparse


def main():
    parser = argparse.ArgumentParser(description='Change Korean sentences to jamo sentences.')
    parser.add_argument(
        "-d",
        "--input_dir",
        type=str,
        default=None,
        help="Directory which Input text file of sentences exists",
    )
    parser.add_argument(
        "-f",
        "--text_fname",
        type=str,
        default="text",
        help="Input text file name",
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    input_dir = args.input_dir
    text_fname = args.text_fname
    text_file_path = os.path.join("data", input_dir, text_fname)
    jamo_file_path = text_file_path + ".jamo"
 
    if not os.path.isfile(text_file_path):
        raise FileNotFoundError("[{}] does not exist.".format(text_file_path))

    logging.info(f"{text_file_path} is processing...")
    count = 0
    with open(text_file_path, "r", encoding="utf-8") as rf, \
        open(jamo_file_path, "a", encoding="utf-8") as wf:
        line = rf.readline()

        while line:
            line = line.strip()
            key, text = re.split("\s+", line, maxsplit=1)
            jamo_text = jamo.j2hcj(jamo.h2j(text))
            jamo_line = "{} {}".format(key, jamo_text)
            wf.write(jamo_line + "\n")

            count += 1
            if count % 100 == 0:
                print(f"{count} lines", end='\r')

            line = rf.readline()
    logging.info(f"{count} is done!")

if __name__ == "__main__":
    main()
