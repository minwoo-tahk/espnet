import os
import re
import sys
# import jamo
import logging
import argparse
from hangul_utils import join_jamos

def main():
    parser = argparse.ArgumentParser(description='Change Korean sentences to jamo sentences.')
    parser.add_argument(
        "-f",
        "--input_file",
        type=str,
        default=None,
        help="Input text file of jamo sentences",
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    input_file = args.input_file
    text_file_path = os.path.splitext(input_file)[0]

    if not os.path.isfile(input_file):
        raise FileNotFoundError("[{}] does not exist.".format(input_file))

    logging.info(f"{input_file} is processing...")
    count = 0
    with open(input_file, "r", encoding="utf-8") as rf, \
        open(text_file_path, "a", encoding="utf-8") as wf:
        line = rf.readline()
        # print(f"line: {line}")
        while line:
            line = line.strip()
            key, jamo_text = re.split("\s+", line, maxsplit=1)
            text = join_jamos(jamo_text)
            new_line = "{} {}".format(key, text)
            wf.write(new_line + "\n")

            count += 1
            if count % 100 == 0:
                print(f"{count} lines", end='\r')

            line = rf.readline()
    logging.info(f"{count} is done!")

if __name__ == "__main__":
    main()
