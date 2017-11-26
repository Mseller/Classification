import tensorflow as tf
import sys

DATA_DIR = '/Users/martinseller/Documents/Programing/Python/Classification/data/'

UNPROCESSED_TRAINING_FILE = "{}{}".format(DATA_DIR, 'adult.data.unprocessed.csv')
UNPROCESSED_EVAL_FILE = "{}{}".format(DATA_DIR, 'adult.test.unprocessed.csv')

TRAINING_FILE = "{}{}".format(DATA_DIR, 'adult.data.csv')
EVAL_FILE = "{}{}".format(DATA_DIR, 'adult.test.csv')


def clean_data(unprocessed_file, output_file):
    with tf.gfile.Open(unprocessed_file, 'r') as temp_eval_file:
        with tf.gfile.Open(output_file, 'w') as eval_file:
            for line in temp_eval_file:
                line = line.strip()  # Removes any whitespace from the line
                line = line.replace(', ', ',')  # Replaces all the commas
                if not line or ',' not in line:
                    continue
                if line[-1] == '.':
                    line = line[:-1]
                line += '\n'
                eval_file.write(line)


def main():
    if not tf.gfile.Exists(UNPROCESSED_TRAINING_FILE):
        print("The file {} does not exists".format(UNPROCESSED_TRAINING_FILE))
        sys.exit()
    if not tf.gfile.Exists(UNPROCESSED_EVAL_FILE):
        print("The file {} does not exists".format(UNPROCESSED_EVAL_FILE))
        sys.exit()
    if not tf.gfile.Exists(TRAINING_FILE):
        print("The file {} does not exists".format(TRAINING_FILE))
        sys.exit()
    if not tf.gfile.Exists(EVAL_FILE):
        print("The file {} does not exists".format(EVAL_FILE))
        sys.exit()

    clean_data(UNPROCESSED_TRAINING_FILE, TRAINING_FILE)
    clean_data(UNPROCESSED_EVAL_FILE, EVAL_FILE)


if __name__ == '__main__':
    main()
