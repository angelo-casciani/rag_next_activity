import csv
import os
import torch
import random
import argparse
import numpy as np

def seed_everything(seed=10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_process_representation(filename):
    filepath = os.path.join("..", "data", filename)
    with open(filepath, 'r') as file:
        file_content = file.read()
        return file_content


def load_csv_questions(filename):
    filepath = os.path.join("..", 'data', 'questions', filename)
    questions = {}
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            question, answer = row
            questions[question] = answer
        return questions


def log_to_file(message, curr_datetime):
    filepath = os.path.join("..", "tests", "outputs", f"output_{curr_datetime}.txt")
    with open(filepath, 'a') as file:
        file.write(message)