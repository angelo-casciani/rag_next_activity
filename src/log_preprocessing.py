import csv
import os
import random
import re


def read_event_log(filename):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', filename)
    with open(file_path, 'r') as file:
        return file.read()


# Extract traces with at least two events
def extract_traces(log_content):
    trace_pattern = re.compile(r'<trace>.*?</trace>', re.DOTALL)
    event_pattern = re.compile(r'<event>.*?</event>', re.DOTALL)
    traces_list = []
    for trace_match in trace_pattern.findall(log_content):
        trace_content = []
        for event_match in event_pattern.findall(trace_match):
            trace_content.append(event_match)
        if len(trace_content) >= 2:
            traces_list.append(trace_content)
    return traces_list


# Generates all possible prefixes for each trace (having at least two events)
def generate_unique_prefixes(traces_list):
    all_prefixes = []
    for trace in traces_list:
        for i in range(2, len(trace) + 1):
            prefix = ''.join(trace[:i])
            if prefix not in all_prefixes:
                all_prefixes.append(prefix)
    return all_prefixes


# Generates all prefixes of length 4 for each trace
def generate_prefix_windows(traces_list):
    prefix_windows = []
    for trace in traces_list:
        for i in range(0, len(trace) + 1, 4):
            j = i + 4
            if j > len(trace) + 1:
                prefix_window = trace[i:]
            else:
                prefix_window = trace[i:j]
            prefix_window = ''.join(prefix_window)
            if prefix_window not in prefix_windows:
                prefix_windows.append(prefix_window)
    return prefix_windows


def extract_traces_concept_names(log_content):
    trace_pattern = re.compile(r'<trace>.*?</trace>', re.DOTALL)
    event_pattern = re.compile(r'<event>.*?</event>', re.DOTALL)
    concept_name_pattern = re.compile(r'<string key="concept:name" value="(.*?)"/>')
    traces_list = []
    for trace_match in trace_pattern.findall(log_content):
        trace_content = []
        for event_match in event_pattern.findall(trace_match):
            concept_name_match = concept_name_pattern.search(event_match)
            if concept_name_match:
                concept_name = concept_name_match.group(1)
                trace_content.append(concept_name)
        if len(trace_content) >= 2:
            traces_list.append(trace_content)
    return traces_list


def generate_training_and_test_set(traces, test_set_size):
    test_set = random.sample(traces, test_set_size)
    training_set = [trace for trace in traces if trace not in test_set]
    return training_set, test_set


def generate_csv_from_test_set(test_set, filename):
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['question', 'answer'])
        for trace in test_set:
            for i in range(1, len(trace)):
                question = ' '.join(trace[:i])
                answer = trace[i]
                csvwriter.writerow([question, answer])


def compute_log_stats(log_name):
    tree_content = read_event_log(log_name)
    traces = extract_traces(tree_content)
    print(f'Total number of traces: {len(traces)}')
    total_events = sum(len(trace) for trace in traces)
    print(f'Total number of events: {total_events}')
    

"""log_name = 'Hospital_log.xes'
tree_content = read_event_log(log_name)
# compute_log_stats(tree_content)
traces = extract_traces(tree_content)
print(trace for trace in traces)
prefixes = generate_prefix_windows(traces)
print(prefixes[:5])
print(len(prefixes))
training_set, test_set = generate_training_and_test_set(traces, 10)
generate_csv_from_test_set(test_set, 'test.csv')
with open('test.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for i, row in enumerate(csvreader):
        if i < 3:
            print(row)
        else:
            break"""