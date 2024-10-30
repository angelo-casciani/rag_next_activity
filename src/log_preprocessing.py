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


# Test set proportion must be a decimal from 0 to 1
def generate_test_set(traces, test_set_proportion):
    test_set_size = int(len(traces) * test_set_proportion)
    test_set = random.sample(traces, test_set_size)
    return test_set


def generate_csv_from_test_set(test_set, log_name):
    test_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'validation',
                         f"test_set_{log_name.split('.xes')[0]}.csv")
    with open(test_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['prefix', 'prediction'])
        for trace in test_set:
            for i in range(1, len(trace)):
                prefix = ', '.join(trace[:i])
                prediction = trace[i]
                csvwriter.writerow([prefix, prediction])
    
    return test_path


def compute_log_stats(log_name):
    tree_content = read_event_log(log_name)
    traces = extract_traces(tree_content)
    print(f'Total number of traces: {len(traces)}')
    total_events = sum(len(trace) for trace in traces)
    print(f'Total number of events: {total_events}')
    

"""log_name = 'Hospital_log.xes'
tree_content = read_event_log(log_name)
# print(compute_log_stats(tree_content))
traces = extract_traces_concept_names(tree_content)
# print(trace for trace in traces)
# prefixes = generate_prefix_windows(traces)
# print(prefixes[:5])
# print(len(prefixes))
training_set, test_set = generate_test_set(traces, 0.3)
test_path = generate_csv_from_test_set(test_set, log_name)
with open(test_path, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for i, row in enumerate(csvreader):
        if i < 3:
            print(row)
        else:
            break"""