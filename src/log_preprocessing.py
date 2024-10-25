import csv
import os
import random
import xml.etree.ElementTree as ET


def read_event_log(filename):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', filename)
    tree = ET.parse(file_path)
    return tree


# Extract traces with at least two events
def extract_traces(tree):
    root = tree.getroot()
    traces = []
    for trace in root.findall('{http://www.xes-standard.org/}trace'):
        trace_content = []
        for event in trace.findall('{http://www.xes-standard.org/}event'):
            event_str = ET.tostring(event, encoding='unicode')
            event_str = event_str.replace('ns0:', '').replace(' xmlns:ns0="http://www.xes-standard.org/"', '')
            trace_content.append(event_str)
        if len(trace_content) >= 2:
            traces.append(trace_content)
    return traces


# Generates all possible prefixes for each trace (having at least two events)
def generate_unique_prefixes(traces):
    all_prefixes = set()
    for trace in traces:
        for i in range(2, len(trace) + 1):
            prefix = tuple(trace[:i])
            all_prefixes.add(prefix)
    return [list(prefix) for prefix in all_prefixes]


# Generates all prefixes of length 4 for each trace
def generate_prefix_windows(traces):
    prefix_windows = []
    for trace in traces:
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
    

log_name = 'Hospital_log.xes'
tree_content = read_event_log(log_name)
# compute_log_stats(tree_content)
traces = extract_traces(tree_content)
prefixes = generate_prefix_windows(traces)
print(prefixes[:5])
print(len(prefixes))
"""training_set, test_set = generate_training_and_test_set(traces, 10)
generate_csv_from_test_set(test_set, 'test.csv')
with open('test.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for i, row in enumerate(csvreader):
        if i < 3:
            print(row)
        else:
            break"""