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
            traces_list.append('; '.join(trace_content))
    traces_list = list(dict.fromkeys(traces_list))
    return traces_list


def extract_traces_with_attributes(log_content):
    trace_pattern = re.compile(r'<trace>.*?</trace>', re.DOTALL)
    event_pattern = re.compile(r'<event>.*?</event>', re.DOTALL)
    attribute_pattern = re.compile(r'key="(.*?)" value="(.*?)"/>')
    traces_list = []
    keys = []
    for trace_match in trace_pattern.findall(log_content):
        trace_content = []
        for event_match in event_pattern.findall(trace_match):
            attributes = []
            for attribute_match in attribute_pattern.findall(event_match):
                key, value = attribute_match
                #if key != "lifecycle:transition" and key != "time:timestamp":
                if key != "lifecycle:transition":
                    key_initial = ''.join([part[:2] for part in key.split(':')])
                    attributes.append(f'{key_initial}:{value}')
                    if f'{key_initial}: {key}' not in keys:
                        keys.append(f'{key_initial}: {key}')
            if attributes:
                trace_content.append(','.join(attributes))
        if len(trace_content) >= 2:
            traces_list.append('; '.join(trace_content))
    traces_list = list(dict.fromkeys(traces_list))
    return traces_list, keys


# Test set proportion must be a decimal from 0 to 1
def generate_test_set(traces, test_set_proportion):
    test_set_size = int(len(traces) * test_set_proportion)
    test_set = random.sample(traces, test_set_size)
    return test_set


def generate_csv_from_test_set(test_set, test_path, base=1, gap=3):
    tests = []
    for trace in test_set:
        trace = trace.split('; ')
        """prefix = '; '.join(trace[:-1]) + ';'
        prediction = trace[-1]
        csvwriter.writerow([prefix, prediction])"""
        """if len(trace) <= 3:
            indices = range(1, len(trace))
        else:
            indices = [random.randint(1, len(trace)//2), len(trace)//2,
                       random.randint(len(trace)//2 + 1, len(trace) - 1)]"""
        indices = [i for i in range(base, len(trace), gap)]
        for index in indices:
            if index < len(trace):
                prefix = '; '.join(trace[:index]) + ';'
                #prediction = trace[index].split('concept:name ')[1].split(',')[0]
                prediction = trace[index].split('cona:')[1].split(',')[0]
                tests.append([prefix, prediction])

    with open(test_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['prefix', 'prediction'])
        for pair in tests:
            csvwriter.writerow([pair[0], pair[1]])


def compute_log_stats(log_name):
    tree_content = read_event_log(log_name)
    traces = extract_traces(tree_content)
    print(f'Total number of traces: {len(traces)}')
    total_events = sum(len(trace) for trace in traces)
    print(f'Total number of events: {total_events}')


def main():
    test_set_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_sets', f"sintetico-2-2var-1rel-1-nonrel.csv")
    content = read_event_log('sintetico-2-2var-1rel-1-nonrel.xes')
    traces, event_attributes = extract_traces_with_attributes(content)
    print(event_attributes)
    test_set = generate_test_set(traces, 0.2)
    generate_csv_from_test_set(test_set=test_set, test_path=test_set_path)

if __name__ == '__main__':
    main()