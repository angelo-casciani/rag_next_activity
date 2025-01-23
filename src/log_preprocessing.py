import csv
import os
import pm4py
import random
import re


def read_event_log(filename):
    file_path = os.path.join(os.path.dirname(__file__), '..', 'logs', filename)
    with open(file_path, 'r') as file:
        return file.read()


def extract_traces(log_content):
    pattern_traces = re.findall(r"<trace>(.*?)</trace>", log_content, re.DOTALL)
    traces_result = []
    c = 1
    for single_trace in pattern_traces:
        events = re.findall(r"<event>(.*?)</event>", single_trace, re.DOTALL)
        if len(events) >= 2:
            traces_result.append(single_trace)
            print(f"Processed trace: {c}")
            c += 1
    print(f"Total traces: {c}")
    return traces_result


def build_prefixes(traces, base=1, gap=3):
    prefixes = []
    for trace in traces:
        events = re.findall(r"<event>.*?</event>", trace, re.DOTALL)
        if len(events) <= 2:
            continue
        else:
            for i in range(base, len(events), gap):
                i += 1
                if i <= len(events):
                    prefix = events[:i]
                    prefixes.append(''.join(prefix))
    return prefixes


def process_prefixes(traces):
    concept_name_pattern = re.compile(r'<string key="concept:name" value="(.*?)"\s*/>', re.DOTALL)
    lifecycle_transition_pattern = re.compile(r'<string key="lifecycle:transition" value="(.*?)"\s*/>', re.DOTALL)
    attribute_pattern = re.compile(r'key="(.*?)" value="(.*?)"/>')
    seen_prefixes = []
    results = {}
    keys = {}
    activities = set()
    for i in range(0, len(traces)):
        trace = traces[i]
        events = re.findall(r"<event>.*?</event>", trace, re.DOTALL)
        cl_list = []
        for event in events:
            concept_name_match = concept_name_pattern.search(event)
            lifecycle_transition_match = lifecycle_transition_pattern.search(event)
            if concept_name_match and lifecycle_transition_match:
                event_name = f"{concept_name_match.group(1)}+{lifecycle_transition_match.group(1)}"
                cl_list.append(event_name)
                activities.add(event_name)
            elif concept_name_match:
                event_name = concept_name_match.group(1)
                cl_list.append(event_name)
                activities.add(event_name)
        cl_list_string = ','.join(cl_list)
        if cl_list_string in seen_prefixes:
            continue
        else:
            seen_prefixes.append(cl_list_string)
        last_evt = cl_list[-1]
        attr_vals = {}
        for evt in events:
            for attribute_match in attribute_pattern.findall(evt):
                key, value = attribute_match
                if key != "lifecycle:transition" and key != "concept:name":
                    key_initial = ''.join([part[:2] for part in key.split(':')])
                    attr_vals[key_initial] = value
                    keys[key_initial] = key

        if cl_list_string not in results:
            results[cl_list_string] = f'Values: {str(attr_vals)} | Next activity: {last_evt}'
        #else:
        #    results[cl_list_string] = f'Values: {attr_vals} | <{last_evt}>'
        print(f"Processed prefix {i}/{len(traces)}")
    print(f"Total unique prefixes: {len(results)}")
    return results, keys, activities


# Test set proportion must be a decimal from 0 to 1
def generate_test_set(traces, test_set_proportion):
    test_set_size = int(len(traces) * test_set_proportion)
    test_set = random.sample(traces, test_set_size)
    return test_set


def generate_csv_from_test_set(test_set, test_path, size=300):
    test_set = build_prefixes(test_set)
    test_set, attr_keys, act_list = process_prefixes(test_set)
    if size > len(test_set):
        size = len(test_set)
    test_set = dict(random.sample(list(test_set.items()), size))
    with open(test_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['prefix', 'prediction'])
        for prefix, prediction in test_set.items():
            attributes = prediction.split('|')[0].strip()
            next_activity = '<' + prediction.split('| Next activity: ')[1].strip() + '>'
            csvwriter.writerow([f'{prefix} - Values: {attributes}', next_activity])


"""Old functions BEGIN

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
    keys = {}
    activities = set()
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
                    if key == 'concept:name':
                        activities.add(value)
                    if key_initial not in keys and key_initial != 'cona':
                        keys[key_initial] = key
            if attributes:
                trace_content.append(','.join(attributes))
        if len(trace_content) >= 2:
            traces_list.append('; '.join(trace_content))
    traces_list = list(dict.fromkeys(traces_list))
    return traces_list, keys, activities


def create_prefixes_with_attribute_last_values(traces, base, gap):
    prefixes_predictions = []
    for trace in traces:
        trace = trace.split('; ')
        indices = [i for i in range(base, len(trace), gap)]
        for index in indices:
            if index < len(trace):
                prefix = '; '.join(trace[:index]) + ';'
                prediction = trace[index].split('; ')[0].split(']')[0]
                prefix = prefix.rstrip(';') + ']'
                prefixes_predictions.append((prefix, prediction))
    return prefixes_predictions


def process_traces_with_last_attribute_values(traces):
    processed_traces = []
    for trace in traces:
        concept_name_trace = []
        attribute_values = {}
        trace = trace.split('; ')
        for event in trace:
            concept_name = event.split('cona:')[1].split(',')[0]
            concept_name_trace.append(concept_name)
            event_attributes = event.split(',')
            for attribute in event_attributes:
                attribute_key = attribute.split(':')[0]
                attribute_value = attribute.split(':')[1]
                if attribute_key != 'cona':
                    attribute_values[attribute_key] = attribute_value
        trace_attributes = str(concept_name_trace) + ' - Values: ' + str(attribute_values)
        processed_traces.append(trace_attributes)
    return processed_traces


def process_prefixes_prediction_with_last_attribute_values(prefix_prediction_pairs):
    processed_prefixes = []
    for pair in prefix_prediction_pairs:
        concept_name_trace = []
        attribute_values = {}
        prefix = pair[0].split('; ')
        prediction = pair[1].split('cona:')[1].split(',')[0]
        for event in prefix:
            concept_name = event.split('cona:')[1].split(',')[0]
            concept_name_trace.append(concept_name)
            event_attributes = event.split(',')
            for attribute in event_attributes:
                attribute_key = attribute.split(':')[0]
                attribute_value = attribute.split(':')[1]
                if attribute_key != 'cona':
                    attribute_values[attribute_key] = attribute_value
        proc_prefix_attributes = str(concept_name_trace) + ' - Values: ' + str(attribute_values)
        processed_prefixes.append((proc_prefix_attributes, prediction))
    return processed_prefixes

    # Test set proportion must be a decimal from 0 to 1
    def generate_test_set(traces, test_set_proportion):
        test_set_size = int(len(traces) * test_set_proportion)
        test_set = random.sample(traces, test_set_size)
        return test_set


    def generate_csv_from_test_set(test_set, test_path):
        size = 300
        test_set = random.sample(test_set, size)
        with open(test_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['prefix', 'prediction'])
            for pair in test_set:
                csvwriter.writerow([pair[0], pair[1]])

Old functions END"""




"""def compute_log_stats(log_name):
    tree_content = read_event_log(log_name)
    traces = extract_traces(tree_content)
    print(f'Total number of traces: {len(traces)}')
    total_events = sum(len(trace) for trace in traces)
    print(f'Total number of events: {total_events}')


def main():
    test_set_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_sets',
                                 f"sintetico-2-2var-1rel-1-nonrel.csv")
    content = read_event_log('sintetico-2-2var-1rel-1-nonrel.xes')
    traces, event_attributes, activities = extract_traces_with_attributes(content)
    print(activities)
    print(event_attributes)
    traces_to_store = process_traces_with_last_attribute_values(traces)
    test_set = generate_test_set(traces, 0.2)
    prefix_prediction = create_prefixes_with_attribute_last_values(test_set, base=1, gap=3)
    prefix_prediction_pairs = process_prefixes_prediction_with_last_attribute_values(prefix_prediction)
    generate_csv_from_test_set(test_set=prefix_prediction_pairs, test_path=test_set_path)


if __name__ == '__main__':
    main()
"""

"""def main():
    content = read_event_log('BPI_Challenge_2012.xes')
    traces = extract_traces(content)
    # print(traces[0])
    prefixes = build_prefixes(traces)
    # print(len(prefixes))
    # print(prefixes[0])
    prefixes, keys, activities = process_prefixes(prefixes)

    test_set = generate_test_set(traces, 0.3)
    test_set_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_sets',
                                 f"BPI_Challenge_2012.csv")
    generate_csv_from_test_set(test_set, test_set_path)
    

if __name__ == '__main__':
    main()"""