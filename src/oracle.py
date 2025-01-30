import datetime
import difflib
import os
import re
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class VerificationOracle:
    def __init__(self, info_run):
        self.true_next_activities = []
        self.predicted_next_activities = []
        self.classes = list(info_run['Activities'])
        self.prefix_with_expected_answer_pairs = {}
        self.accuracy = 0
        self.precision_macro = 0
        self.recall_macro = 0
        self.f1score_macro = 0
        self.results = []
        self.run_info = info_run
        self.start_time = time.time()
        self.end_time = 0
        self.elapsed_time = 0

    def add_prefix_with_expected_answer_pair(self, prefix, expected_answer):
        self.prefix_with_expected_answer_pairs[prefix] = expected_answer

    def verify_answer(self, prompt, prefix, model_answer):
        result = {
            'prompt': prompt,
            'prefix': prefix,
            'model_answer': model_answer,
            'expected_answer': None,
            'verification_result': None
        }
        expected_answer = self.prefix_with_expected_answer_pairs.get(prefix)
        if expected_answer is not None:
            result["expected_answer"] = expected_answer
            match = re.search(r'\\boxed\{([^}]*)\}', model_answer)
            if match:
                content = match.group(1).strip("'")
                model_answer = f'\\boxed{{{content}}}'
            else:
                model_answer = '\\boxed{Wrong format}'
            if "None" in model_answer:
                model_answer = '\\boxed{Wrong format}'

            model_answer = re.sub(r'\\text\{(.*?)\}', r'\1', model_answer)  # Remove \text{}
            model_answer = re.sub(r'[^a-zA-Z0-9\s]', '', model_answer).strip()  # Remove special characters
            closest_match = difflib.get_close_matches(model_answer, self.classes, n=1, cutoff=0.6)
            if closest_match:
                model_answer = closest_match[0]  # Set model_answer to the best matching label
            else:
                model_answer = '\\boxed{Wrong format}'

            result["verification_result"] = (
                    expected_answer.lower().replace(" ", "") in model_answer.lower().replace(" ", "")
            )
            print(
                f"Prompt: {prompt}\n"
                f"Answer: {model_answer}\n"
                f"Expected Answer: {expected_answer}\n"
                f"Result: {result['verification_result']}"
            )
            if result["verification_result"]:
                model_answer = expected_answer

        self.results.append(result)
        self.true_next_activities.append(expected_answer.removeprefix("\\boxed{").removesuffix("}"))
        self.predicted_next_activities.append(model_answer.removeprefix("\\boxed{").removesuffix("}"))

        return result["verification_result"]

    def compute_stats(self):
        # self.classes = list(set(self.true_next_activities + self.predicted_next_activities))
        self.precision_macro = precision_score(self.true_next_activities, self.predicted_next_activities,
                                               labels=self.classes, average='macro')
        self.recall_macro = recall_score(self.true_next_activities, self.predicted_next_activities, labels=self.classes,
                                         average='macro')
        self.f1score_macro = f1_score(self.true_next_activities, self.predicted_next_activities, labels=self.classes,
                                      average='macro')
        self.accuracy = accuracy_score(self.true_next_activities,
                                       self.predicted_next_activities)
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time

    """ Writing the verification results to a file.

    This method produces in output the results of the validation procedure. 
    """

    def write_results_to_file(self):
        file_path = os.path.join(os.path.dirname(__file__), "..", "tests", "validation",
                                 f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
        self.compute_stats()

        with open(file_path, 'w') as file:
            file.write('INFORMATION ON THE RUN\n\n')
            for key in self.run_info.keys():
                file.write(f"{key}: {self.run_info[key]}\n")
            file.write('\n-----------------------------------\n')
            file.write(f"Accuracy: {self.accuracy:.4f}\n")
            file.write(f"Precision (macro): {self.precision_macro:.4f}\n")
            file.write(f"Recall (macro): {self.recall_macro:.4f}\n")
            file.write(f"F1-score (macro): {self.f1score_macro:.4f}\n")
            file.write(f"{self.classes}\n")
            file.write(f"Elapsed time: {self.elapsed_time:.2f} seconds\n")
            file.write("-----------------------------------\n\n")

            for result in self.results:
                file.write(f"Prompt: {result['prompt']}\n")
                file.write(f"Model Answer: {result['model_answer']}\n")
                file.write(f"Expected Answer: {result['expected_answer']}\n")
                file.write(f"Verification Result: {result['verification_result']}\n")
                file.write("\n#####################################################################################\n")
