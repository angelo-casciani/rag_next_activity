import datetime
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


"""

# True next activities and predicted next activities
true_activities = ['A', 'B', 'A', 'C', 'B']
predicted_activities = ['A', 'A', 'B', 'C', 'B']

# Unique activities (classes)
classes = list(set(true_activities + predicted_activities))

precision = precision_score(true_activities, predicted_activities, labels=classes, average='macro')
recall = recall_score(true_activities, predicted_activities, labels=classes, average='macro')
f1 = f1_score(true_activities, predicted_activities, labels=classes, average='macro')
accuracy = accuracy_score(true_activities, predicted_activities) # Accuracy = correct predictions / total predictions)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print(f"Macro Precision: {precision:.2f}")
print(f"Macro Recall: {recall:.2f}")
print(f"Macro F1-Score: {f1:.2f}")

"""
class VerificationOracle:
    def __init__(self, info_run):
        self.true_next_activities = []
        self.predicted_next_activities = []
        self.classes = list(set(self.true_next_activities + self.predicted_next_activities))
        self.prefix_with_expected_answer_pairs = {}
        self.accuracy = 0
        self.precision_macro = 0
        self.recall_macro = 0
        self.f1score_macro = 0
        self.results = []
        self.run_info = info_run


    def add_prefix_with_expected_answer_pair(self, prefix, expected_answer):
        self.prefix_with_expected_answer_pairs[prefix] = expected_answer

    """ Verifying the answer correctness.

    This method checks whether the model's answer matches the expected answer for a given prompt.
    """

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
            # expected_answer = '<' + expected_answer + '>'
            result['expected_answer'] = expected_answer
            if "Answer: <" in model_answer and '>' not in model_answer:
                model_answer = model_answer.split("Answer: ")[1] + '>'
            elif "Answer: " in model_answer and '<' not in model_answer:
                model_answer = '<' + model_answer.split("Answer: ")[1] + '>'
            result['verification_result'] = expected_answer.lower().replace(" ", "") in model_answer.lower().replace(" ", "")
            print(f"Prompt: {prompt}\nAnswer: {model_answer}\nExpected Answer: {expected_answer}\nResult: {result['verification_result']}")
        self.results.append(result)
        self.true_next_activities.append(expected_answer.strip('<').strip('>'))
        self.predicted_next_activities.append(model_answer.strip('<').strip('>'))

        return result['verification_result']

    """ Computing the metrics for the run.
    
       This method computes and stores the metrics for the run.
    """

    def compute_stats(self):
        self.precision_macro = precision_score(self.true_next_activities, self.predicted_next_activities, labels=self.classes, average='macro')
        self.recall_macro = recall_score(self.true_next_activities, self.predicted_next_activities, labels=self.classes, average='macro')
        self.f1score_macro = f1_score(self.true_next_activities, self.predicted_next_activities, labels=self.classes, average='macro')
        self.accuracy = accuracy_score(self.true_next_activities, self.predicted_next_activities)    # Accuracy = correct predictions / total predictions)

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
            file.write(f"Accuracy: {self.accuracy:.2f}\n")
            file.write(f"Precision (macro): {self.accuracy:.2f}\n")
            file.write(f"Recall (macro): {self.accuracy:.2f}\n")
            file.write(f"F1-score (macro): {self.accuracy:.2f}\n")
            file.write("-----------------------------------\n\n")

            for result in self.results:
                file.write(f"Prompt: {result['prompt']}\n")
                file.write(f"Model Answer: {result['model_answer']}\n")
                file.write(f"Expected Answer: {result['expected_answer']}\n")
                file.write(f"Verification Result: {result['verification_result']}\n")
                file.write("\n#####################################################################################\n")