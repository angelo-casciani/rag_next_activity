# -*- coding: utf-8 -*-
""" Python class for the verification oracle.

The class instantiate an oracle to check if the answers provided by the LLM are correct with respect to the
provided ground truth (prompt-expected answer pairs).
"""
import datetime
import os


class AnswerVerificationOracle:
    def __init__(self):
        self.prompt_expected_answer_pairs = []
        self.positives = 0
        self.negatives = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.f1score = 0
        self.results = []

    """ Adding the prompt-answer pairs.
    
    This method allows to add the prompt-expected answer pairs to the ground truth of the oracle.
    """

    def add_prompt_expected_answer_pair(self, prompt, expected_answer):
        """Add a prompt-expected answer pair to the oracle."""
        self.prompt_expected_answer_pairs.append((prompt, expected_answer))
        if expected_answer == "Yes":
            self.positives += 1
        elif expected_answer == "No":
            self.negatives += 1

    """ Verifying the answer correctness.

    This method checks whether the model's answer matches the expected answer for a given prompt.
    """

    def verify_answer(self, model_answer, prompt, llama3=False):
        if llama3:
            index_i = prompt.find('<|start_question_id|>')
            index_e = prompt.find('<|end_question_id|>')
            question = prompt[index_i + len('<|start_question_id|>'):index_e].strip()
        elif 'Question:' in prompt:
            index_i = prompt.find('Question: ')
            index_e = prompt.find('Answer: ')
            question = prompt[index_i + len('Question: '):index_e].strip()
        else:
            index_i = prompt.find('<<QUESTION>>')
            index_e = prompt.find('<</QUESTION>>')
            question = prompt[index_i + len('<<QUESTION>>'):index_e].strip()
        result = {
            'prompt': prompt,
            'model_answer': model_answer,
            'expected_answer': None,
            'verification_result': None
        }
        for prompt_text, expected_answer in self.prompt_expected_answer_pairs:
            if prompt_text == question:
                result['expected_answer'] = expected_answer
                result['verification_result'] = False
                for word in model_answer.split():
                    if expected_answer.lower() in word.strip(' .,').lower():
                        result['verification_result'] = True
                """
                if result['verification_result'] == False:
                    print(f'\n++++++++++\nRAG Answer: {model_answer}\nExpected Answer: {expected_answer}.\n++++++++++')
                    human_feedback = input('t - True or f - False: ')
                    if human_feedback == 't': result['verification_result'] = True
                """
                print('Answer:' + model_answer + ' - Result: ' + str(result['verification_result']))
                break

        self.results.append(result)
        return result['verification_result']

    """ Computing the metrics for the run.
    
       This method computes and stores the metrics for the run.
    """

    def compute_stats(self):
        total_results = len(self.results)
        correct_results = sum(int(result['verification_result']) for result in self.results)

        self.accuracy = (correct_results / total_results) * 100 if total_results > 0 else 0

        for result in self.results:
            if result['verification_result']:
                if result['expected_answer'] == 'Yes':
                    self.true_positives += 1
                else:
                    self.true_negatives += 1
            else:
                if result['expected_answer'] == 'Yes':
                    self.false_negatives += 1
                else:
                    self.false_positives += 1

        if self.true_positives + self.false_positives != 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives) * 100
        if self.true_positives + self.false_negatives != 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives) * 100
        if self.precision + self.recall != 0:
            self.f1score = 2 * (self.precision * self.recall) / (self.precision + self.recall) / 100

    """ Writing the verification results to a file.

    This method produces in output the results of the validation procedure. 
    """

    def write_results_to_file(self):
        file_path = os.path.join("..", "tests", "validation",
                                 f"results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt")
        self.compute_stats()

        with open(file_path, 'w') as file:
            file.write(f"Accuracy: {self.accuracy:.2f}%\n")
            file.write(f"Precision: {self.precision:.2f}%\n")
            file.write(f"Recall: {self.recall:.2f}%\n")
            file.write(f"F1-Score: {self.f1score:.2f}%\n\n")
            file.write("-----------------------------------\n\n")
            file.write(f"Positives: {self.positives}\n")
            file.write(f"True Positives: {self.true_positives}\n")
            file.write(f"False Negatives: {self.false_negatives}\n")
            file.write(f"Negatives: {self.negatives}\n")
            file.write(f"True Negatives: {self.true_negatives}\n")
            file.write(f"False Positives: {self.false_positives}\n\n")
            file.write("-----------------------------------\n\n")

            for result in self.results:
                file.write(f"Prompt: {result['prompt']}\n")
                file.write(f"Model Answer: {result['model_answer']}\n")
                file.write(f"Expected Answer: {result['expected_answer']}\n")
                file.write(f"Verification Result: {result['verification_result']}\n")
                file.write("\n#####################################################################################\n")
                