# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is a metric module to do Partial Answer & Justification Match (pajm).
"""

import evaluate
import datasets

from typing import List, Tuple
from rouge_score import rouge_scorer

_CITATION = """\
"""

_DESCRIPTION = """\
This is a metric module to do Partial Answer & Justification Match (pajm).

Answers are scored by exact matched.
Justifications are scored by Rouge-L score.

These two scores are merged by weighted average with a given `justification_weight`.
TODO: try harmonic mean instead of weighted average
"""


_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    score: partial match score of answer and justifications
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> pajm_metric = evaluate.load("pajm", justification_weight=0.5)
    >>> results = pajm_metric.compute(references=['(1) Answer: Yes. Justification: "golden evidence"'],
    ...                               predictions=['(1) Answer: Yes. Justification: "model hallucinations"'])
    >>> print(results)
    {'score': 0.5}
"""


def remove_prefix(text: str, prefix: str) -> str:
    """
    remove {prefix} from {text}.

    this function is no longer needed for python >= 3.9 since the 'str' class
    has a built-in method `removeprefix`.
    """
    return text[len(prefix):] if text.startswith(prefix) else text


def split_justifications(text: str, separators: List[str]) -> List[str]:
    if not text or not text.strip():
        return []

    text = text.replace('“', '"').replace('”', '"')
    current_index = 0
    separator_indices = []
    while current_index >= 0: # -1 if there is no match
        next_idx = float('inf')
        next_sep = None
        for sep in separators:
            idx = text.find(sep, current_index)
            if idx < 0: # no match
                continue
            if idx < next_idx:
                next_idx = idx
                next_sep = sep
        if not next_sep:
            break
        current_end = next_idx + 1
        next_start = next_idx + len(next_sep) - 1
        separator_indices += [current_end, next_start]
        current_index = next_start + 1
    if not separator_indices: # only one justification
        return [text]

    separator_indices = [0] + separator_indices + [len(text)]
    justifications = []
    for i in range(0, len(separator_indices), 2):
        start_idx = separator_indices[i]
        end_idx = separator_indices[i+1]
        justifications.append(text[start_idx:end_idx].strip("\"' "))
    return justifications

def split_number_list(text: str) -> List[str]:
    start_indices = []
    list_number = 1
    while True:
        number_index = text.find(f"({list_number})")
        if number_index < 0:
            break
        start_indices.append(number_index)
        list_number += 1

    if not start_indices:
        return []
    if start_indices[0] != 0:
        return []

    items = []
    for i, start_index in enumerate(start_indices):
        end_index = start_indices[i + 1] if i < len(start_indices) - 1 else len(text)
        span = text[start_index:end_index]
        number_len = len(f"({i + 1})")
        span = span.strip()[number_len:].strip() # remove the number index
        items.append(span)
    return items

def split_answer_justification(text: str) -> Tuple[str, str]:
    # results are converted to lowercase before comparison
    answer_tag = "answer:"
    justification_tag = "justification:"

    if not text.startswith(answer_tag):
        return None, None
    if justification_tag not in text:
        return None, None

    justification_index = text.find(justification_tag)
    answer = text[:justification_index]
    justification = text[justification_index:]

    answer = remove_prefix(answer, answer_tag).strip().rstrip('.')
    justification = remove_prefix(justification, justification_tag).strip().rstrip('.')

    return answer, justification


def parse_completion(text: str, has_justification: bool=True) -> List[Tuple[str, str]]:
    answer_list = split_number_list(text.strip().lower())
    return [split_answer_justification(l) for l in answer_list] if has_justification else answer_list


def split_answer_justification_cot(text: str) -> Tuple[str, str]:
    # results are converted to lowercase before comparison
    answer_tag = "so the answer is"
    justification_tag = "the clinical note says"

    if answer_tag not in text:
        return None, None
    answer_index = text.find(answer_tag)
    answer = text[answer_index:]
    answer = remove_prefix(answer, answer_tag).strip().rstrip('.').strip("\"' ")

    if answer == 'na':
        justification = 'na'
    else:
        justification = text[:answer_index].strip()
        justification = remove_prefix(justification, justification_tag).strip().rstrip('.')
    return answer, justification


def parse_completion_cot(text: str, has_justification: bool=True) -> List[Tuple[str, str]]:
    answer_list = split_number_list(text.strip().lower())
    return [split_answer_justification_cot(l) for l in answer_list] if has_justification else answer_list


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class PAJM(evaluate.Metric):
    """TODO: Short description of my evaluation module."""
    def __init__(self, *, has_justification=True, justification_weight=0.5, completion_format="label", **kwargs):
        completion_parser_map = {
            "label": parse_completion,
            "labeled": parse_completion,
            "chain-of-though": parse_completion_cot,
            "cot": parse_completion_cot,
        }
        if completion_format not in completion_parser_map:
            raise ValueError(f"Supported completion formats are: {list(completion_parser_map.keys())}, but got '{completion_format}'")

        super().__init__(**kwargs)

        # set the justification flag and weight
        self._has_justification = has_justification
        self._justification_weight = justification_weight

        self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

        justification_delimiters = [',', ';', 'and']
        self._justification_separators = []
        for d in justification_delimiters:
            for left_space in ['"', '" ']:
                for right_space in [' "', '"']:
                    self._justification_separators.append(left_space + d + right_space)
        self._completion_parser = completion_parser_map[completion_format]

    def _info(self):
        # TODO: Specifies the evaluate.EvaluationModuleInfo object
        return evaluate.MetricInfo(
            # This is the description that will appear on the modules page.
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # Homepage of the module for documentation
            homepage="http://module.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_module"],
            reference_urls=["http://path.to.reference.url/new_module"]
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        # TODO: Download external resources if needed
        pass

    def _compute(self, predictions, references):
        """Returns the scores"""
        score = self._pajm_score(predictions, references)
        return {
            "score": score,
        }


    def _score_justifications(self, predicted_justifications, gold_justifications):
        targets = split_justifications(gold_justifications, self._justification_separators)
        predictions = split_justifications(predicted_justifications, self._justification_separators)
        return self._scorer.score(". ".join(targets), ". ".join(predictions))["rougeL"].fmeasure


    def _score_prediction(self, predicted_result, golden_result):
        score = 0.0

        for pred, gold in zip(predicted_result, golden_result):
            if self._has_justification:
                # only check the correctness of justifications if answer is correct
                if pred[0] == gold[0]:
                    score += 1.0 * (1.0 - self._justification_weight) + \
                             self._score_justifications(pred[1], gold[1]) * self._justification_weight
            else:
                if pred == gold:
                    score += 1.0

        return score / max(1, len(golden_result))


    def _single_pajm_score(self, prediction, golden_completion):
        predicted_result = self._completion_parser(prediction, self._has_justification)
        golden_result = self._completion_parser(golden_completion, self._has_justification)
        return self._score_prediction(predicted_result, golden_result)


    def _pajm_score(self, predictions, golden_completions):
        scores = [self._single_pajm_score(p, d) for p, d in zip(predictions, golden_completions)]
        return sum(scores) / len(scores) if scores else 0.0
