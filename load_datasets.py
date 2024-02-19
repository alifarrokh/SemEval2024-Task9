"""
Load datasets
"""
import re
import random
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets, interleave_datasets
from utils import chunks, suppress


class DatasetManager:
    """Load, preprocess, and merge datasets"""

    def __init__(self, ignore_case: bool, force_4_choices: bool, ds_format: str):
        assert ds_format in ['bert', 't5'], "Invalid dataset format"
        assert ds_format == 'bert' or not force_4_choices, "Don't force the number of choices in T5."
        self.ignore_case = ignore_case
        self.force_4_choices = force_4_choices
        self.ds_format = ds_format


    def __clean_text(self, text):
        """Clean a piece of text"""
        text = text.replace(',', ', ')
        text = re.sub(r'\?+', '?', text)
        text = text.replace('\n', ' ')
        text = re.sub(r'(\.\s?)+', '. ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if self.ignore_case:
            text = text.lower()
        return text


    def __t5_format(self, question: str, choices: list, answer_key: str):
        """Convert a question into T5 format"""
        answer_keys = list('abcde')
        answer_index = np.argmax([int(k == answer_key) for k in answer_keys])
        label = choices[answer_index]

        result = f"{question} \\n"
        for i, c in zip(answer_keys, choices):
            result += f" ({i.upper()}) {c}"

        return {'text': result, 'label': label}


    def __bert_format(self, question: str, choices: list, answer_key: str):
        """Convert a question into BERT format"""
        answer_keys = list('abcde')
        answer_index = np.argmax([int(k == answer_key) for k in answer_keys])
        result = {
            'label': answer_index,
            'text': question
        }
        for i, choice in enumerate(choices):
            result[f'choice{i}'] = choice
        return result


    def __format(self, x: dict, question: str, choices: list, answer_key: str):
        """Format a given question"""
        answer_keys = list('abcde')
        answer_key = answer_key.lower().strip()
        assert answer_key in answer_keys, "Answer key is invalid."

        question = self.__clean_text(question)
        choices = [self.__clean_text(c) for c in choices]

        if self.ds_format == 'bert':
            formatted = self.__bert_format(question, choices, answer_key)
            for i in range(len(choices)):
                x[f'choice{i}'] = formatted[f'choice{i}']
        else:
            formatted = self.__t5_format(question, choices, answer_key)

        x["text"] = formatted['text']
        x["label"] = formatted['label']
        return x


    def __process_csqa(self, x):
        """Process and format a single example of CSQA dataset"""
        choices = x["choices"]["text"]
        answer_index = np.argmax([int(k == x["answerKey"].lower()) for k in list('abcde')])
        answer = choices.pop(answer_index)
        random.shuffle(choices)
        if self.force_4_choices:
            choices.pop()
        new_index = random.randint(0, 3)
        choices.insert(new_index, answer)
        new_answer_key = list('abcde')[new_index]
        return self.__format(x, x["question"], choices, new_answer_key)


    def load_csqa(self) -> Dataset:
        """Load CSQA dataset"""
        ds_dict = load_dataset("tau/commonsense_qa")
        dataset = DatasetDict(train=ds_dict["train"], test=ds_dict["validation"])
        return dataset.map(self.__process_csqa, remove_columns=dataset["train"].column_names)


    def __process_rs(self, x):
        """Process and format a single example of RiddleSense dataset"""
        choices = x["choices"]["text"]
        answer_index = np.argmax([int(k == x["answerKey"].lower()) for k in list('abcde')])
        answer = choices.pop(answer_index)
        random.shuffle(choices)
        if self.force_4_choices:
            choices.pop()
        new_index = random.randint(0, 3)
        choices.insert(new_index, answer)
        new_answer_key = list('abcde')[new_index]
        return self.__format(x, x["question"], choices, new_answer_key)


    def load_rs(self) -> Dataset:
        """Load RiddleSense dataset"""
        ds_dict = load_dataset("riddle_sense")
        dataset = DatasetDict(train=ds_dict["train"], test=ds_dict["validation"])
        return dataset.map(self.__process_rs, remove_columns=dataset["train"].column_names)


    def __process_bt(self, x):
        """Process and format a single example of BrainTeaser dataset"""
        choices = [str(c) for c in x["choice_list"]]
        answer_key = list('abcd')[x["label"]]
        return self.__format({}, x["question"], choices, answer_key)


    def load_bt5fold(self) -> Dataset:
        """Load 5-fold version of BrainTeaser dataset"""
        data = np.load('data/SP-train.npy', allow_pickle=True).tolist()
        k_fold = 5

        # Group similar questions
        question_groups = chunks(data, 3)
        random.seed(42)
        random.shuffle(question_groups)

        # Create k partitions
        partitions = [[] for i in range(k_fold)]
        next_partition = 0
        for q_group in question_groups:
            partitions[next_partition].extend(q_group)
            next_partition = (next_partition + 1) % k_fold

        # Create datasets
        partitions = [list(map(self.__process_bt, p)) for p in partitions]
        partitions = [Dataset.from_list(p) for p in partitions]
        datasets = []
        with suppress():
            for i in range(k_fold):
                train_partitions = [p for j,p in enumerate(partitions) if j != i]
                datasets.append(DatasetDict(
                    train=concatenate_datasets(train_partitions).shuffle(seed=i),
                    test=partitions[i]
                ))
        dataset = {f"fold{i}": p for i, p in enumerate(datasets)}
        dataset = DatasetDict(**dataset)
        return dataset


    def load_bt_fold0(self) -> Dataset:
        """Load the fold 0 of BrainTeaser dataset"""
        return self.load_bt5fold()["fold0"]


    def load_bt_fold1(self) -> Dataset:
        """Load the fold 1 of BrainTeaser dataset"""
        return self.load_bt5fold()["fold1"]


    def load_bt_fold2(self) -> Dataset:
        """Load the fold 2 of BrainTeaser dataset"""
        return self.load_bt5fold()["fold2"]


    def load_bt_fold3(self) -> Dataset:
        """Load the fold 3 of BrainTeaser dataset"""
        return self.load_bt5fold()["fold3"]


    def load_bt_fold4(self) -> Dataset:
        """Load the fold 4 of BrainTeaser dataset"""
        return self.load_bt5fold()["fold4"]


    def load_bt_test(self) -> Dataset:
        """Load the test set of BrainTeaser dataset"""
        data = np.load("data/SP_test.npy", allow_pickle=True).tolist()
        data_answer = np.load("data/SP_test_answer.npy", allow_pickle=True).tolist()
        test_questions = []
        for q, answer in zip(data, data_answer):
            answer_key = list('abcd')[int(answer[1])]
            test_questions.append(self.__format({}, q['question'], q['choice_list'], answer_key))
        return Dataset.from_list(test_questions)


    def load_bt_final(self) -> Dataset:
        """Load BrainTeaser dataset"""
        bts_fold0 = self.load_bt5fold()["fold0"]
        return DatasetDict(
            train=concatenate_datasets([bts_fold0["train"], bts_fold0["test"]]).shuffle(seed=42),
            test=self.load_bt_test()
        )


    def __process_swag(self, x):
        """Process and format a single example of SWAG dataset"""
        answer_key = list('abcd')[x['label_idx']]
        choices = [x['sent2'] + ' ' + x[f'ending{i}']  for i in range(4)]
        return self.__format(x, x['sent1'], choices, answer_key)


    def load_swag(self) -> Dataset:
        """Load SWAG dataset"""
        train_ds, test_ds = load_dataset('swag', split=['train', 'validation'])
        dataset = DatasetDict(train=train_ds, test=test_ds)
        dataset = dataset.rename_column("label", "label_idx")
        return dataset.map(self.__process_swag, remove_columns=dataset["train"].column_names)


    def __process_hellaswag(self, x):
        """Process and format a single example of HellaSWAG dataset"""
        answer_key = list('abcd')[int(x['label_idx'])]
        choices = [x['ctx_b'] + ' ' + x['endings'][i]  for i in range(4)]
        return self.__format(x, x['ctx_a'], choices, answer_key)


    def load_hellaswag(self) -> Dataset:
        """Load HellaSWAG dataset"""
        train_ds, test_ds = load_dataset('Rowan/hellaswag', split=['train', 'validation'])
        dataset = DatasetDict(train=train_ds, test=test_ds)
        dataset = dataset.rename_column("label", "label_idx")
        return dataset.map(self.__process_hellaswag, remove_columns=dataset["train"].column_names)


    def __process_siqa(self, x):
        """Process and format a single example of SIQA dataset"""
        choices = [x['answerA'], x['answerB'], x['answerC']]
        if self.force_4_choices:
            choices.append('dummy option')
        answer_key = list('abcd')[int(x['label_number']) - 1]
        return self.__format(x, f"{x['context']} {x['question']}", choices, answer_key)


    def load_siqa(self) -> Dataset:
        """Load SIQA dataset"""
        ds_dict = load_dataset("social_i_qa")
        dataset = DatasetDict(train=ds_dict["train"], test=ds_dict["validation"])
        dataset = dataset.rename_column("label", "label_number")
        return dataset.map(self.__process_siqa, remove_columns=dataset["train"].column_names)


    def __process_piqa(self, x):
        """Process and format a single example of PIQA dataset"""
        choices = [x['sol1'], x['sol2']]
        if self.force_4_choices:
            choices.append('dummy option')
            choices.append('dummy option')
        answer_key = list('abcd')[int(x['label_idx'])]
        return self.__format(x, x['goal'], choices, answer_key)


    def load_piqa(self) -> Dataset:
        """Load PIQA dataset"""
        ds_dict = load_dataset("piqa")
        dataset = DatasetDict(train=ds_dict["train"], test=ds_dict["validation"])
        dataset = dataset.rename_column("label", "label_idx")
        return dataset.map(self.__process_piqa, remove_columns=dataset["train"].column_names)


    def load_ds(self, ds_name: str) -> Dataset:
        """Load a dataset given its name"""
        fn_name = f"load_{ds_name}"
        fn = getattr(self, fn_name, None)
        if callable(fn):
            return fn()
        raise ValueError('The dataset name is invalid.')


    def load_combined_datasets(self, primary_ds: str, secondary_ds: str) -> Dataset:
        """
        Load the union of two datasets
        The test set of the primary dataset will be used as the final test set.
        """
        primary_ds = self.load_ds(primary_ds)
        return DatasetDict(
            train=interleave_datasets(
                [
                    primary_ds["train"],
                    self.load_ds(secondary_ds)["train"]
                ],
                probabilities=[0.5, 0.5],
                stopping_strategy="all_exhausted",
                seed=42,
            ),
            test=primary_ds["test"]
        )
