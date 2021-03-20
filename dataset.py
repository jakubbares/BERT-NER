import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
TAGS = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O", "None"]
MAX_LENGTH = 150
TRAIN_BATCH_SIZE = 20
TEST_BATCH_SIZE = 1


class FileLoader:
    def __init__(self):
        self.folder_path = "CONLL2003"

    def load_data(self, type):
        try:
            self.df = self.load_dataframe(type)
        except:
            self.df = self.load_data_from_file(type)
            self.save_dataframe(type)
        return self.df

    def process_line(self, text):
        word, pos, iob, tag = text.rstrip('\n').split(" ")
        return word, pos, iob, tag, TAGS.index(tag)

    def load_data_from_file(self, type):
        data = []
        path = os.path.join(self.folder_path, f"{type}.txt")
        file = open(path, "r")
        sentence_index = 0
        for line in file.readlines():
            if line == '\n':
                sentence_index += 1
            else:
                data.append([sentence_index, *self.process_line(line)])
        return pd.DataFrame(data, columns=["sentence", "word", "pos", "iob", "tag", "label"])

    def save_dataframe(self, type):
        path = os.path.join(self.folder_path, f"{type}_df.csv")
        self.df.to_csv(path)

    def load_dataframe(self, type):
        path = os.path.join(self.folder_path, f"{type}_df.csv")
        return pd.read_csv(path)

class NERDataLoader:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.file_loader = FileLoader()
        self.grouped = {}

    @property
    def max_length(self):
        return max([labels for type in ["train", "test"] for labels in self.grouped[type]['label'].str.len()])

    def load_data(self, type):
        df = self.file_loader.load_data(type)
        self.grouped[type] = df.groupby(['sentence'], as_index=False)['word','label'].agg(lambda x: list(x))
        return self.grouped[type]["word"], self.grouped[type]["label"]

    def train(self):
        train_sentences, train_labels = self.load_data("train")
        training_set = NERDataset(self.tokenizer, train_sentences, train_labels, MAX_LENGTH)
        train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
        training_loader = DataLoader(training_set, **train_params)
        return training_loader

    def test(self):
        test_sentences, test_labels = self.load_data("test")
        testing_set = NERDataset(self.tokenizer, test_sentences, test_labels, MAX_LENGTH)
        test_params = {'batch_size': TEST_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
        testing_loader = DataLoader(testing_set, **test_params)
        return testing_loader


class NERDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        input_data = inputs['input_ids']
        mask_data = inputs['attention_mask']
        label = self.labels[index]
        label.extend([9]*MAX_LENGTH)
        label=label[:MAX_LENGTH]

        return {
            'ids': torch.tensor(input_data, dtype=torch.long),
            'mask': torch.tensor(mask_data, dtype=torch.long),
            'tags': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return self.len
