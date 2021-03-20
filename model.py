from torch import nn
from transformers import BertForTokenClassification
from dataset import TAGS

class Classifier(nn.Module):
    def __init__(self, model_name):
        super(Classifier, self).__init__()
        self.l1 = BertForTokenClassification.from_pretrained(model_name, num_labels=len(TAGS))

    def forward(self, ids, mask, labels):
        output_1= self.l1(ids, mask, labels = labels)
        return output_1
