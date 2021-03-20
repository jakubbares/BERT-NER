import torch
from torch import optim
from model import Classifier
from dataset import NERDataLoader, TAGS
from sklearn.metrics import  f1_score
import numpy as np
from transformers import BertTokenizer
EPOCHS = 5
LEARNING_RATE = 4e-05

class ModelRunner:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Device is {self.device}")
        model_name = 'bert-base-cased'
        self.model_path = 'NER_CONLL2003.pt'
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = Classifier(model_name)
        self.loader = NERDataLoader(tokenizer=self.tokenizer)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=LEARNING_RATE)

    def train_and_save(self):
        for epoch in range(EPOCHS):
            self.train_epoch(epoch)
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        loaded_state = torch.load(self.model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(loaded_state)

    def train_epoch(self, epoch):
        training_loader = self.loader.train()
        self.model.train()
        for _, data in enumerate(training_loader, 0):
            input_data = data['ids'].to(self.device, dtype = torch.long)
            mask_data = data['mask'].to(self.device, dtype = torch.long)
            labels = data['tags'].to(self.device, dtype = torch.long)

            loss = self.model(input_data, mask_data, labels=labels)[0]
            if _%10==0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self):
        testing_loader = self.loader.test()
        self.model.eval()
        eval_loss = 0; eval_accuracy = 0
        predictions , true_labels = [], []
        nb_eval_steps, nb_eval_examples = 0, 0
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                input_data = data['ids'].to(self.device, dtype = torch.long)
                mask_data = data['mask'].to(self.device, dtype = torch.long)
                labels = data['tags'].to(self.device, dtype = torch.long)

                output = self.model(input_data, mask_data, labels=labels)
                loss, logits = output[:2]
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.append(label_ids)
                accuracy = self.flat_accuracy(logits, label_ids)
                eval_loss += loss.mean().item()
                eval_accuracy += accuracy
                nb_eval_examples += input_data.size(0)
                nb_eval_steps += 1
            eval_loss = eval_loss/nb_eval_steps
            print("Validation loss: {}".format(eval_loss))
            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            pred_tags = [TAGS[p_i] for p in predictions for p_i in p]
            valid_tags = [TAGS[l_ii] for l in true_labels for l_i in l for l_ii in  l_i]
            clean_pred_tags, clean_valid_tags = self.remove_none_tags(pred_tags, valid_tags)
            print("Micro F1-Score: {}".format(f1_score(clean_pred_tags, clean_valid_tags, average='micro')))
            print("Macro F1-Score: {}".format(f1_score(clean_pred_tags, clean_valid_tags, average='macro')))

    def remove_none_tags(self, pred_tags, valid_tags):
        return zip(*[(pred, valid) for pred, valid in zip(pred_tags, valid_tags) if "None" != valid])

    def flat_accuracy(self, preds, labels):
        flat_preds = np.argmax(preds, axis=2).flatten()
        flat_labels = labels.flatten()
        return np.sum(flat_preds == flat_labels)/len(flat_labels)

