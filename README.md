## BERT Named entity recognition task
NER model created by fine-tuning BERT using PyTorch and Transformers library

### Benefits of this implementation
* Most recent pip packages are being used
* Code is clearly separated into classes
* Although there were empty tags added to each sentence, at validation these tags are removed to provide more accurate score
* Both micro and macro F1 scores are being shown

### Disadvantages of this implementation
* Model was trained only once overnight on CPU due to time limit of the task
* Therefore, no experiments were done in terms of e.g. adding extra layers, changing the learning rate, etc.

### Dataset
The dataset used is CONLL2003 where only NER tags are being used (omitting POS and IOB)

