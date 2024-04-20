import torch
import pandas as pd
import numpy as np
from torch.nn import BCEWithLogitsLoss
from transformers import LongformerTokenizerFast, LongformerModel, LongformerConfig, Trainer, TrainingArguments, EvalPrediction, AutoTokenizer, AutoModel
from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel, LongformerClassificationHead
from torch.utils.data import Dataset, DataLoader
import wandb
import random
import tensorflow as tf
import os
from transformers import EarlyStoppingCallback
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

WANDB_PROJECT='icd_Clinical_longfomer_plain'
WANDB_LOG_MODEL=True

# read the dataframe
train = pd.read_csv('../data/mimic3/5/train_data_5_level_1_bkp.csv')
train['Chunk1'] = train['Chunk1'].fillna('')
train['Chunk2'] = train['Chunk2'].fillna('')
train['Chunk3'] = train['Chunk3'].fillna('')
train['Chunk4'] = train['Chunk4'].fillna('')
train['Chunk5'] = train['Chunk5'].fillna('')
train['Chunk6'] = train['Chunk6'].fillna('')
train['Chunk7'] = train['Chunk7'].fillna('')
train['Chunk8'] = train['Chunk8'].fillna('')
train['Chunk9'] = train['Chunk9'].fillna('')
train['Chunk10'] = train['Chunk10'].fillna('')

# Concatenate the 'Chunk' columns into a single 'CombinedChunk' column
train['CombinedChunk'] = train['Chunk1'] + train['Chunk2'] + train['Chunk3'] + train['Chunk4'] + train['Chunk5'] + train['Chunk6'] + train['Chunk7'] + train['Chunk8'] + train['Chunk9'] + train['Chunk10']
train = train.drop(['Chunk1', 'Chunk2', 'Chunk3', 'Chunk4', 'Chunk5', 'Chunk6', 'Chunk7', 'Chunk8', 'Chunk9', 'Chunk10'], axis=1)
train
column_order = ['hadm_id', 'CombinedChunk', '38.93', '401.9', '414.01', '427.31', '428']

# Reorder the columns in the DataFrame based on the specified order
train = train[column_order]
train['labels'] = train[train.columns[2:]].values.tolist()
train = train.drop(['38.93', '401.9', '414.01', '427.31', '428'], axis=1)
print(train)


# Read the 'dev' DataFrame from a CSV file (replace '/path/to/dev_data.csv' with the actual path)
dev = pd.read_csv('../data/mimic3/5/dev_data_5_level_1_bkp.csv')

# Fill NaN values in 'Chunk' columns with empty strings
dev['Chunk1'] = dev['Chunk1'].fillna('')
dev['Chunk2'] = dev['Chunk2'].fillna('')
dev['Chunk3'] = dev['Chunk3'].fillna('')
dev['Chunk4'] = dev['Chunk4'].fillna('')
dev['Chunk5'] = dev['Chunk5'].fillna('')
dev['Chunk6'] = dev['Chunk6'].fillna('')
dev['Chunk7'] = dev['Chunk7'].fillna('')
dev['Chunk8'] = dev['Chunk8'].fillna('')
dev['Chunk9'] = dev['Chunk9'].fillna('')
dev['Chunk10'] = dev['Chunk10'].fillna('')

# Concatenate the 'Chunk' columns into a single 'CombinedChunk' column
dev['CombinedChunk'] = dev['Chunk1'] + dev['Chunk2'] + dev['Chunk3'] + dev['Chunk4'] + dev['Chunk5'] + dev['Chunk6'] + dev['Chunk7'] + dev['Chunk8'] + dev['Chunk9'] + dev['Chunk10']
dev = dev.drop(['Chunk1', 'Chunk2', 'Chunk3', 'Chunk4', 'Chunk5', 'Chunk6', 'Chunk7', 'Chunk8', 'Chunk9', 'Chunk10'], axis=1)

# Define the desired column order for 'dev' (similar to 'train' and 'test')
column_order = ['hadm_id', 'CombinedChunk', '38.93', '401.9', '414.01', '427.31', '428']

# Reorder the columns in the 'dev' DataFrame based on the specified order
dev = dev[column_order]

# Create a 'labels' column in 'dev' (if needed)
dev['labels'] = dev[dev.columns[2:]].values.tolist()
dev = dev.drop(['38.93', '401.9', '414.01', '427.31', '428'], axis=1)
print(dev)
# Now, you have performed the same operations on the 'dev' DataFrame as you did for 'train' and 'test'
# Read the 'test' DataFrame from a CSV file (replace '/path/to/test_data.csv' with the actual path)
test = pd.read_csv('../data/mimic3/5/test_data_5_level_1_bkp.csv')

# Fill NaN values in 'Chunk' columns with empty strings
test['Chunk1'] = test['Chunk1'].fillna('')
test['Chunk2'] = test['Chunk2'].fillna('')
test['Chunk3'] = test['Chunk3'].fillna('')
test['Chunk4'] = test['Chunk4'].fillna('')
test['Chunk5'] = test['Chunk5'].fillna('')
test['Chunk6'] = test['Chunk6'].fillna('')
test['Chunk7'] = test['Chunk7'].fillna('')
test['Chunk8'] = test['Chunk8'].fillna('')
test['Chunk9'] = test['Chunk9'].fillna('')
test['Chunk10'] = test['Chunk10'].fillna('')

# Concatenate the 'Chunk' columns into a single 'CombinedChunk' column
test['CombinedChunk'] = test['Chunk1'] + test['Chunk2'] + test['Chunk3'] + test['Chunk4'] + test['Chunk5'] + test['Chunk6'] + test['Chunk7'] + test['Chunk8'] + test['Chunk9'] + test['Chunk10']
test = test.drop(['Chunk1', 'Chunk2', 'Chunk3', 'Chunk4', 'Chunk5', 'Chunk6', 'Chunk7', 'Chunk8', 'Chunk9', 'Chunk10'], axis=1)

# Define the desired column order for 'test' (similar to 'train')
column_order = ['hadm_id', 'CombinedChunk', '38.93', '401.9', '414.01', '427.31', '428']

# Reorder the columns in the 'test' DataFrame based on the specified order
test = test[column_order]

# Create a 'labels' column in 'test' (if needed)
test['labels'] = test[test.columns[2:]].values.tolist()
test = test.drop(['38.93', '401.9', '414.01', '427.31', '428'], axis=1)
print(test)
# Now, you have performed the same operations on the 'test' DataFrame as you did for 'train'


class LongformerForMultiLabelICDClassification(LongformerPreTrainedModel):
    """
    We instantiate a class of LongFormer adapted for a multilabel classification task.
    This instance takes the pooled output of the LongFormer based model and passes it through a classification head. We replace the traditional Cross Entropy loss with a BCE loss that generate probabilities for all the labels that we feed into the model.
    """

    def __init__(self, config):
        super(LongformerForMultiLabelICDClassification, self).__init__(config)
        self.num_labels = config.num_labels
        # self.transformer_layer = AutoModel.from_pretrained("pretrained/ClinicalplusXLNet/")
        self.longformer = LongformerModel(config)
        self.classifier = LongformerClassificationHead(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None,
                token_type_ids=None, position_ids=None, inputs_embeds=None,
                labels=None):

        # create global attention on sequence, and a global attention token on the `s` token
        # the equivalent of the CLS token on BERT models. This is taken care of by HuggingFace
        # on the LongformerForSequenceClassification class
        if global_attention_mask is None:
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
        #transformer_output = self.transformer_layer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        # pass arguments to longformer model
        #transformer_output = tuple(torch.tensor(tf.convert_to_tensor(hs)).detach().numpy() for hs in transformer_output.hidden_states)
        outputs = self.longformer(
            input_ids = input_ids,
            attention_mask = attention_mask,
            global_attention_mask = global_attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids)

        # if specified the model can return a dict where each key corresponds to the output of a
        # LongformerPooler output class. In this case we take the last hidden state of the sequence
        # which will have the shape (batch_size, sequence_length, hidden_size).
        sequence_output = outputs['last_hidden_state']

        # pass the hidden states through the classifier to obtain thee logits
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.view(-1, self.num_labels))
            #outputs = (loss,) + outputs
            outputs = (loss,) + outputs


        return outputs
    
class Data_Processing(object):
    def __init__(self, tokenizer, id_column, text_column, label_column):

        # define the text column from the dataframe
        self.text_column = text_column.tolist()

        # define the label column and transform it to list
        self.label_column = label_column.tolist()

        # define the id column and transform it to list
        self.id_column = id_column.tolist()


# iter method to get each element at the time and tokenize it using bert
    def __getitem__(self, index):
        comment_text = str(self.text_column[index])
        comment_text = " ".join(comment_text.split())
        # encode the sequence and add padding
        inputs = tokenizer.encode_plus(comment_text,
                                       add_special_tokens = True,
                                       max_length= 512,
                                       padding = 'max_length',
                                       return_attention_mask = True,
                                       truncation = True,
                                       return_tensors='pt')
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        labels_ = torch.tensor(self.label_column[index], dtype=torch.float)
        id_ = self.id_column[index]
        return {'input_ids':input_ids[0], 'attention_mask':attention_mask[0],
                'labels':labels_, 'id_':id_}

    def __len__(self):
        return len(self.text_column)

batch_size = 2
# create a class to process the training and test data
tokenizer = AutoTokenizer.from_pretrained('yikuan8/Clinical-Longformer',
                                                    padding = 'max_length',
                                                    truncation=False,
                                                    max_length = 4096,
                                                    padding_side="right")
training_data = Data_Processing(tokenizer,
                                train['hadm_id'],
                                train['CombinedChunk'],
                                train['labels'])
dev_data = Data_Processing(tokenizer,
                             dev['hadm_id'],
                             dev['CombinedChunk'],
                             dev['labels'])
test_data =  Data_Processing(tokenizer,
                             test['hadm_id'],
                             test['CombinedChunk'],
                             test['labels'])

# use the dataloaders class to load the data
dataloaders_dict = {'train': DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=4),
                    'val': DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
                   }

dataset_sizes = {'train':len(training_data),
                 'val':len(test_data)
                }

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def model_init():
    model = LongformerForMultiLabelICDClassification.from_pretrained('yikuan8/Clinical-Longformer',
                                                  gradient_checkpointing=False,
                                                  attention_window = 32,
                                                  num_labels = 5,
                                                  return_dict=True)
    return model
    
# method
sweep_config = {
    'method': 'random'
}


# hyperparameters
parameters_dict = {
    'epochs': {
        'value': 1
        },
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 5e-5,
        'max': 5e-3
    },
    'weight_decay': {
        'values': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    },
}


sweep_config['parameters'] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project='icd_Clinical_longfomer_plain')


# model = LongformerForMultiLabelICDClassification.from_pretrained('yikuan8/Clinical-Longformer',
#                                                   gradient_checkpointing=False,
#                                                   attention_window = 32,
#                                                   num_labels = 5,
#                                                   return_dict=True)
# model.to(torch.device("cuda:0"))

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def multi_label_metrics(predictions, labels):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_true = labels
    y_pred[np.where(probs >= 0.5)] = 1
    f1_scores_per_label = f1_score(y_true, y_pred, average=None)
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # define dictionary of metrics to return
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy,
               'f1_scores_per_label': dict(zip(range(len(f1_scores_per_label)), f1_scores_per_label))}
    return metrics

# EvalPrediction class to obtain prediction labels
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

def train(config=None):
  with wandb.init(config=config):
    # set sweep configuration
    config = wandb.config
    # define the training arguments
    training_args = TrainingArguments(
        output_dir = '../model/new',
        num_train_epochs = 1,
        max_steps = 100,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 64,
        per_device_eval_batch_size= 2,
        evaluation_strategy = "steps",
        disable_tqdm = False,
        load_best_model_at_end=True,
        warmup_steps = 2000,
        learning_rate = config.learning_rate,
        weight_decay=config.weight_decay,
        logging_steps = 8,
        fp16 = False,
        logging_dir='/logs',
        dataloader_num_workers = 0,
        run_name = 'longformer_multilabel_paper_trainer_2e5',
        report_to = 'wandb',
        metric_for_best_model = 'eval_loss',
        greater_is_better=False
    )
    # instantiate the trainer class and check for available devices
    trainer = Trainer(
        # model=model,
        model_init = model_init,
        args=training_args,
        train_dataset=training_data,
        eval_dataset=dev_data,
        compute_metrics = compute_metrics,
        # callbacks = [EarlyStoppingCallback(early_stopping_patience=3, )],
        callbacks=[EarlyStoppingCallback(3, 0.0)]
        #data_collator = Data_Processing(),
    )

    trainer.train()
    
wandb.agent(sweep_id, train, count=20)

# test_predictions = trainer.predict(test_dataset=test_data)
# test_metrics = compute_metrics(test_predictions)
# f1_scores_per_label = test_metrics['f1_scores_per_label']

# print(test_predictions)

# print("Individual F1 scores")
# for label_idx, f1_score_label in f1_scores_per_label.items():
#     print(f"F1-score for label {label_idx}: {f1_score_label}")
