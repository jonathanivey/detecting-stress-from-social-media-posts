import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnnutils

from transformers import BertModel, BertPreTrainedModel, RobertaModel, RobertaPreTrainedModel

from configs import const


class BertMultitask(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        """
        Our Multi-Alt model
        @param config: a BERTConfig, as used by huggingface
        @param kwargs: kwargs as generated by utils/model_utils.py
        """
        super().__init__(config)

        self.num_labels = kwargs["num_output_labels"]

        self.is_multilabel = kwargs["is_multilabel"]

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, i) for i in self.num_labels])

        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        essentially a copy-paste of the existing .from_pretrained() (e.g., BertForSequenceClassification)
        @param pretrained_model_name_or_path: used by huggingface to identify models
        @param model_args: huggingface model args
        @param kwargs: huggingface model kwargs
        @return: an initialized model using some pretrained checkpoint
        """
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        return model

    def forward(self, dataset_id, x, type_ids, attn_mask):
        """
        Forward pass of network
        @param dataset_id: index of the task we want (0 for stress, 1 for emotion, etc.)
        @param x: tensor, (batch_size, seq_length)
        @param type_ids: tensor, (batch_size, seq_length)
        @param attn_mask: tensor, (batch_size, seq_length)
        @return: logits, predictions for this single task
        """
        outputs = self.bert(x, token_type_ids=type_ids, attention_mask=attn_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifiers[dataset_id](pooled_output)

        return logits

    def get_loss(self, dataset_id, x, type_ids, attn_mask, loss_calc, golds):
        """
        Calculate the loss on one batch
        @param dataset_id: index of the task we want (0 for stress, 1 for emotion, etc.)
        @param x: tensor, (batch_size, seq_length)
        @param type_ids: tensor, (batch_size, seq_length)
        @param attn_mask: tensor, (batch_size, seq_length)
        @param loss_calc: an instance of our loss calculator (utils/loss_utils.py)
        @param golds: gold labels, tensor (batch_size, seq_length)
        @return: labels (predicted labels), loss (tensor, usually scalar)
        """
        logits = self(dataset_id, x, type_ids, attn_mask)

        if self.is_multilabel[dataset_id]:
            labels = (logits.sigmoid() > const.MULTILABEL_THRESHOLD) * 1
        else:
            labels = logits.argmax(dim=1)

        return labels, loss_calc.get_loss(logits, golds, dataset_id=dataset_id)

    def predict(self, dataset_id, x, type_ids, attn_mask):
        """
        Predict labels for one batch
        @param dataset_id: the ID of the task we want (0 for stress, 1 for emotion, etc.)
        @param x: tensor, (batch_size, seq_length)
        @param type_ids: tensor, (batch_size, seq_length)
        @param attn_mask: tensor, (batch_size, seq_length)
        @return: preds (predicted labels)
        """
        preds = self(dataset_id, x, type_ids, attn_mask)

        if self.is_multilabel[dataset_id]:
            preds = (preds.sigmoid() > const.MULTILABEL_THRESHOLD) * 1
        else:
            preds = preds.argmax(dim=1)

        return preds


class RobertaMultitask(RobertaPreTrainedModel):
    def __init__(self, config, **kwargs):
        """
        Our Multi-Alt model
        @param config: a BERTConfig, as used by huggingface
        @param kwargs: kwargs as generated by utils/model_utils.py
        """
        super().__init__(config)

        self.num_labels = kwargs["num_output_labels"]

        self.is_multilabel = kwargs["is_multilabel"]

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, i) for i in self.num_labels])

        self.init_weights()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        essentially a copy-paste of the existing .from_pretrained() (e.g., BertForSequenceClassification)
        @param pretrained_model_name_or_path: used by huggingface to identify models
        @param model_args: huggingface model args
        @param kwargs: huggingface model kwargs
        @return: an initialized model using some pretrained checkpoint
        """
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        return model

    def forward(self, dataset_id, x, type_ids, attn_mask):
        """
        Forward pass of network
        @param dataset_id: index of the task we want (0 for stress, 1 for emotion, etc.)
        @param x: tensor, (batch_size, seq_length)
        @param type_ids: tensor, (batch_size, seq_length)
        @param attn_mask: tensor, (batch_size, seq_length)
        @return: logits, predictions for this single task
        """
        outputs = self.roberta(x, attention_mask=attn_mask)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifiers[dataset_id](pooled_output)

        return logits

    def get_loss(self, dataset_id, x, type_ids, attn_mask, loss_calc, golds):
        """
        Calculate the loss on one batch
        @param dataset_id: index of the task we want (0 for stress, 1 for emotion, etc.)
        @param x: tensor, (batch_size, seq_length)
        @param type_ids: tensor, (batch_size, seq_length)
        @param attn_mask: tensor, (batch_size, seq_length)
        @param loss_calc: an instance of our loss calculator (utils/loss_utils.py)
        @param golds: gold labels, tensor (batch_size, seq_length)
        @return: labels (predicted labels), loss (tensor, usually scalar)
        """
        logits = self(dataset_id, x, type_ids, attn_mask)

        if self.is_multilabel[dataset_id]:
            labels = (logits.sigmoid() > const.MULTILABEL_THRESHOLD) * 1
        else:
            labels = logits.argmax(dim=1)

        return labels, loss_calc.get_loss(logits, golds, dataset_id=dataset_id)

    def predict(self, dataset_id, x, type_ids, attn_mask):
        """
        Predict labels for one batch
        @param dataset_id: the ID of the task we want (0 for stress, 1 for emotion, etc.)
        @param x: tensor, (batch_size, seq_length)
        @param type_ids: tensor, (batch_size, seq_length)
        @param attn_mask: tensor, (batch_size, seq_length)
        @return: preds (predicted labels)
        """
        preds = self(dataset_id, x, type_ids, attn_mask)

        if self.is_multilabel[dataset_id]:
            preds = (preds.sigmoid() > const.MULTILABEL_THRESHOLD) * 1
        else:
            preds = preds.argmax(dim=1)

        return preds