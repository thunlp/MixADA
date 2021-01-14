import torch
import torch.nn as nn
from transformers import *
from transformers.modeling_bert import BertEmbeddings, BertPooler, BertLayer
from transformers.modeling_roberta import RobertaEmbeddings, RobertaModel, RobertaClassificationHead
from transformers.configuration_roberta import ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP, RobertaConfig

## if using BERT, just switch to BertModel4Mix in the MixText model
## and change name to self.bert

class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        # self.output_attentions = config.output_attentions
        # self.output_hidden_states = config.output_hidden_states
        self.output_attentions = False
        self.output_hidden_states = True 
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, l=None, mix_layer=1000, attention_mask=None, attention_mask2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        ## mix_layer == -1: mixup at embedding layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1-l)*hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = l * hidden_states + (1-l)*hidden_states2 
                    attention_mask = attention_mask.long() | attention_mask2.long()
                    ## sentMix: (bsz, len, hid)
                    # hidden_states[:, 0, :] = l * hidden_states[:, 0, :] + (1-l)*hidden_states2[:, 0, :] 

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        # print (len(outputs))
        # print (len(outputs[1])) ##hidden states: 13
        return outputs

class BertModel4Mix(BertPreTrainedModel): 

    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)
        # self.embeddings = BertEmbeddings(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask, input_ids2=None, attention_mask2=None, l=None, mix_layer=1000,  
        token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):

        input_shape = input_ids.size() 
        device = input_ids.device 

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:
            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)
            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (
                1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

        if input_ids2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, l, mix_layer,
                                           extended_attention_mask, extended_attention_mask2, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output, embedding_output) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs

class RobertaModel4Mix(BertModel4Mix):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel4Mix, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

class MixText(BertPreTrainedModel):
    def __init__(self, config):
        super(MixText, self).__init__(config)

        self.num_labels = config.num_labels 
        self.bert = BertModel4Mix(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, x, attention_mask, x2=None, attention_mask2=None, l=None, mix_layer=1000, inputs_embeds=None):

        if x2 is not None:
            outputs = self.bert(x, attention_mask, x2, attention_mask, l, mix_layer, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]

        else:
            outputs = self.bert(x, attention_mask, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]


        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output)

        return logits, outputs 

class RobertaMixText(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaMixText, self).__init__(config)

        self.num_labels = config.num_labels 
        self.roberta = RobertaModel4Mix(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(self, x, attention_mask, x2=None, attention_mask2=None, l=None, mix_layer=1000, inputs_embeds=None):

        if x2 is not None:
            outputs = self.roberta(x, attention_mask, x2, attention_mask, l, mix_layer, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            # pooled_output = outputs[1]

        else:
            outputs = self.roberta(x, attention_mask, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            # pooled_output = outputs[1]


        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        return logits, outputs 




class BertEncoder4SentMix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4SentMix, self).__init__()
        # self.output_attentions = config.output_attentions
        # self.output_hidden_states = config.output_hidden_states
        self.output_attentions = False
        self.output_hidden_states = True 
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, l=None, mix_layer=1000, attention_mask=None, attention_mask2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        ## mix_layer == -1: mixup at embedding layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1-l)*hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if hidden_states2 is not None:
                    # hidden_states = l * hidden_states + (1-l)*hidden_states2 
                    # attention_mask = attention_mask.long() | attention_mask2.long()
                    # sentMix: (bsz, len, hid)
                    hidden_states[:, 0, :] = l * hidden_states[:, 0, :] + (1-l)*hidden_states2[:, 0, :] 

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        # print (len(outputs))
        # print (len(outputs[1])) ##hidden states: 13
        return outputs

class BertModel4SentMix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4SentMix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4SentMix(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask, input_ids2=None, attention_mask2=None, l=None, mix_layer=1000,  
        token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):

        input_shape = input_ids.size() 
        device = input_ids.device 

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:
            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)
            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (
                1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, l, mix_layer,
                                           extended_attention_mask, extended_attention_mask2, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs

class RobertaModel4SentMix(BertModel4SentMix):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel4SentMix, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


class SentMix(BertPreTrainedModel):
    def __init__(self, config):
        super(SentMix, self).__init__(config)

        self.num_labels = config.num_labels 
        self.bert = BertModel4SentMix(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, x, attention_mask, x2=None, attention_mask2=None, l=None, mix_layer=1000, inputs_embeds=None):

        if x2 is not None:
            outputs = self.bert(x, attention_mask, x2, attention_mask, l, mix_layer, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]

        else:
            outputs = self.bert(x, attention_mask, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            pooled_output = outputs[1]


        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # sequence_output = outputs[0]
        # logits = self.classifier(sequence_output)

        return logits, outputs 


class RobertaSentMix(BertPreTrainedModel):
    def __init__(self, config):
        super(RobertaSentMix, self).__init__(config)

        self.num_labels = config.num_labels 
        self.roberta = RobertaModel4SentMix(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.classifier = RobertaClassificationHead(config)

        self.init_weights()

    def forward(self, x, attention_mask, x2=None, attention_mask2=None, l=None, mix_layer=1000, inputs_embeds=None):

        if x2 is not None:
            outputs = self.roberta(x, attention_mask, x2, attention_mask, l, mix_layer, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            # pooled_output = outputs[1]

        else:
            outputs = self.roberta(x, attention_mask, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            # pooled_output = outputs[1]


        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        return logits, outputs 



















## a model for probing purpose only 
## Use MixText model for the actual TMix
class BertEncoder4TokenMix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4TokenMix, self).__init__()
        # self.output_attentions = config.output_attentions
        # self.output_hidden_states = config.output_hidden_states
        self.output_attentions = False
        self.output_hidden_states = True 
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, l=None, mix_layer=1000, attention_mask=None, attention_mask2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        ## mix_layer == -1: mixup at embedding layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1-l)*hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if hidden_states2 is not None:
                    # hidden_states = l * hidden_states + (1-l)*hidden_states2 
                    # attention_mask = attention_mask.long() | attention_mask2.long()
                    # sentMix: (bsz, len, hid)
                    hidden_states[:, 1:, :] = l * hidden_states[:, 1:, :] + (1-l)*hidden_states2[:, 1:, :] 

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        # print (len(outputs))
        # print (len(outputs[1])) ##hidden states: 13
        return outputs

class BertModel4TokenMix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4TokenMix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4TokenMix(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask, input_ids2=None, attention_mask2=None, l=None, mix_layer=1000,  
        token_type_ids=None, position_ids=None, head_mask=None):

        input_shape = input_ids.size() 
        device = input_ids.device 

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:
            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)
            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (
                1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, l, mix_layer,
                                           extended_attention_mask, extended_attention_mask2, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs

class RobertaModel4TokenMix(BertModel4TokenMix):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel4TokenMix, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

class TokenMix(BertPreTrainedModel):
    def __init__(self, config):
        super(TokenMix, self).__init__(config)

        self.num_labels = config.num_labels 
        self.roberta = RobertaModel4TokenMix(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.classifier = RobertaClassificationHead(config)

        # self.init_weights()

    def forward(self, x, attention_mask, x2=None, attention_mask2=None, l=None, mix_layer=1000, inputs_embeds=None):

        if x2 is not None:
            outputs = self.roberta(x, attention_mask, x2, attention_mask, l, mix_layer, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            # pooled_output = outputs[1]

        else:
            outputs = self.roberta(x, attention_mask, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            # pooled_output = outputs[1]


        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        return logits, outputs 




class BertEncoderATM(nn.Module):
    def __init__(self, config):
        super(BertEncoderATM, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.map_linear = nn.Linear(config.hidden_size, config.hidden_size) 
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, l=None, mix_layer=1000, attention_mask=None, attention_mask2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        ## mix_layer == -1: mixup at embedding layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1-l)*hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                ## add attention 
                ## TODO: masked softmax
                if hidden_states2 is not None:
                    map_hidden_states = self.map_linear(hidden_states) #(bsz, len1, hid)
                    # map_hidden_states = hidden_states
                    trans_hidden_states = map_hidden_states.bmm(torch.transpose(hidden_states2, 1, 2)) #(bsz, len1, len2)
                    attn = nn.functional.softmax(trans_hidden_states, dim=-1) #(bsz, len1, len2)
                    attn_hidden_states2 = attn.bmm(hidden_states2) #(bsz, len1, h)
                    hidden_states = l * hidden_states + (1-l)*attn_hidden_states2 
                    attention_mask = attention_mask.long() | attention_mask2.long()

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs

class BertModelATM(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModelATM, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoderATM(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, attention_mask, input_ids2=None, attention_mask2=None, l=None, mix_layer=1000,  
        token_type_ids=None, position_ids=None, head_mask=None):

        input_shape = input_ids.size() 
        device = input_ids.device 

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2, device=device)
            attention_mask = torch.ones_like(input_ids, device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long, device=device)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:
            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)
            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (
                1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, l, mix_layer,
                                           extended_attention_mask, extended_attention_mask2, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs

class RobertaModelATM(BertModelATM):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModelATM, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

class ATM(BertPreTrainedModel):
    def __init__(self, config):
        super(ATM, self).__init__(config)

        self.num_labels = config.num_labels 
        self.roberta = RobertaModelATM(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.classifier = RobertaClassificationHead(config)

        # self.init_weights()

    def forward(self, x, attention_mask, x2=None, attention_mask2=None, l=None, mix_layer=1000, inputs_embeds=None):

        if x2 is not None:
            outputs = self.roberta(x, attention_mask, x2, attention_mask, l, mix_layer, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            # pooled_output = outputs[1]

        else:
            outputs = self.roberta(x, attention_mask, inputs_embeds=inputs_embeds)

            # pooled_output = torch.mean(outputs[0], 1)
            # pooled_output = outputs[1]


        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        return logits, outputs 