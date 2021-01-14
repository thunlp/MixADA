# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange


import logging
import os
import random 

from transformers import is_tf_available
from transformers import DataProcessor, InputExample, InputFeatures
from sklearn.feature_extraction.text import TfidfVectorizer

if is_tf_available():
    import tensorflow as tf

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    XLMConfig,
    XLMForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
    XLMTokenizer,
    XLNetConfig,
    XLNetForSequenceClassification,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

from mixtext_model import MixText, SentMix, RobertaMixText, RobertaSentMix
from attackEval import load_custom_dataset, load_examples, ModelClassifier, get_attacker


import OpenAttack
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from OpenAttack.utils.dataset import Dataset, DataInstance
from OpenAttack.attackers import * 
import csv 

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import DataProcessor, InputExample, InputFeatures
from transformers import (
    BertConfig, 
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            BertConfig,
            XLNetConfig,
            XLMConfig,
            RobertaConfig,
            DistilBertConfig,
            AlbertConfig,
            XLMRobertaConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)





def train(args, train_dataset, model, tokenizer):
    global extracted_grads

    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    if args.mix_option == 1:
        logger.info("Random Mixup")
    else:
        logger.info("No Mixup")


    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    processor = processors[args.task_name]()
    attacker = get_attacker(args.attacker)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    ## Add Mixup in Batch
    epoch = 0
    for _ in train_iterator:
        epoch += 1 
        
        if epoch > 1 and args.iterative:
            ## augment the current train dataset with new batch of adversarial exampels generated by the currect model
            orig_data = load_custom_dataset(os.path.join(args.data_dir, "train.tsv"), all_data=True, number=args.num_adv) 
            clsf = ModelClassifier(tokenizer, model, args)
            attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf, progress_bar=True)
            adv_egs = attack_eval.eval(orig_data, visualize=False, return_examples=True)
            adv_examples = processor._create_examples(adv_egs, "adv_train")
            logger.info("Epoch: {}, Number of adversarial examples added to training: {}".format(epoch, len(adv_examples)))
            adv_dataset = convert_examples_dataset(args, adv_examples, tokenizer) 
            train_dataset = ConcatDataset([train_dataset, adv_dataset])

            ## start training on augmented data (we will shuffle the training data)
            # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

            logger.info("Current Num examples = %d", len(train_dataset))
    
        epoch_iterator = train_dataloader
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            
            ## normal training
            ## for now, just ignore token type ids
            input_ids = batch[0] #(bsz, len)
            attention_mask = batch[1]
            batch_size = input_ids.size(0)
            length = input_ids.size(1)
            labels = batch[3] #(bsz,)
            logits, outputs = model(input_ids, attention_mask) #(bsz, num_labels)
            # x_embeddings = outputs[2] # (bsz, len, dim)
            # x_embeddings.register_hook(save_grad("x_emb"))
            # logger.info("#outputs 1: " + str(len(outputs[-1])))
            L_ori = nn.CrossEntropyLoss()(logits.view(-1, args.num_labels), labels.view(-1))

            ## RandomMix
            if args.mix_option == 1:
                idx = torch.randperm(batch_size)
                input_ids_2 = input_ids[idx]
                labels_2 = labels[idx]
                attention_mask_2 = attention_mask[idx]
                ## convert the labels to one-hot
                labels = torch.zeros(batch_size, args.num_labels).to(args.device).scatter_(
                    1, labels.view(-1, 1), 1
                )
                labels_2 = torch.zeros(batch_size, args.num_labels).to(args.device).scatter_(
                    1, labels_2.view(-1, 1), 1
                )
                
                l = np.random.beta(args.alpha, args.alpha)
                # l = max(l, 1-l) ## not needed when only using labeled examples
                mixed_labels = l * labels + (1-l) * labels_2 

                mix_layer = np.random.choice(args.mix_layers_set, 1)[0]
                mix_layer = mix_layer - 1 
                
                logits, outputs = model(input_ids, attention_mask, input_ids_2, attention_mask_2, l, mix_layer)
                probs = torch.softmax(logits, dim=1) #(bsz, num_labels)
                L_mix = F.kl_div(probs.log(), mixed_labels, None, None, 'batchmean')

                loss = L_ori + L_mix 
           
            else:
                loss = L_ori 
            
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            tr_loss += loss.item()

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    # print(json.dumps({**logs, **{"step": global_step}}))
                    
                    logging.info("Global Step: "+str(global_step))
                    logging.info("Loss: "+str(loss_scalar))
        
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    ## save the final epoch only
    if args.local_rank in [-1, 0]:
        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, "final-checkpoint")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training                    
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", output_dir)

        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
        logger.info("Saving optimizer and scheduler states to %s", output_dir)

    

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", do_test=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True, do_test=do_test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        # logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                # if args.model_type != "distilbert":
                #     inputs["token_type_ids"] = (
                #         batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                #     )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                input_ids = batch[0]
                attention_mask = batch[1]
                labels = batch[3]
                logits, outputs = model(input_ids, attention_mask)
                # tmp_eval_loss, logits = outputs[:2]
                loss = nn.CrossEntropyLoss()(logits, labels)
                # _, predicted = torch.max(logits.data, 1)

                eval_loss += loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        cur_acc = result["acc"]
        results.update(result)

        # output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        # with open(output_eval_file, "w") as writer:
        #     # logger.info("***** Eval results {} *****".format(prefix))
        #     for key in sorted(result.keys()):
        #         # logger.info("  %s = %s", key, str(result[key]))
        #         writer.write("%s = %s\n" % (key, str(result[key])))
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

    return results, cur_acc


def load_and_cache_examples(args, task, tokenizer, evaluate=False, do_test=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    ## if we load features, sim_list will be None 
    if not evaluate:
        cache_type = "train"
    if evaluate and not do_test:
        cache_type = "dev"
    if evaluate and do_test:
        cache_type = "test"
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            cache_type,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache and args.second_data_dir is None:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if do_test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = (
                processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
            )  
        if (not evaluate) and (not do_test) and args.second_data_dir:
            logger.info("Augmenting dataset file at %s", args.second_data_dir)
            augment_examples = processor.get_train_examples(args.second_data_dir)
            augment_examples = augment_examples[ : int(args.adv_ratio * len(augment_examples))]
            examples = examples + augment_examples
            # random.shuffle(examples)
        if (not evaluate) and (not do_test) and args.third_data_dir:
            logger.info("Augmenting dataset file at %s", args.third_data_dir)
            augment_examples = processor.get_train_examples(args.third_data_dir)
            examples = examples + augment_examples
            # random.shuffle(examples)

        ## downsample the training set
        if not evaluate:
            # # examples = examples[:500]
            # if args.num_examples < len(examples):
            #     examples = examples[:args.num_examples]
            logger.info("#Train examples used : {}".format(str(len(examples))))
        
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset 

def convert_examples_dataset(args, examples, tokenizer):
    label_list = ["0", "1"]
    task = args.task_name
    output_mode = output_modes[task]

    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode=output_mode,
        pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
    )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset 


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    # parser.add_argument(
    #     "--test_file",
    #     default=None,
    #     type=str,
    #     help="The test file. Should contain the .tsv files (or other data files) for the task.",
    # )
    parser.add_argument(
        "--second_data_dir",
        default=None,
        type=str,
        required=False,
        help="second data dir, for data augmentation",
    )
    parser.add_argument(
        "--third_data_dir",
        default=None,
        type=str,
        required=False,
        help="third data dir, for data augmentation",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--mix_type",
        default=None,
        type=str,
        required=True,
        help="One of following three: nomix, tmix, atm, sentmix",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    # parser.add_argument(
    #     "--num_examples",
    #     default=9999999,
    #     type=int,
    #     help="Number of labeled examples."
    # )
    # parser.add_argument(
    #     "--do_at",
    #     default=False,
    #     action='store_true',
    #     help="do Adversarial Training or not"
    # )
    # parser.add_argument(
    #     "--epsilon",
    #     default=0,
    #     type=float,
    #     help="weight epsilon for AT loss"
    # )
    # parser.add_argument(
    #     "--do_vat",
    #     default=False,
    #     action='store_true',
    #     help="do Virtual Adversarial Training or not"
    # )
    # parser.add_argument(
    #     "--do_freelb",
    #     default=False,
    #     action='store_true',
    #     help="do FreeLB or not"
    # )
    parser.add_argument(
        "--mix-layers-set",
        nargs='+',
        default=[7,9,12],
        type=int,
        help="define mix layer set"
    )
    
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="alpha for beta distribution"
    )
    parser.add_argument(
        "--adv_ratio",
        type=float,
        default=1.0,
        help="proportion of adv examples to sample."
    )
    # parser.add_argument(
    #     "--gamma",
    #     type=float,
    #     default=1.0,
    #     help="gamma for L_mix loss"
    # )
    parser.add_argument(
        "--num_labels",
        type=int,
        default=2,
        help="Num of labels for classification. Needed to define Linear layer."
    )
    parser.add_argument(
        "--num_adv",
        type=int,
        default=500,
        help="Num of adversarial examples to add per epoch."
    )
    parser.add_argument(
        "--attacker",
        default="pwws",
        type=str,
        help="The attacker to use.",
    )
    parser.add_argument(
        "--iterative",
        default=False,
        type=bool,
        help="Whether to use iterative ADA or not.",
    )
    # parser.add_argument(
    #     "--mix_option",
    #     default=0,
    #     type=int,
    #     help="0: no mix at all; 1: random mix; 2: SimMix"
    # )
    # parser.add_argument(
    #     "--do_atm",
    #     default=0,
    #     type=int,
    #     help="Whether to do attentive mix-up."
    # )
    # parser.add_argument(
    #     "--sim_option",
    #     default=0,
    #     type=int,
    #     help="0: simMix; 1: DisSimMix"
    # )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    # parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    assert args.mix_type.lower() in ['nomix', 'tmix', 'sentmix'], "Mix Type Wrong!"
    logger.info("Using Model: " + args.mix_type)
    if args.mix_type.lower() == 'nomix':
        if args.model_type.lower() == 'roberta':
            model_class = RobertaMixText
        else:
            model_class = MixText 
        args.mix_option = 0 
    elif args.mix_type.lower() == 'tmix':
        if args.model_type.lower() == 'roberta':
            model_class = RobertaMixText
        else:
            model_class = MixText 
        args.mix_option = 1
    elif args.mix_type.lower() == 'sentmix':
        if args.model_type.lower() == 'roberta':
            model_class = RobertaSentMix
        else:
            model_class = SentMix
        args.mix_option = 1

    if args.do_train:
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=args.task_name,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        config.num_labels = args.num_labels

        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)


    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        max_acc = 0
        best_prefix = None
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            # logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            tokenizer = tokenizer_class.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
            model.to(args.device)
            result, cur_acc = evaluate(args, model, tokenizer, prefix=prefix)
            if cur_acc > max_acc:
                max_acc = cur_acc 
                best_model = model 
                best_tokenizer = tokenizer 
                best_prefix = prefix 
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
        logger.info("Best Dev Acc: "+str(max_acc))

        ## eval on test
        # result, cur_acc = evaluate(args, best_model, best_tokenizer, prefix=prefix, do_test=False)
        # logger.info("Best Dev Acc: "+str(max_acc))
        
        try:
            result, cur_acc = evaluate(args, best_model, best_tokenizer, prefix=best_prefix, do_test=True)
            logger.info("Test Acc: "+str(cur_acc))
        except:
            logger.info("Testing on test set skipped (test labels not provided.")

    return results


if __name__ == "__main__":
    main()
