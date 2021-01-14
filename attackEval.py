'''
This example code shows how to use the genetic algorithm-based attack model to attack a customized sentiment analysis model.
'''
import OpenAttack
import numpy as np
import torch 
import logging 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from OpenAttack.utils.dataset import Dataset, DataInstance
from OpenAttack.attackers import * 
import csv 

import argparse 
from mixtext_model import MixText, ATM, SentMix, TokenMix, RobertaMixText
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import DataProcessor, InputExample, InputFeatures, is_tf_available
from transformers import glue_compute_metrics as compute_metrics
# from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors
from transformers import (
    BertConfig, 
    BertForSequenceClassification,
    BertTokenizer,
    RobertaConfig,
    RobertaTokenizer,
)
import random 
if is_tf_available():
    import tensorflow as tf

random.seed(20) ## seed fixed in this file for fair comparisons
num_labels = 2


logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)



class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir, data_file=None):
        if not data_file:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        else:
            return self._create_examples(self._read_tsv(os.path.join(data_dir, data_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            if len(line) < 2:
                continue 
            text_a = line[0]
            label = line[1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        # if set_type == "train":
        #     random.shuffle(examples)
        # print ("Examples are shuffled")
        return examples
    
    def _create_examples_adv(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            text_a = line
            label = "0" ## label here doesn't matter
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        
        return examples

processors["sst-2"] = Sst2Processor

# over-ridden here to avoid the annoying logging 
def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if (ex_index+1) % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len(examples)))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        
        
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label.strip()]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        #     logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features



## used to load custom dataset 
def load_custom_dataset(test_file, all_data=False, number=None):
    # all_data: to return full dataset or not
    label_list = ["0", "1"]
    label_map = {label: i for i, label in enumerate(label_list)}
    with open(test_file, "r", encoding="utf-8-sig") as f:
        lines = list(csv.reader(f, delimiter="\t"))
    dataset = []
    for (i, line) in enumerate(lines):
        text = line[0].lower()
        label = label_map[line[1]]
        dataset.append(
            DataInstance(
                x = text,
                y = label,
                target = 1 - label,
                meta = {}
            )
        )
    ## for IMDb train set, sample 1200 / 1000 to generate adv examples
    if len(dataset) > 10000 and not all_data:
        random.shuffle(dataset)
        # if 'pwws' in args.attacker:
        #     dataset = dataset[:1200] #pwws needs more
        # else:
        #     dataset = dataset[:1000] #textfooler
        num = int(len(dataset) * 0.2)
        dataset = dataset[ : num]
    elif all_data and number:
        random.shuffle(dataset)
        dataset = dataset[:number]
    # print ("#examples: ", len(dataset))
    return Dataset(dataset)


def load_examples(args, tokenizer, inputs):
    ### inputs: list of raw sentences 
    task = "sst-2"
    # logger.info("Number of examples : {}".format(str(inputs)))
    processor = processors[task]()
    output_mode = output_modes[task]
    label_list = processor.get_labels()

    examples = processor._create_examples_adv(inputs, "test")

    features = convert_examples_to_features(
        examples,
        tokenizer,
        label_list=label_list,
        max_length=args.max_seq_length,
        output_mode=output_mode,
        pad_on_left=True,  
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
    )
    
    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
    return dataset 


# load our own classifier here
class BertClassifier(OpenAttack.Classifier):
    def __init__(self, args):
        if args.model_type == "roberta":
            model_class = RobertaMixText 
            config_class = RobertaConfig
            tokenizer_class = RobertaTokenizer
        else:
            model_class = MixText 
            config_class = BertConfig
            tokenizer_class = BertTokenizer
           
        self.config = config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task="sst-2",
        )

        self.tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path,
            do_lower_case=True,
        )

        self.model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=self.config,
        )
        self.model.to(args.device)

        self.args = args
        
        logger.info("Model Loaded")


    
    def get_prob(self, input_):
        args = self.args 
        # input_ = load_examples(args, self.tokenizer, input_)
        ret = []
        eval_dataset = load_examples(args, self.tokenizer, input_)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        
        for batch in eval_dataloader:
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            
            with torch.no_grad():
                input_ids = batch[0]
                attention_mask = batch[1]
                logits, outputs = self.model(input_ids, attention_mask)
                
                # print (logits.detach().cpu().numpy())
                ret.extend(logits.cpu().numpy())
        return np.array(ret)

    def get_pred(self, input_):
        logits = self.get_prob(input_)
        return np.argmax(logits, axis=1)        



# a general classifier for any model
class ModelClassifier(OpenAttack.Classifier):
    def __init__(self, tokenizer, model, args):
        self.tokenizer = tokenizer
        self.model = model
        self.args = args
        
    def get_prob(self, input_):
        args = self.args
        ret = []
        eval_dataset = load_examples(self.args, self.tokenizer, input_)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=64) # input_ bsz is just 1
        
        for batch in eval_dataloader:
            self.model.eval()
            batch = tuple(t.to(args.device) for t in batch)
            
            with torch.no_grad():
                input_ids = batch[0]
                attention_mask = batch[1]
                logits, outputs = self.model(input_ids, attention_mask)
                
                # print (logits.detach().cpu().numpy())
                ret.extend(logits.cpu().numpy())
        return np.array(ret)

    def get_pred(self, input_):
        logits = self.get_prob(input_)
        return np.argmax(logits, axis=1)    



# # configure access interface of the customized victim model by extending OpenAttack.Classifier.
# class MyClassifier(OpenAttack.Classifier):
#     def __init__(self):
#         # nltk.sentiment.vader.SentimentIntensityAnalyzer is a traditional sentiment classification model.
#         self.model = SentimentIntensityAnalyzer()
    
#     # access to the classification probability scores with respect input sentences
#     def get_prob(self, input_):
#         ret = []
#         for sent in input_:
#             # SentimentIntensityAnalyzer calculates scores of “neg” and “pos” for each instance
#             res = self.model.polarity_scores(sent)

#             # we use 𝑠𝑜𝑐𝑟𝑒_𝑝𝑜𝑠 / (𝑠𝑐𝑜𝑟𝑒_𝑛𝑒𝑔 + 𝑠𝑐𝑜𝑟𝑒_𝑝𝑜𝑠) to represent the probability of positive sentiment
#             # Adding 10^−6 is a trick to avoid dividing by zero.
#             prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 2e-6)

#             ret.append(np.array([1 - prob, prob]))
        
#         # The get_prob method finally returns a np.ndarray of shape (len(input_), 2). See Classifier for detail.
#         return np.array(ret)

attacker_map = {"pwws": PWWSAttacker, "generic": GeneticAttacker, "hotflip": HotFlipAttacker,
                "pso": PSOAttacker, "textfooler": TextFoolerAttacker,
                "uat": UATAttacker, "viper": VIPERAttacker}
        
def get_attacker(attacker):
    return attacker_map[attacker]()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
            "--model_name_or_path",
            default="/home/sichenglei/bert-base-uncased",
            type=str,
            help="Path to pre-trained model or shortcut name selected in the list:",
        )
    # parser.add_argument(
    #         "--do_atm",
    #         default=0,
    #         type=int,
    #         help="Whether to do attentive mix-up."
    #     )
    parser.add_argument(
            "--batch_size",
            default=128,
            type=int,
            help="Whether to do attentive mix-up."
        )
    parser.add_argument(
            "--attacker",
            default="pwws",
            type=str,
            help="The attacker to use.",
        )
    parser.add_argument(
            "--data_dir",
            default="/home/sichenglei/sst-2/test.tsv",
            type=str,
            help="directory of test file",
        )
    parser.add_argument(
            "--save_dir",
            default=None,
            type=str,
            help="directory to save the adversarial examples",
        )
    parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="Max Sequence Length"
        )
    parser.add_argument(
            "--model_type",
            default="bert",
            type=str,
            help="Use BERT or RoBERTa",
        )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    print ("#GPU: ", args.n_gpu)

    TEST_FILE = args.data_dir

    clsf = BertClassifier(args)
    
    dataset = load_custom_dataset(TEST_FILE)

    ## sanity check: original accuracy without attack
    # correct = 0
    # for data in dataset:
    #     pred = clsf.get_pred([data.x])
    #     if pred[0] == data.y:
    #         correct += 1
    # print ("Acc: {}/{}={}".format(correct, len(dataset), correct/len(dataset)*100))
   
    attacker = attacker_map[args.attacker]()
    
    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, clsf, progress_bar=True)

    res = attack_eval.eval(dataset, visualize=False, save_dir=args.save_dir)
    print (res)
    
if __name__ == "__main__":
    main()