from ..attack_eval import AttackEval
from ..classifier import Classifier
import json, sys, time
from tqdm import tqdm
from ..utils import visualizer, result_visualizer, check_parameters, DataInstance, Dataset
from ..exceptions import ClassifierNotSupportException
from ..text_processors import DefaultTextProcessor
import csv 

DEFAULT_CONFIG = {
    "processor": DefaultTextProcessor(),
    "language_tool": None,
    "language_model": None,
    "sentence_encoder": None,
    "levenshtein_tool": None,
    "modification_tool": None,

    "success_rate": True,
    "fluency": False,
    "mistake": False,
    "semantic": False,
    "levenstein": False,
    "word_distance": False,
    "modification_rate": True,
    "running_time": True,
}

class MetaClassifierWrapper(Classifier):
    def __init__(self, clsf):
        self.__meta = None
        self.__clsf = clsf
    def set_meta(self, meta):
        self.__meta = meta

    def get_pred(self, input_):
        return self.__clsf.get_pred(input_, self.__meta)
    
    def get_prob(self, input_):
        return self.__clsf.get_prob(input_, self.__meta)
    
    def get_grad(self, input_, labels):
        return self.__clsf.get_grad(input_, labels, self.__meta)

class DefaultAttackEval(AttackEval):
    """
    DefaultAttackEval is the default implementation of AttackEval that provides basic evaluation functions.

    In this class, there are four key methods that maybe useful for your extension.

    * **measure:** Measures the basic metrics.
    * **update:** Accumulates the measurement.
    * **get_result:** Calculates the final results.
    * **clear:** Clear all the Accumulated results.

    The workflow is: ``clear -> measure -> update -> measure -> update -> ... -> get_result``.
    You can override these four methods to add your custom measurement.

    See :doc:`Example 4 </examples/example4>` for detail.

    """
    def __init__(self, attacker, classifier, progress_bar=True, **kwargs):
        """
        :param Attacker attacker: The attacker you use.
        :param Classifier classifier: The classifier you want to attack.
        :param bool running_time: If true, returns "Avg. Running Time" in summary. **Default:** True
        :param bool progress_bar: Dispaly a progress bar(tqdm). **Default:** True
        :param bool success_rate: If true, returns "Attack Success Rate". **Default:** True
        :param bool fluency: If true, returns "Avg. Fluency (ppl)". **Default:** False
        :param bool mistake: If true, returns "Avg. Grammatical Errors". **Default:** False
        :param bool semantic: If true, returns "Avg. Semantic Similarity". **Default:** False
        :param bool levenstein: If true, returns "Avg. Levenshtein Edit Distance". **Default:** False
        :param bool word_distance: If true, applies token-level levenstein edit distance. **Default:** False
        :param bool modification_rate: If true, returns "Avg. Word Modif. Rate". **Default:** False
        :param TextProcessor processor: Text processor used in DefaultAttackEval. **Default:** :any:`DefaultTextProcessor`

        :Package Requirements:
            * **language_tool_python** (for `mistake` option)
            * **Java** (for `mistake` option)
            * **transformers** (for `fluency` option)
            * **tensorflow** >= 2.0.0 (for `semantic` option)
            * **tensorflow_hub** (for `semantic` option)

        :Data Requirements:
            * :py:data:`.UniversalSentenceEncoder` (for `semantic` option)
        """
        self.__config = DEFAULT_CONFIG.copy()
        self.__config.update(kwargs)
        check_parameters(DEFAULT_CONFIG.keys(), self.__config)
        self.clear()
        self.attacker = attacker
        self.classifier = classifier
        
        self.__progress_bar = progress_bar
    
    def eval(self, dataset, total_len=None, visualize=False, save_dir=None, return_examples=False):
        """
        :param Dataset dataset: A :py:class:`.Dataset` or a list of :py:class:`.DataInstance`.
        :type dataset: list or generator
        :param int total_len: If `dataset` is a generator, total_len is passed the progress bar.
        :param bool visualize: Display a visualized result for each instance and the summary.
        :return: Returns a dict of the summary.
        :rtype: dict
        :uat_label: used to specify what label UAT is targeting

        In this method, ``eval_results`` is called and gets the result for each instance iteratively.
        """
        if hasattr(dataset, "__len__"):
            total_len = len(dataset)
        
        counter = 0
        orig_correct = 0 ## originally correctly answered
        adv_correct = 0 ## after attack still correct 
        return_lst = []
        label_list = ["0", "1"]

        def tqdm_writer(x):
            return tqdm.write(x, end="")
        
        if save_dir is not None:
            save_file = open(save_dir, "w")
            writer = csv.writer(save_file, delimiter='\t')

        time_start = time.time()
        for data, x_adv, y_adv, info in (tqdm(self.eval_results(dataset), total=total_len) if self.__progress_bar else self.eval_results(dataset)):
            # print (data.x)
            # print (x_adv)
            if return_examples and x_adv:
                eg = [x_adv, str(label_list[data.y])]
                return_lst.append(eg)
            elif not return_examples:
                x_orig = data.x
                counter += 1
                ## if x_adv is not None, then adv_pred == target
                orig_pred = self.classifier.get_pred([x_orig])[0]
                if save_dir:
                    ## if attack failed: save the orig for test set; skip for train set.
                    eg = []
                    x_ = x_adv 
                    if x_ is None and 'test' in save_dir:
                        x_ = x_orig
                    if x_ is not None:
                        eg.append(x_)
                        eg.append(str(data.y))
                        writer.writerow(eg)

                if x_adv:
                    adv_pred = self.classifier.get_pred([x_adv])[0]
                else:
                    adv_pred = orig_pred # didn't change pred 
                if orig_pred == data.y:
                    orig_correct += 1
                # if x_adv and adv_pred == data.y:
                #     adv_correct += 1
                if adv_pred == data.y:
                    adv_correct += 1
                if visualize:
                    try:
                        if x_adv is not None:
                            res = self.classifier.get_prob([x_orig, x_adv], data.meta)
                            y_orig = res[0]
                            y_adv = res[1]
                        else:
                            y_orig = self.classifier.get_prob([x_orig], data.meta)[0]
                    except ClassifierNotSupportException:
                        if x_adv is not None:
                            res = self.classifier.get_pred([x_orig, x_adv], data.meta)
                            y_orig = int(res[0])
                            y_adv = int(res[1])
                        else:
                            y_orig = int(self.classifier.get_pred([x_orig], data.meta)[0])

                    if self.__progress_bar:
                        visualizer(counter, x_orig, y_orig, x_adv, y_adv, info, tqdm_writer)
                    else:
                        visualizer(counter, x_orig, y_orig, x_adv, y_adv, info, sys.stdout.write)
            
        if not return_examples:
            res = self.get_result()
            if self.__config["running_time"]:
                res["Avg. Running Time"] = (time.time() - time_start) / counter

            if visualize:
                result_visualizer(res, sys.stdout.write)
            orig_acc = orig_correct / counter
            adv_acc = adv_correct / counter
            drop = (adv_acc - orig_acc) / orig_acc * 100
            print ("Total #examples:", counter)
            print ("Orig Correct Acc: ", orig_acc)
            print ("Adv Correct Acc: ", adv_acc)
            print ("Percentage Drop: ", drop)

            # for data in dataset:
            #     x_orig = data.x
            #     counter += 1
            #     orig_pred = self.classifier.get_pred([x_orig])[0]
            #     if orig_pred == data.y:
            #         orig_correct += 1

            # print ("Acc: {}/{}={}".format(orig_correct, counter, orig_correct / counter))
            if save_dir is not None:
                print ("Adv Examples Saved at: ", save_dir)
                save_file.close()
            return res
        else:
            return return_lst

    def print(self):
        print( json.dumps( self.get_result(), indent="\t" ) )

    def dump(self, file_like_object):
        json.dump( self.get_result(), file_like_object )

    def dumps(self):
        return json.dumps( self.get_result() )
    
    def __update(self, sentA, sentB):
        info = self.measure(sentA, sentB)
        return self.update(info)

    def eval_results(self, dataset):
        """
        :param dataset: A :py:class:`.Dataset` or a list of :py:class:`.DataInstance`.
        :type dataset: Dataset or generator
        :return: A generator which generates the result for each instance, *(DataInstance, x_adv, y_adv, info)*.
        :rtype: generator
        """
        self.clear()

        clsf_wrapper = MetaClassifierWrapper(self.classifier)
        for data in dataset:
            assert isinstance(data, DataInstance)
            clsf_wrapper.set_meta(data.meta)
            try:
                res = self.attacker(clsf_wrapper, data.x, data.target)
            except:
            #     print ("Error")
                res = None 
            if res is None:
                info = self.__update(data.x, None)
            else:
                info = self.__update(data.x, res[0])
            if not info["Succeed"]:
                yield (data, None, None, info)
            else:
                yield (data, res[0], res[1], info)
    
    def __levenshtein(self, sentA, sentB):
        if self.__config["levenshtein_tool"] is None:
            from ..metric import Levenshtein
            self.__config["levenshtein_tool"] = Levenshtein()
        return self.__config["levenshtein_tool"](sentA, sentB)

    def __get_tokens(self, sent):
        return list(map(lambda x: x[0], self.__config["processor"].get_tokens(sent)))
    
    def __get_mistakes(self, sent):
        if self.__config["language_tool"] is None:
            import language_tool_python
            self.__config["language_tool"] = language_tool_python.LanguageTool('en-US')
        
        return len(self.__config["language_tool"].check(sent))
    
    def __get_fluency(self, sent):
        if self.__config["language_model"] is None:
            from ..metric import GPT2LM
            self.__config["language_model"] = GPT2LM()
        
        if len(sent.strip()) == 0:
            return 1
        return self.__config["language_model"](sent)
    
    def __get_semantic(self, sentA, sentB):
        if self.__config["sentence_encoder"] is None:
            from ..metric import UniversalSentenceEncoder
            self.__config["sentence_encoder"] = UniversalSentenceEncoder()
        
        return self.__config["sentence_encoder"](sentA, sentB)
    
    def __get_modification(self, sentA, sentB):
        if self.__config["modification_tool"] is None:
            from ..metric import Modification
            self.__config["modification_tool"] = Modification()
        tokenA = self.__get_tokens(sentA)
        tokenB = self.__get_tokens(sentB)
        return self.__config["modification_tool"](tokenA, tokenB)

    def measure(self, input_, attack_result):
        """
        :param str input_: The original sentence.
        :param attack_result: The adversarial sentence which is generated by attackers. If is None, means attacker failed to generate an adversarial sentence.
        :type attack_result: str or None
        :return: A dict contains all the results for this instance.
        :rtype: dict

        In this method, we measure all the metrics which corresponding options are setted to True.
        """
        if attack_result is None:
            return { "Succeed": False }

        info = { "Succeed": True }

        if self.__config["levenstein"]:
            va = input_
            vb = attack_result
            if self.__config["word_distance"]:
                va = self.__get_tokens(va)
                vb = self.__get_tokens(vb)
            info["Edit Distance"] =  self.__levenshtein(va, vb)
        
        if self.__config["mistake"]:
            info["Grammatical Errors"] = self.__get_mistakes(attack_result)
        
        if self.__config["fluency"]:
            info["Fluency (ppl)"] = self.__get_fluency(attack_result)
            
        if self.__config["semantic"]:
            info["Semantic Similarity"] = self.__get_semantic(input_, attack_result)

        if self.__config["modification_rate"]:
            info["Word Modif. Rate"] = self.__get_modification(input_, attack_result)
        return info
        
    def update(self, info):
        """
        :param dict info: The result returned by ``measure`` method.
        :return: Just return the parameter **info**.
        :rtype: dict

        In this method, we accumulate the results from ``measure`` method.
        """
        if "total" not in self.__result:
            self.__result["total"] = 0
        self.__result["total"] += 1

        if self.__config["success_rate"]:
            if "succeed" not in self.__result:
                self.__result["succeed"] = 0
            if info["Succeed"]:
                self.__result["succeed"] += 1
        
        # early stop
        if not info["Succeed"]:
            return info

        if self.__config["levenstein"]:
            if "edit" not in self.__result:
                self.__result["edit"] = 0
            self.__result["edit"] += info["Edit Distance"]
        
        if self.__config["mistake"]:
            if "mistake" not in self.__result:
                self.__result["mistake"] = 0
            self.__result["mistake"] += info["Grammatical Errors"]

        if self.__config["fluency"]:
            if "fluency" not in self.__result:
                self.__result["fluency"] = 0
            self.__result["fluency"] += info["Fluency (ppl)"]

        if self.__config["semantic"]:
            if "semantic" not in self.__result:
                self.__result["semantic"] = 0
            self.__result["semantic"] += info["Semantic Similarity"]
        
        if self.__config["modification_rate"]:
            if "modification" not in self.__result:
                self.__result["modification"] = 0
            self.__result["modification"] += info["Word Modif. Rate"]
        return info
        
    def get_result(self):
        """
        :return: The results which were accumulated previously.
        :rtype: dict

        This method summarizes and returns to previous accumulated results.
        """
        ret = {}
        ret["Total Attacked Instances"] = self.__result["total"]
        if self.__config["success_rate"]:
            ret["Successful Instances"] = self.__result["succeed"]
            ret["Attack Success Rate"] = self.__result["succeed"] / self.__result["total"]
        if self.__result["succeed"] > 0:
            if self.__config["levenstein"]:
                if "edit" not in self.__result:
                    self.__result["edit"] = 0
                ret["Avg. Levenshtein Edit Distance"] = self.__result["edit"] / self.__result["succeed"]
            if self.__config["mistake"]:
                if "mistake" not in self.__result:
                    self.__result["mistake"] = 0
                ret["Avg. Grammatical Errors"] = self.__result["mistake"] / self.__result["succeed"]
            if self.__config["fluency"]:
                if "fluency" not in self.__result:
                    self.__result["fluency"] = 0
                ret["Avg. Fluency (ppl)"] = self.__result["fluency"] / self.__result["succeed"]
            if self.__config["semantic"]:
                if "semantic" not in self.__result:
                    self.__result["semantic"] = 0
                ret["Avg. Semantic Similarity"] = self.__result["semantic"] / self.__result["succeed"]
            if self.__config["modification_rate"]:
                if "modification" not in self.__result:
                    self.__result["modification"] = 0
                ret["Avg. Word Modif. Rate"] = self.__result["modification"] / self.__result["succeed"]
        return ret

    def clear(self):
        """
        Clear all the accumulated results.
        """
        self.__result = {}
    
    def generate_adv(self, dataset, total_len=None):
        """
        :param Dataset dataset: A :py:class:`.Dataset` or a list of :py:class:`.DataInstance`.
        :return: A :py:class:`.Dataset` consists of adversarial samples.
        :rtype: Dataset
        """
        if hasattr(dataset, "__len__"):
            total_len = len(dataset)

        ret = []
        print ("Generating")
        ## we will keep everything, including a None 
        counter = 0
        for data, x_adv, y_adv, info in (tqdm(self.eval_results(dataset), total=total_len) if self.__progress_bar else self.eval_results(dataset)):
            counter += 1
            if counter % 500 == 0:
                print ("Current At: ", counter)
            if x_adv is None:
                x_adv = "None"
            # print (data.x)
            # print (x_adv)
            ret.append(DataInstance (
                x=x_adv,
                y=data.y,
                pred=y_adv,
                meta={
                    "original": data.x,
                    "info": info
                }
            ))
        return Dataset(ret)
            
            