# MixADA: Adversarial Training with Mixup Augmentation for Robust Fine-tuning

This is the repo for reproducing our paper - Better Robustness by More Coverage: Adversarial Training with Mixup Augmentation for Robust Fine-tuning ([arxiv](https://arxiv.org/abs/2012.15699)). 

## Dependencies 

I conducted all experiments under Torch==1.4.0, Transformers==2.3.0. You can see a complete list of dependencies in `requirements.txt`, although you don't have to install all of them as most of them are unnecessary for this codebase.

## Data

We provide the exact data that we used in our experiments for easier reproduction. The download link is [here](https://drive.google.com/file/d/1MIFljjU8sOzxZshBvq7gFqX9MidqUSFe/view?usp=sharing).

## Running 

I have included examples of how to run model training with MixADA as well as how to evaluate the models under adversarial attacks in `run_job.sh` and `run_job2.sh`. However, you need to modify the scripts to fill in your dataset and pretrained model checkpoint paths.

## Reference

Please consider citing our work if you found this code or our paper beneficial to your research.

```
@article{Si2020BetterRB,
  title={Better Robustness by More Coverage: Adversarial Training with Mixup Augmentation for Robust Fine-tuning},
  author={Chenglei Si and Zhengyan Zhang and Fanchao Qi and Zhiyuan Liu and Yasheng Wang and Qun Liu and Maosong Sun},
  journal={ArXiv},
  year={2020},
  volume={abs/2012.15699}
}
```

## Contact

If you encounter any problems, feel free to raise them in issues or contact the authors.