# Training Fietje

To train Fietje, I used the [alignment-hanbook](https://github.com/huggingface/alignment-handbook/). I added quite some PRs to the library to enable what I wanted to do, in particular support for continued pretraining (cpt). You should be able to use the alignment handbook with the recipes in this directory.

I used SLURM to launch my jobs, inspired by the description in [the handbook](https://github.com/huggingface/alignment-handbook/tree/main/scripts#launching-jobs-on-a-slurm-cluster). To run these on your own cluster, you may need to adapt the launch script. My own SLURM script is therefore somewhat different from the original, but the commands below should work with the default script after adding the `recipes/fietje-2b` directories to the corresponding directory in your clone of the alignment handbook.

## Continue-pretraining phase

Because of the size of the dataset, I decided to preprocess the data (tokenize and split into even chunks) and save it to disk. That's also useful if you are working with interrupting jobs, where you have to continue jobs often, so that the preprocessing only has to happen once. To prepare the data you can use [prepare_data.py](prepare_data.py). 

To prepare the data, you can use the following command. Then you should update the recipe in `fietje-2b/cpt` to point to the correct dataset path that you saved the new dataset to.

Note that fietje-2b was trained on an earlier version of this dataset. Filtering and structure is the same, but the shuffling and exact number of tokens might be different than the current dataset. 

Note that `20B` refers to the approximate number of white-space tokens in a configuration, *not* the subword tokens.

```shell
python prepare_data.py \
  --dataset_name BramVanroy/wikipedia_culturax_dutch \
  --dataset_config_name 20B \
  --model_name_or_path microsoft/phi-2 \
  --block_size 2048 \
  --preprocessing_num_workers 64 \
  --output_dir alignment-handbook/data/fietje-2b-cpt-prep
```


To train [Fietje 2B](https://huggingface.co/BramVanroy/fietje-2b) (base). This takes the longest. Actual training took around two weeks on four nodes of four GPUs (16x A100 80GB).

```shell
sbatch --job-name=fietje_cpt --nodes=4 recipes/launch.slurm fietje-2b cpt full deepspeed_zero3
```

## Supervised-finetuning phase

To create [Fietje 2B Instruct](https://huggingface.co/BramVanroy/fietje-2b-instruct). This is much faster and should take around a day (probably less) on 16x A100 80GB. 

```shell
sbatch --job-name=fietje_sft --nodes=4 recipes/launch.slurm fietje-2b sft full deepspeed_zero3
```

## Preference optimalisation phase

To finetune [Fietje 2B Chat](https://huggingface.co/BramVanroy/fietje-2b-chat). This should be relatively fast as the dataset is only around 20k samples. One run should take around 9 hours one one A100 80GB.

```shell
sbatch --job-name=fietje_dpo --nodes=1 recipes/launch.slurm fietje-2b dpo full deepspeed_zero3
```
