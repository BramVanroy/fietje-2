import argparse
from itertools import chain

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data (tokenize and group for CLM) and save to disk")
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="The name of the dataset to use (via the datasets library).",
        required=True,
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
        required=True,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models to select the tokenizer.",
        required=True,
    )
    parser.add_argument(
        "--text_column",
        default="text",
        type=str,
        help="text column to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to save preprocessed dataset.",
        required=True,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
        required=True,
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    args = parser.parse_args()
    return args


def main():
    # Heavily borrowed from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
    args = parse_args()

    datasets = load_dataset(args.dataset_name, args.dataset_config_name, num_proc=args.preprocessing_num_workers)
    datasets["train"] = datasets["train"].shuffle(seed=42, keep_in_memory=True)
    print("Shuffled training part of dataset!")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    def tokenize_function(examples):
        return tokenizer(examples[args.text_column], truncation=False, padding=False)

    column_names = datasets["train"].column_names
    datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        batch_size=10_000,
        desc="Running tokenizer on dataset",
    )
    print(f"Tokenized dataset with {args.model_name_or_path}!")

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // args.block_size) * args.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map
    datasets = datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        desc=f"Grouping texts in chunks of {args.block_size}",
        batch_size=10_000,
    )
    print(f"Grouped dataset in chunks of {args.block_size}!")
    try:
        datasets.save_to_disk(args.output_dir, num_proc=args.preprocessing_num_workers)
    except Exception:
        datasets.save_to_disk(args.output_dir)

    print(f"Saved dataset to {args.output_dir}!")
    print(datasets)


if __name__ == "__main__":
    main()
