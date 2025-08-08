from datasets import load_dataset
from loguru import logger
import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a dataset from a Parquet file.")
    parser.add_argument("--data_path", type=str, default="androidcontrol_low_test_fixed_scroll_wo_direction.parquet", help="Path to the Parquet file containing the dataset.")
    parser.add_argument("--subset_size", type=int, default=500, help="Size of the subset to create from the dataset.")
    args = parser.parse_args()

    dataset = load_dataset("parquet", data_files=args.data_path, split="train")
    logger.info(f"Dataset loaded with {len(dataset)} samples.")
    if len(dataset) < args.subset_size:
        logger.warning(f"Requested subset size {args.subset_size} is larger than the dataset size {len(dataset)}. Using the entire dataset.")
        subset_size = len(dataset)
    else:
        subset_size = args.subset_size
    random_indices = random.sample(range(len(dataset)), subset_size)
    subset = dataset.select(random_indices)
    logger.info(f"Subset created with {len(subset)} samples.")
    output_path = args.data_path.replace(".parquet", f"_subset_{subset_size}.parquet")
    subset.to_parquet(output_path)
    logger.info(f"Subset saved to {output_path}.")
