import contextlib
import math
import os
import shutil
from multiprocessing import Pool, cpu_count

from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm

from dalla_data_processing.utils.logger import get_logger

logger = get_logger(__name__)


def get_directory_size(path):
    total_size = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            # Skip if it's a symbolic link
            if not os.path.islink(filepath):
                with contextlib.suppress(OSError):
                    total_size += os.path.getsize(filepath)
    return total_size


def remove_path(path):
    """Safely remove a file, symlink, or directory tree."""
    try:
        if os.path.islink(path) or os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except Exception as e:
        logger.warning("Failed to remove path", path=path, error=str(e))


class DatasetPacker:
    def __init__(
        self,
        input_dataset,
        output_dataset,
        tokenizer,
        subset_order=None,
        num_workers=4,
        chunk_size_gb=2,
        max_seq_length=2048,
        sft=False,
        rbpe=False,
        text_column=None,
    ):
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self.tokenizer = tokenizer
        self.num_workers = num_workers
        self.chunk_size_bytes = int(chunk_size_gb * 1024**3)
        self.max_seq_length = max_seq_length
        self.rbpe = rbpe
        self.subset_order = subset_order
        self.sft = sft
        # Set text_column: use provided value, or default based on sft mode
        if text_column:
            self.text_column = text_column
        else:
            self.text_column = "messages" if sft else "text"
        if self.rbpe:
            self.parallel = False
            if self.sft:
                self.add_special_tokens = False
                self.append_concat_token = True
            else:
                self.add_special_tokens = True
                self.append_concat_token = False
        else:
            self.parallel = True
            self.add_special_tokens = True
            self.append_concat_token = True
        os.makedirs(output_dataset, exist_ok=True)

    def get_directory_sizes(self, base_path):
        # Handle both directory and direct dataset paths
        if os.path.isfile(os.path.join(base_path, "dataset_info.json")):
            # Direct dataset path
            size_bytes = 0
            for root, _dirs, files in os.walk(base_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    size_bytes += os.path.getsize(file_path)
            return {"dataset": size_bytes}
        else:
            sizes = {}
            dirs_to_check = self.subset_order if self.subset_order else os.listdir(base_path)
            for dir_name in dirs_to_check:
                path = os.path.join(base_path, dir_name)
                if not os.path.exists(path):
                    logger.warning("Directory not found", directory=dir_name, base_path=base_path)
                    continue
                size_bytes = 0
                for root, _dirs, files in os.walk(path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        size_bytes += os.path.getsize(file_path)
                sizes[dir_name] = size_bytes
            return sizes

    def split_dataset(self, base_path, sizes):
        counter = 0
        splits = []

        # Handle single dataset case
        if "dataset" in sizes:
            dataset = load_from_disk(base_path)
            size = sizes["dataset"]
            num_splits = math.ceil(size / self.chunk_size_bytes)
            total_size = len(dataset)
            subset_size = total_size // num_splits
            remainder = total_size % num_splits
            start_idx = 0

            for i in range(num_splits):
                current_size = subset_size + (1 if i < remainder else 0)
                end_idx = start_idx + current_size

                subset_data = dataset.select(range(start_idx, end_idx))
                name = f"{self.output_dataset}/split_{counter}"

                counter += 1
                subset_data.save_to_disk(name)
                size_bytes = get_directory_size(name)
                splits.append((name, size_bytes))
                del subset_data
                start_idx = end_idx
            return splits

        # Multiple datasets case
        logger.info("Splitting datasets", subset_order=self.subset_order)
        for subset in self.subset_order or sizes.keys():
            size = sizes[subset]
            dataset = load_from_disk(f"{base_path}/{subset}")
            num_splits = math.ceil(size / self.chunk_size_bytes)
            total_size = len(dataset)
            subset_size = total_size // num_splits
            remainder = total_size % num_splits
            start_idx = 0

            for i in range(num_splits):
                # Add one extra item to early splits if there's a remainder
                current_size = subset_size + (1 if i < remainder else 0)
                end_idx = start_idx + current_size

                subset_data = dataset.select(range(start_idx, end_idx))
                name = f"{self.output_dataset}/split_{counter}"

                counter += 1
                subset_data.save_to_disk(name)
                size_bytes = get_directory_size(name)
                splits.append((name, size_bytes))
                del subset_data
                start_idx = end_idx
        return splits

    def create_chunks(self, sizes):
        chunks = []
        current_chunk = []
        current_size = 0

        # Use subset_order instead of sorting
        for dir_name, size in sizes:
            if current_size + size > self.chunk_size_bytes and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            current_chunk.append(dir_name)
            current_size += size

        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def tokenize_batch(self, texts):
        if self.sft:
            return self.tokenizer.apply_chat_template(
                texts,
                truncation=False,
                padding=False,
                return_assistant_tokens_mask=True,
                return_dict=True,
                add_special_tokens=self.add_special_tokens,
            )
        else:
            return self.tokenizer(texts, truncation=False, padding=False)

    def process_chunk(self, chunk_ranges, chunk_idx):
        # Load and concatenate datasets in chunk
        dataset_splits = []
        for range_name in chunk_ranges:
            try:
                split = load_from_disk(range_name)
                dataset_splits.append(split)
                logger.info("Loaded split", split_name=range_name)
            except Exception as e:
                logger.error("Error loading split", split_name=range_name, error=str(e))

        if not dataset_splits:
            return None
        # Concatenate splits
        concatenated = concatenate_datasets(dataset_splits)
        del split
        del dataset_splits
        # delete splits
        for split in chunk_ranges:
            remove_path(split)
        # Tokenize
        logger.info("Tokenizing chunk", chunk_idx=chunk_idx)
        texts = concatenated[self.text_column]

        num_cores = cpu_count() - 1  # Leave some cores free
        text_chunk_size = len(texts) // num_cores
        text_chunks = [
            texts[i : i + text_chunk_size] for i in range(0, len(texts), text_chunk_size)
        ]

        if self.parallel:
            logger.info("Tokenizing in parallel", num_cores=num_cores)
            pool = Pool(num_cores)
            try:
                tokenized_chunks = list(
                    tqdm(
                        pool.imap(self.tokenize_batch, text_chunks),
                        total=len(text_chunks),
                        desc="Tokenizing",
                    )
                )
            finally:
                pool.close()
                pool.join()
        else:
            tokenized_chunks = [
                self.tokenize_batch(text_chunk)
                for text_chunk in tqdm(text_chunks, desc="Tokenizing")
            ]

        input_ids = [i["input_ids"] for i in tokenized_chunks]
        assistant_masks = (
            [i["assistant_masks"] for i in tokenized_chunks] if self.sft else input_ids
        )
        all_input_ids = [item for sublist in input_ids for item in sublist]
        all_assistant_masks = (
            [item for sublist in assistant_masks for item in sublist] if self.sft else all_input_ids
        )
        zeros = False
        for assistant_mask in all_assistant_masks:
            if all(i == 0 for i in assistant_mask):
                zeros = True
                break
        if zeros:
            tokenized_dataset = Dataset.from_dict(
                {"input_ids": all_input_ids, "labels": all_input_ids}
            )
        else:
            logger.debug("Assistant masks not all zeros")
            new_labels = []
            for assistant_mask, input_id in zip(all_assistant_masks, all_input_ids, strict=True):
                # new_labels.append([-100 * i*j for i, j in zip(assistant_mask, input_id)])
                # if i in attention_mask is 0, then have -100, otherwise have input_id
                new_labels.append(
                    [-100 if i == 0 else j for i, j in zip(assistant_mask, input_id, strict=True)]
                )

            tokenized_dataset = Dataset.from_dict(
                {"input_ids": all_input_ids, "labels": new_labels}
            )

        # Save tokenized dataset
        chunk_name = f"chunk_{chunk_idx}_tokenized"
        tokenized_path = os.path.join(self.output_dataset, chunk_name)
        tokenized_dataset.save_to_disk(tokenized_path)

        return tokenized_path

    def pack_sequences(self, input_path, output_path):
        packed_sequences = []
        batch_input_ids = []
        batch_labels = []
        batch_len = 0
        skipped_examples = []
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        tokenized = load_from_disk(input_path)
        logger.debug("Processing tokenized dataset", input_path=input_path)
        for n, example in tqdm(enumerate(tokenized), desc="Packing sequences"):
            masking = True
            if example["input_ids"] == example["labels"]:
                masking = False

            example_len = len(example["input_ids"])

            # Account for separator token if appending concat token
            sep_len = 1 if self.append_concat_token else 0
            if example_len + sep_len > self.max_seq_length:
                skipped_examples.append(n)
                continue

            if batch_len + example_len + sep_len > self.max_seq_length:
                # Pad and add current batch
                batch_input_ids.extend([pad_token_id] * (self.max_seq_length - batch_len))
                if masking:
                    batch_labels.extend([-100] * (self.max_seq_length - batch_len))
                else:
                    batch_labels.extend([pad_token_id] * (self.max_seq_length - batch_len))
                packed_sequences.append({"input_ids": batch_input_ids, "labels": batch_labels})
                batch_input_ids = []
                batch_labels = []
                batch_len = 0

            batch_input_ids.extend(example["input_ids"])
            batch_labels.extend(example["labels"])
            if self.append_concat_token:
                batch_input_ids.append(eos_token_id)  # Add separator token
                if masking:
                    batch_labels.append(-100)
                else:
                    batch_labels.append(eos_token_id)
                batch_len += example_len + 1
            else:
                batch_len += example_len

        # Handle last batch if not empty
        if batch_input_ids:
            batch_input_ids.extend([pad_token_id] * (self.max_seq_length - batch_len))
            if masking:
                batch_labels.extend([-100] * (self.max_seq_length - batch_len))
            else:
                batch_labels.extend([pad_token_id] * (self.max_seq_length - batch_len))
            packed_sequences.append({"input_ids": batch_input_ids, "labels": batch_labels})
        logger.info(
            "Skipped examples that exceeded max sequence length", num_skipped=len(skipped_examples)
        )
        del tokenized
        packed_dataset = Dataset.from_list(packed_sequences)
        packed_dataset.save_to_disk(output_path)
        remove_path(input_path)
        return output_path

    def _pack_dataset_wrapper(self, paths):
        input_path, output_path = paths
        try:
            self.pack_sequences(input_path, output_path)
            return output_path
        except Exception as e:
            logger.error("Error packing dataset", input_path=input_path, error=str(e))
            return None

    def pack_datasets_sequentially(self, tokenized_paths):
        """Pack datasets one at a time without parallel processing"""
        packed_paths = []
        for input_path in tokenized_paths:
            output_path = input_path.replace("_tokenized", "_packed")
            logger.info("Packing dataset", input_path=input_path)
            try:
                self.pack_sequences(input_path, output_path)
                packed_paths.append(output_path)
            except Exception as e:
                logger.error("Error packing dataset", input_path=input_path, error=str(e))
        return packed_paths

    def run_parallel_packing(self, tokenized_paths):
        # Prepare input/output path pairs
        pack_args = [
            (input_path, input_path.replace("_tokenized", "_packed"))
            for input_path in tokenized_paths
        ]

        pool = Pool(processes=self.num_workers)
        try:
            completed_paths = list(
                tqdm(
                    pool.imap(self._pack_dataset_wrapper, pack_args),
                    total=len(pack_args),
                    desc="Packing datasets",
                )
            )
        finally:
            pool.close()
            pool.join()

        # Filter out None values (failed packing attempts)
        logger.debug("Completed packing paths", completed_paths=completed_paths)
        return [path for path in completed_paths if path is not None]

    def concatenate_final_dataset(self, packed_paths):
        # Create a mapping of chunk index to dataset
        chunk_datasets = {}
        for path in packed_paths:
            # Extract chunk index from path
            chunk_idx = int(os.path.basename(path).split("_")[1])
            chunk_datasets[chunk_idx] = load_from_disk(path)

        # Load datasets in the original chunk order
        datasets = [chunk_datasets[i] for i in range(len(chunk_datasets))]
        final_dataset = concatenate_datasets(datasets)
        final_path = os.path.join(self.output_dataset, "final_dataset")
        dataset_dict = DatasetDict({"train": final_dataset})
        dataset_dict.save_to_disk(final_path)
        return final_path

    def process(self):
        # Step 1: Analyze and create chunks
        logger.info("Analyzing directory sizes")
        log_data = {}
        sizes = self.get_directory_sizes(self.input_dataset)
        log_data["sizes"] = sizes

        splits = self.split_dataset(self.input_dataset, sizes)
        log_data["splits"] = splits

        chunks = self.create_chunks(splits)
        log_data["chunks"] = chunks

        # Step 2: Process each chunk (concatenate and tokenize)
        tokenized_paths = []
        for i, chunk in enumerate(chunks):
            logger.info("Processing chunk", chunk_num=i + 1, total_chunks=len(chunks))
            tokenized_path = self.process_chunk(chunk, i)
            if tokenized_path:
                tokenized_paths.append(tokenized_path)

        log_data["tokenized"] = tokenized_paths

        # Step 3: Pack datasets (parallel or sequential)
        if self.parallel:
            logger.info("Packing datasets in parallel")
            packed_paths = self.run_parallel_packing(tokenized_paths)
        else:
            logger.info("Packing datasets sequentially")
            packed_paths = self.pack_datasets_sequentially(tokenized_paths)

        log_data["packed"] = packed_paths

        # Step 4: Concatenate final dataset
        logger.info("Concatenating final dataset")
        final_path = self.concatenate_final_dataset(packed_paths)
        for path in packed_paths:
            remove_path(path)

        log_data["final"] = final_path
        self.log_data = log_data
        logger.info("Processing complete! Final dataset saved", final_path=final_path)
        return final_path
