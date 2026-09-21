import argparse
import json
import os
import re
import sys
import time
import pickle
import hashlib
from collections import defaultdict, OrderedDict, deque
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue, Process
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from util import nethook
from dsets import KnownsDataset


# ============================================================
# CONFIGURATION CLASSES
# ============================================================

@dataclass
class T5DocIDEditHyperParams:
    """
    Hyperparameter configuration for T5-DOME DocID editing.

    Defines all parameters needed for the model editing process, including
    network structure, optimization parameters, and data parameters.

    Attributes:
        layers: List of decoder layer indices to edit
        decoder_module_tmp: Template path for T5 FFN output layer
        decoder_attn_module_tmp: Template path for T5 self-attention output layer
        v_loss_layer: Layer for loss computation (-1 = last layer)
        v_lr: Learning rate for z-vector optimization
        v_num_grad_steps: Number of gradient steps for z-vector optimization
        v_weight_decay: Weight decay coefficient
        clamp_norm_factor: Norm constraint factor for delta vectors
        kl_factor: Weight for KL divergence loss
        fact_token: Token position strategy for facts (default: "last")
        mom2_dataset: Dataset for covariance computation
        mom2_n_samples: Number of samples for covariance computation
        mom2_dtype: Data type for covariance matrix
        mom2_update_weight: Update weight for covariance matrix
        covariance_cache_dir: Directory for caching covariance matrices
        pseudo_num: Number of pseudo DocIDs
        z_vector_max: Maximum norm for z-vectors
        use_prob_threshold: Whether to use probability threshold
        preserve_lambda: MOSE weight on the preserved-key block of the
            Procrustes objective; scales mom2_update_weight
        mose_svd_reg: MOSE ridge added to M before the SVD, breaking ties
            toward the identity when M is rank-deficient
    """

    # Network structure parameters
    layers: List[int]

    # T5 module template paths
    decoder_module_tmp: str = "decoder.block.{}.layer.2.DenseReluDense.wo"
    decoder_attn_module_tmp: str = "decoder.block.{}.layer.0.SelfAttention.o"

    # Loss computation parameters
    v_loss_layer: int = -1
    v_lr: float = 0.5
    v_num_grad_steps: int = 300
    v_weight_decay: float = 0.2
    clamp_norm_factor: float = 1
    kl_factor: float = 0.0625

    # Fact token positioning strategy
    fact_token: str = "last"

    # Covariance matrix parameters
    mom2_dataset: str = "wikipedia"
    mom2_n_samples: int = 400000
    mom2_dtype: str = "float32"
    mom2_update_weight: float = 100
    covariance_cache_dir: str = "cache/covariance"

    # DocID-specific parameters
    pseudo_num: int = 10
    z_vector_max: int = 8000

    # Probability threshold parameters
    use_prob_threshold: bool = True
    prob_threshold: float = 0.3

    # MOSE parameters (multiplicative orthogonal update)
    preserve_lambda: float = 1.0
    mose_svd_reg: float = 0.0

    def save(self, save_path: str):
        """
        Save hyperparameter configuration to JSON file.

        Args:
            save_path: Path to save the configuration
        """
        with open(save_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
        print(f"Hyperparameters saved to {save_path}")


# ============================================================
# COVARIANCE CACHE MANAGEMENT
# ============================================================

class CovarianceManager:
    """
    Intelligent covariance matrix cache manager.

    Provides functionality to:
    1. Cache covariance matrices to disk intelligently
    2. Generate unique cache keys based on data content and model parameters
    3. Automatically detect stale caches
    4. Load and save covariance matrices with metadata

    The cache key is generated from:
    - Model name
    - Module name
    - Data file content hash (MD5)
    - Sample size
    - Data type

    This ensures cache invalidation when any of these parameters change.
    """

    def __init__(self, cache_dir: str = "cache/covariance"):
        """
        Initialize covariance cache manager.

        Args:
            cache_dir: Directory for storing cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(
            self,
            model_name: str,
            module_name: str,
            cov_data_file: str,
            sample_size: int,
            dtype: str
    ) -> str:
        """
        Generate unique cache key for covariance matrix.

        Creates a fingerprint by hashing the data file content and
        combining it with model/module parameters.

        Args:
            model_name: Name of the model
            module_name: Name of the module
            cov_data_file: Path to data file
            sample_size: Number of samples
            dtype: Data type string

        Returns:
            Unique cache key string
        """
        # Hash data file content for cache invalidation
        try:
            with open(cov_data_file, 'r', encoding='utf-8') as f:
                file_content = f.read()
            file_hash = hashlib.md5(file_content.encode()).hexdigest()
        except Exception as e:
            print(f"Warning: Failed to hash data file {cov_data_file}: {e}")
            file_hash = "no_hash"

        # Construct cache key components
        cache_components = [
            model_name.replace("/", "_").replace("\\", "_"),
            module_name.replace(".", "_"),
            file_hash[:12],  # Use first 12 chars of hash
            str(sample_size),
            dtype
        ]

        return "_".join(cache_components)

    def _get_cache_filepath(self, cache_key: str) -> Path:
        """Get cache file path for covariance matrix."""
        return self.cache_dir / f"cov_{cache_key}.pkl"

    def _get_metadata_filepath(self, cache_key: str) -> Path:
        """Get metadata file path for cache entry."""
        return self.cache_dir / f"meta_{cache_key}.json"

    def save_covariance(
            self,
            covariance_matrix: torch.Tensor,
            cache_key: str,
            metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save covariance matrix to cache with metadata.

        Args:
            covariance_matrix: The covariance matrix to cache
            cache_key: Unique cache identifier
            metadata: Additional metadata to store
        """
        cache_filepath = self._get_cache_filepath(cache_key)
        metadata_filepath = self._get_metadata_filepath(cache_key)

        try:
            # Save covariance matrix
            with open(cache_filepath, 'wb') as f:
                pickle.dump(covariance_matrix.cpu(), f)

            # Prepare and save metadata
            if metadata is None:
                metadata = {}

            metadata.update({
                "cache_key": cache_key,
                "matrix_shape": list(covariance_matrix.shape),
                "matrix_dtype": str(covariance_matrix.dtype),
                "created_at": str(pd.Timestamp.now()) if 'pandas' in sys.modules else "unknown",
                "cache_version": "1.0"
            })

            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✓ Covariance matrix cached at: {cache_filepath}")
            print(f"✓ Metadata saved at: {metadata_filepath}")

        except Exception as e:
            print(f"Error saving covariance to cache: {e}")
            # Clean up partial files
            for filepath in [cache_filepath, metadata_filepath]:
                if filepath.exists():
                    filepath.unlink()

    def load_covariance(self, cache_key: str) -> Optional[torch.Tensor]:
        """
        Load covariance matrix from cache.

        Args:
            cache_key: Unique cache identifier

        Returns:
            Covariance matrix if found, None otherwise
        """
        cache_filepath = self._get_cache_filepath(cache_key)
        metadata_filepath = self._get_metadata_filepath(cache_key)

        if not cache_filepath.exists():
            return None

        try:
            # Validate metadata if available
            if metadata_filepath.exists():
                with open(metadata_filepath, 'r') as f:
                    metadata = json.load(f)
                print(f"Loading cached covariance: {metadata.get('matrix_shape', 'unknown shape')}")

            # Load covariance matrix
            with open(cache_filepath, 'rb') as f:
                covariance_matrix = pickle.load(f)

            print(f"✓ Covariance matrix loaded from cache: {cache_filepath}")
            return covariance_matrix

        except Exception as e:
            print(f"Error loading covariance from cache: {e}")
            print(f"Cache file might be corrupted, will recompute...")
            # Remove corrupted cache files
            for filepath in [cache_filepath, metadata_filepath]:
                if filepath.exists():
                    try:
                        filepath.unlink()
                    except:
                        pass
            return None

    def cache_exists(self, cache_key: str) -> bool:
        """Check if cache exists for given key."""
        return self._get_cache_filepath(cache_key).exists()

    def clear_cache(self, cache_key: Optional[str] = None):
        """
        Clear cache files.

        Args:
            cache_key: Specific cache to clear, or None to clear all
        """
        if cache_key:
            # Clear specific cache
            cache_filepath = self._get_cache_filepath(cache_key)
            metadata_filepath = self._get_metadata_filepath(cache_key)
            for filepath in [cache_filepath, metadata_filepath]:
                if filepath.exists():
                    filepath.unlink()
                    print(f"Removed cache file: {filepath}")
        else:
            # Clear all caches
            for cache_file in self.cache_dir.glob("cov_*.pkl"):
                cache_file.unlink()
            for meta_file in self.cache_dir.glob("meta_*.json"):
                meta_file.unlink()
            print(f"Cleared all covariance cache in {self.cache_dir}")

    def list_cache(self):
        """List all cached covariance matrices with details."""
        cache_files = list(self.cache_dir.glob("cov_*.pkl"))
        print(f"Found {len(cache_files)} cached covariance matrices:")

        for cache_file in cache_files:
            cache_key = cache_file.stem.replace("cov_", "")
            metadata_file = self.cache_dir / f"meta_{cache_key}.json"

            file_size = cache_file.stat().st_size / (1024 * 1024)  # MB

            metadata_info = ""
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    shape = metadata.get('matrix_shape', 'unknown')
                    created = metadata.get('created_at', 'unknown')
                    metadata_info = f" | Shape: {shape} | Created: {created}"
                except:
                    metadata_info = " | Metadata error"

            print(f"  {cache_key} ({file_size:.1f}MB){metadata_info}")


# ============================================================
# MODEL AND TOKENIZER WRAPPER
# ============================================================

class ModelAndTokenizer:
    """
    Wrapper for T5 model and tokenizer with DocID token support.

    Functionality:
    1. Automatically load T5 model and tokenizer
    2. Add DocID special tokens ($0$, $1$, ..., $N$)
    3. Compute model layer information
    4. Pre-compute DocID token IDs for fast access

    Attributes:
        model: T5 model instance
        tokenizer: T5 tokenizer instance
        id_num: Number of DocID tokens
        num_decoder_layers: Number of decoder layers
        docid_token_ids: Pre-computed DocID token IDs
    """

    def __init__(
            self,
            model_name: Optional[str] = None,
            model: Optional[AutoModelForSeq2SeqLM] = None,
            tokenizer: Optional[AutoTokenizer] = None,
            low_cpu_mem_usage: bool = False,
            torch_dtype: Optional[torch.dtype] = None,
            id_num: int = 64,
            epoch_path: Optional[str] = 'data/output/t5-large-rq-ckpts/34.pt',
    ):
        """
        Initialize model and tokenizer with DocID support.

        Args:
            model_name: HuggingFace model name or local path
            model: Pre-loaded model instance (optional)
            tokenizer: Pre-loaded tokenizer instance (optional)
            low_cpu_mem_usage: Whether to use low CPU memory mode
            torch_dtype: Data type for model weights
            id_num: Number of DocID tokens to add
            epoch_path: Path to pre-trained checkpoint (optional)
        """
        # Initialize tokenizer
        if tokenizer is None:
            assert model_name is not None, "Must provide model_name if tokenizer is None"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Add DocID tokens
            for i in range(id_num):
                tokenizer.add_tokens([f"${i}$"])

        # Initialize model
        if model is None:
            assert model_name is not None, "Must provide model_name if model is None"
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                low_cpu_mem_usage=low_cpu_mem_usage,
                torch_dtype=torch_dtype
            )
            model.resize_token_embeddings(len(tokenizer))

            # Load pre-trained weights if available
            if epoch_path is not None and os.path.exists(epoch_path):
                print(f"Loading pretrained weights from {epoch_path}")
                model.load_state_dict(torch.load(epoch_path))
            elif epoch_path is not None:
                raise FileNotFoundError(
                    f"GR checkpoint not found: {epoch_path}\n"
                    f"DOME edits a GR model that was already trained on docIDs "
                    f"(GR/training.py). Editing the raw pretrained t5-large instead "
                    f"would make every reported number meaningless. Point "
                    f"--gr_checkpoint at a real checkpoint, or pass --gr_checkpoint '' "
                    f"to deliberately run without one."
                )

            # Freeze model and move to GPU
            nethook.set_requires_grad(False, model)
            model.eval().cuda()

        self.tokenizer = tokenizer
        self.model = model
        self.id_num = id_num

        # Calculate T5 model layer information
        if hasattr(model, "decoder") and hasattr(model.decoder, "block"):
            self.num_decoder_layers = len(list(model.decoder.block))
        else:
            self.num_decoder_layers = 0

        # Pre-compute DocID token IDs for fast access
        self.docid_token_ids = []
        for i in range(id_num):
            token_id = tokenizer.convert_tokens_to_ids(f"${i}$")
            self.docid_token_ids.append(token_id)
        self.docid_token_ids = torch.tensor(self.docid_token_ids)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} [T5-style, "
            f"{self.num_decoder_layers} decoder layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


# ============================================================
# MULTIPROCESSING WORKER FUNCTIONS
# ============================================================

def layer_keys_targets_worker(
        gpu_id: int,
        process_id: int,
        input_queue: Queue,
        output_queue: Queue,
        model_state_dict: Dict,
        tokenizer_config: Dict,
        hparams: T5DocIDEditHyperParams,
        id_num: int,
        target_layer: int
):
    """
    Worker process for computing layer keys and targets in parallel.

    This worker:
    - Reconstructs model on assigned GPU
    - Processes batches of edit targets
    - Extracts FFN inputs (keys) and target layer outputs
    - Returns results as NumPy arrays to reduce IPC overhead

    Performance optimizations:
    - Batch processing to maximize GPU utilization
    - Mixed precision (bfloat16/float16) for faster computation
    - Minimal data transfer between processes
    - Dynamic d_ff detection for model compatibility

    Args:
        gpu_id: CUDA device ID to use
        process_id: Worker process identifier
        input_queue: Queue for receiving tasks
        output_queue: Queue for sending results
        model_state_dict: Model weights (CPU)
        tokenizer_config: Tokenizer configuration
        hparams: Hyperparameters
        id_num: Number of DocID tokens
        target_layer: Target layer for output extraction
    """
    import traceback
    import numpy as np

    try:
        init_start = time.time()
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        # Enable low-level acceleration
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Reconstruct model and tokenizer
        model_init_start = time.time()
        mt_worker = ModelAndTokenizer(
            model_name=tokenizer_config['model_name'],
            id_num=id_num,
            epoch_path=None
        )

        # Load weights to GPU
        state_dict_gpu = {}
        for k, v in model_state_dict.items():
            state_dict_gpu[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        mt_worker.model.load_state_dict(state_dict_gpu, strict=True)
        mt_worker.model.eval()

        DOME_worker = T5DocIDDOME(hparams)
        model_init_time = time.time() - model_init_start

        # Module name templates
        wo_name_tmpl = "decoder.block.{}.layer.2.DenseReluDense.wo"
        tgt_mod_tmpl = hparams.decoder_module_tmp

        # Automatic mixed precision
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        init_total_time = time.time() - init_start
        print(f"[Process {process_id}] Initialization complete: {init_total_time:.3f}s "
              f"(model loading: {model_init_time:.3f}s)")

        while True:
            task = input_queue.get()
            if task is None:
                break

            task_id, layer, position, batch_items = task
            batch_start = time.time()

            try:
                model = mt_worker.model
                tokenizer = mt_worker.tokenizer

                wo_layer_name = wo_name_tmpl.format(layer)
                target_module_name = tgt_mod_tmpl.format(target_layer)

                # Dynamic d_ff detection (critical fix for different T5 variants)
                config_start = time.time()
                d_ff = getattr(model.config, "d_ff", None)
                if d_ff is None:
                    # Infer from wo weight shape: wo.weight.shape = [d_model, d_ff]
                    wo_w = nethook.get_parameter(model, f"{wo_layer_name}.weight")
                    d_ff = int(wo_w.shape[1])
                config_time = time.time() - config_start

                # Prepare batch data
                prep_start = time.time()
                orig_idx_list = [orig for (orig, _) in batch_items]
                batch_targets = [t for _, t in batch_items]
                batch_queries = [t["query"] for t in batch_targets]
                prep_time = time.time() - prep_start

                # Encode queries (fixed: removed non_blocking keyword)
                encode_start = time.time()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                    enc = tokenizer(
                        batch_queries,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(model.device)
                encode_time = time.time() - encode_start

                # Prepare decoder inputs
                decoder_prep_start = time.time()

                def _build_same_position_decoder_batch(batch_targets, position, tokenizer, mt, device):
                    """Build decoder inputs with identical positions."""
                    tokens = []
                    start_tok = mt.model.config.decoder_start_token_id or tokenizer.pad_token_id
                    for t in batch_targets:
                        full_docids = t.get("full_target_docids", [])
                        seq = [start_tok]
                        for j in range(position):
                            if j < len(full_docids):
                                tok = tokenizer.convert_tokens_to_ids(f"${full_docids[j]}$")
                                if tok == tokenizer.unk_token_id:
                                    tok = mt.docid_token_ids[0].item()
                            else:
                                tok = mt.docid_token_ids[0].item()
                            seq.append(tok)
                        tokens.append(torch.tensor(seq, device=device))
                    return torch.stack(tokens) if tokens else torch.empty(0, device=device)

                dec_inp = _build_same_position_decoder_batch(
                    batch_targets, position, tokenizer, mt_worker, model.device
                )
                decoder_prep_time = time.time() - decoder_prep_start

                # Forward pass with tracing
                forward_start = time.time()
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                    with nethook.TraceDict(
                            module=model,
                            layers=[wo_layer_name, target_module_name],
                            retain_input=True,
                            retain_output=True,
                    ) as tr:
                        _ = model(
                            input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"],
                            decoder_input_ids=dec_inp
                        )
                forward_time = time.time() - forward_start

                # Extract features
                extract_start = time.time()
                # Extract FFN input (K, shape [B, d_ff])
                if wo_layer_name in tr:
                    wo_in = tr[wo_layer_name].input
                    wo_t = wo_in[0] if isinstance(wo_in, (tuple, list)) else wo_in  # [B, L, d_ff]
                    pos_idx = position + 1
                    if wo_t.ndim == 3:
                        if pos_idx < wo_t.shape[1]:
                            K = wo_t[:, pos_idx, :].detach()
                        else:
                            K = wo_t[:, -1, :].detach()
                    else:
                        # Fallback for unexpected shapes
                        K = torch.zeros(len(batch_items), d_ff, device=model.device)
                else:
                    K = torch.zeros(len(batch_items), d_ff, device=model.device)

                # Target layer output at last position (R, shape [B, d_model])
                d_model = model.config.d_model
                if target_module_name in tr:
                    tout = tr[target_module_name].output
                    tt = tout[0] if isinstance(tout, (tuple, list)) else tout  # [B, L, d_model]
                    if tt.ndim == 3:
                        R = tt[:, -1, :].detach()
                    else:
                        R = torch.zeros(len(batch_items), d_model, device=model.device)
                else:
                    R = torch.zeros(len(batch_items), d_model, device=model.device)
                extract_time = time.time() - extract_start

                # Convert and send results
                output_start = time.time()
                output_queue.put({
                    "task_id": task_id,
                    "process_id": process_id,
                    "orig_idx": np.asarray(orig_idx_list, dtype=np.int32),
                    "K": K.to("cpu", dtype=torch.float32).contiguous().numpy(),
                    "R": R.to("cpu", dtype=torch.float32).contiguous().numpy(),
                })
                output_time = time.time() - output_start

                batch_total_time = time.time() - batch_start

                # Report timing breakdown
                print(f"[Process {process_id}] Task {task_id} complete (batch_size={len(batch_items)}):")
                print(f"  Config retrieval: {config_time:.3f}s")
                print(f"  Data preparation: {prep_time:.3f}s")
                print(f"  Encoding: {encode_time:.3f}s")
                print(f"  Decoder prep: {decoder_prep_time:.3f}s")
                print(f"  Forward pass: {forward_time:.3f}s")
                print(f"  Feature extraction: {extract_time:.3f}s")
                print(f"  Result output: {output_time:.3f}s")
                print(f"  Batch total: {batch_total_time:.3f}s")
                print("-" * 50)

            except Exception as e:
                traceback.print_exc()
                output_queue.put({"task_id": task_id, "error": str(e), "process_id": process_id})

    except Exception:
        traceback.print_exc()
    finally:
        try:
            output_queue.put({"task_id": None, "process_id": process_id, "done": True})
        except Exception:
            pass


def z_vector_worker(
        gpu_id: int,
        process_id: int,
        input_queue: Queue,
        output_queue: Queue,
        model_state_dict: Dict,
        tokenizer_config: Dict,
        hparams: T5DocIDEditHyperParams,
        id_num: int
):
    """
    Worker process for computing z-vectors in parallel.

    Z-vectors represent the desired output changes for each edit target.
    This worker optimizes z-vectors through gradient descent.

    Args:
        gpu_id: CUDA device ID
        process_id: Worker identifier
        input_queue: Task queue
        output_queue: Result queue
        model_state_dict: Model weights
        tokenizer_config: Tokenizer configuration
        hparams: Hyperparameters
        id_num: Number of DocID tokens
    """
    try:
        # Set CUDA device
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        # Reconstruct model
        mt_worker = ModelAndTokenizer(
            model_name=tokenizer_config['model_name'],
            id_num=id_num,
            epoch_path=None
        )

        # Load model state
        state_dict_gpu = {}
        for key, value in model_state_dict.items():
            if isinstance(value, torch.Tensor):
                state_dict_gpu[key] = value.to(device)
            else:
                state_dict_gpu[key] = value

        mt_worker.model.load_state_dict(state_dict_gpu)
        mt_worker.model.eval()

        # Create DOME instance
        DOME_worker = T5DocIDDOME(hparams)

        print(f"Worker {process_id} started on GPU {gpu_id}")

        while True:
            try:
                batch_data = input_queue.get(timeout=30)
                if batch_data is None:
                    break

                batch_idx, edit_targets_batch, target_layer, start = batch_data

                # Group by position to ensure consistent sequence lengths
                position_groups = DOME_worker._group_targets_by_position(edit_targets_batch)

                batch_results = []
                skip_indices = []

                for _position, _targets in position_groups.items():
                    batch_size = 128
                    for i in range(0, len(_targets), batch_size):
                        position = _position
                        targets = _targets[i:i + batch_size]

                        print(f"Process {process_id} processing position {position} "
                              f"batch with {len(targets)} targets")

                        # Batch compute z-vectors
                        z_vectors, skipped = DOME_worker._compute_z_vectors_batch(
                            mt_worker, targets, target_layer, position
                        )

                        # Collect results
                        for idx, (target, z_vector) in enumerate(zip(targets, z_vectors)):
                            if z_vector is not None:
                                global_idx = start + edit_targets_batch.index(target)
                                batch_results.append((global_idx, z_vector.detach().cpu().numpy()))

                # Return results
                output_queue.put({
                    'batch_idx': batch_idx,
                    'process_id': process_id,
                    'z_vectors': batch_results,
                    'skip_indices': skip_indices
                })

            except Exception as e:
                if "timeout" not in str(e).lower():
                    print(f"Worker {process_id} error: {e}")
                import traceback
                traceback.print_exc()
                break

    except Exception as e:
        print(f"Worker {process_id} initialization error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Worker {process_id} finished")


# ============================================================
# MAIN DOME EDITING CLASS
# ============================================================

class T5DocIDDOME:
    """
    T5-DOME implementation for generative retrieval DocID editing.

    Core functionality:
    1. Check model's current DocID generation capability
    2. Compute edit vectors for incorrect positions
    3. Apply model weight modifications
    4. Validate edit effectiveness

    The DOME algorithm works by:
    - Computing desired output representations (z-vectors)
    - Extracting key vectors from intermediate layers
    - Solving for optimal weight updates using covariance matrices
    - Applying updates layer by layer

    Attributes:
        hparams: Hyperparameter configuration
        cov_cache: In-memory cache for covariance matrices
        covariance_manager: Disk cache manager
        docid_z_cache: Cache for computed z-vectors
    """

    def __init__(self, hparams: T5DocIDEditHyperParams):
        """
        Initialize T5-DOME editor.

        Args:
            hparams: Hyperparameter configuration object
        """
        self.hparams = hparams
        self.cov_cache = {}  # In-memory covariance cache
        self.context_templates_cache = None  # Context template cache
        self.covariance_manager = CovarianceManager(hparams.covariance_cache_dir)
        self.docid_z_cache = defaultdict(list)  # Z-vector cache by (layer, docid, position)

    # ============================================================
    # TARGET GROUPING AND ORGANIZATION
    # ============================================================

    def _group_targets_by_position(self, edit_targets: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Group edit targets by position and interleave by target_docid.

        This ensures that targets are processed in a round-robin fashion
        across different docids, which helps with cache efficiency and
        load balancing.

        Args:
            edit_targets: List of edit target dictionaries

        Returns:
            Dict mapping position to interleaved target list
        """
        position_groups = {}

        # First group by position
        for target in edit_targets:
            pos = target["position"]
            position_groups.setdefault(pos, []).append(target)

        # Interleave within each position group by docid
        for pos, group in position_groups.items():
            # Use OrderedDict to maintain docid order
            buckets = OrderedDict()
            for t in group:
                buckets.setdefault(t["target_docid"], deque()).append(t)

            interleaved = []
            remaining = sum(len(q) for q in buckets.values())

            # Round-robin: take one from each docid queue until all are empty
            while remaining:
                for q in buckets.values():
                    if q:
                        interleaved.append(q.popleft())
                        remaining -= 1

            position_groups[pos] = interleaved

        return position_groups

    # ============================================================
    # Z-VECTOR COMPUTATION AND OPTIMIZATION
    # ============================================================

    def _compute_z_vectors_batch(
            self,
            mt: ModelAndTokenizer,
            targets: List[Dict],
            target_layer: int,
            position: int
    ) -> Tuple[List[Optional[torch.Tensor]], List[int]]:
        """
        Batch compute z-vectors for targets at the same position.

        Uses annealed soft/hard label optimization:
        - Soft labels: Model's probability distribution
        - Hard labels: One-hot target
        - Annealing: Gradually increase hard label weight from 0.3 to 1.0

        Args:
            mt: Model and tokenizer wrapper
            targets: List of edit targets (same position)
            target_layer: Target layer index
            position: Current position

        Returns:
            Tuple of (z_vectors, skip_indices)
        """
        start_time = time.time()

        model = mt.model
        tokenizer = mt.tokenizer
        device = model.device
        batch_size = len(targets)

        # ============================================================
        # Phase 1: Initialization
        # ============================================================
        init_start = time.time()
        batch_deltas = torch.zeros(batch_size, model.config.d_model, requires_grad=True, device=device)
        batch_target_inits = [None] * batch_size
        batch_satisfied = [False] * batch_size
        optimizer = torch.optim.Adam([batch_deltas], lr=self.hparams.v_lr)
        print(f"[TIMER] Initialization: {time.time() - init_start:.4f}s")

        # ============================================================
        # Phase 2: Input Preparation
        # ============================================================
        prep_start = time.time()
        enc = self._prepare_batch_encoder_inputs(mt, targets)
        dec = self._prepare_batch_decoder_inputs(mt, targets, position)
        tgt_ids = self._get_batch_target_token_ids(mt, targets)  # [B]
        print(f"[TIMER] Input preparation: {time.time() - prep_start:.4f}s")

        # ============================================================
        # Phase 3: Cache Check
        # ============================================================
        cache_start = time.time()
        cache_hits = self._check_cache_for_same_docids(mt, targets, target_layer, batch_satisfied)
        final_z_vectors = [None] * batch_size
        for i, z in cache_hits.items():
            final_z_vectors[i] = z
        print(f"[TIMER] Cache check: {time.time() - cache_start:.4f}s, hits: {len(cache_hits)}")

        # ============================================================
        # Phase 4: Smart Caching (Encoder and Decoder Prefix)
        # ============================================================
        base_cache_start = time.time()
        cached_encoder_states = None
        target_layer_name = f"decoder.block.{target_layer}.layer.2.DenseReluDense"

        with torch.no_grad():
            encoder_outputs = model.encoder(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            cached_encoder_states = encoder_outputs.last_hidden_state.detach()
        print(f"[TIMER] Smart caching: {time.time() - base_cache_start:.4f}s")

        # Build decoder prefix cache
        prefix_past = None
        if cached_encoder_states is not None:
            from transformers.modeling_outputs import BaseModelOutput
            enc_out = BaseModelOutput(last_hidden_state=cached_encoder_states)
            prefix_past = self._build_decoder_past_prefix(
                model=model,
                encoder_outputs=enc_out,
                attention_mask=enc["attention_mask"],
                decoder_input_ids=dec[:, :-1],
            )

        # ============================================================
        # Phase 5: Training Loop with Annealed Labels
        # ============================================================
        training_start = time.time()
        forward_time = 0
        backward_time = 0
        validation_time = 0

        active_indices = [i for i in range(batch_size) if not batch_satisfied[i]]

        # Annealing parameters
        anneal_steps = 50
        initial_hard_weight = 0.3

        for step in range(self.hparams.v_num_grad_steps):
            if len(active_indices) == 0:
                break

            forward_start = time.time()

            def smart_edit_fn(cur_out, cur_layer):
                """Apply z-vector edits to target layer output."""
                if cur_layer == target_layer_name:
                    if isinstance(cur_out, (list, tuple)):
                        modified_out = list(cur_out)
                        original_hidden = modified_out[0]
                        for i in active_indices:
                            if batch_target_inits[i] is None:
                                batch_target_inits[i] = original_hidden[i, -1, :].clone()
                        modified_hidden = original_hidden.clone()
                        for i in active_indices:
                            if batch_target_inits[i] is not None:
                                modified_hidden[i, -1, :] = batch_target_inits[i] + batch_deltas[i]
                        modified_out[0] = modified_hidden
                        return type(cur_out)(modified_out)
                    else:
                        for i in active_indices:
                            if batch_target_inits[i] is None:
                                batch_target_inits[i] = cur_out[i, -1, :].clone()
                        modified_out = cur_out.clone()
                        for i in active_indices:
                            if batch_target_inits[i] is not None:
                                modified_out[i, -1, :] = batch_target_inits[i] + batch_deltas[i]
                        return modified_out
                return cur_out

            # Forward pass with cached encoder outputs
            from transformers.modeling_outputs import BaseModelOutput
            if cached_encoder_states is not None:
                encoder_outputs = BaseModelOutput(last_hidden_state=cached_encoder_states)
            else:
                encoder_outputs = None

            with nethook.TraceDict(
                    module=model,
                    layers=[target_layer_name],
                    retain_input=False,
                    retain_output=True,
                    edit_output=smart_edit_fn,
            ):
                outputs = model(
                    encoder_outputs=encoder_outputs,
                    input_ids=None if encoder_outputs is not None else enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    decoder_input_ids=dec[:, -1:],
                    use_cache=True,
                    past_key_values=prefix_past,
                    output_hidden_states=False,
                    output_attentions=False,
                )

            logits = outputs.logits
            last = logits[:, -1, :]  # [B, V]

            # ============================================================
            # Annealed Soft/Hard Label Construction
            # ============================================================
            loss_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            loss_mask[active_indices] = True

            p = F.softmax(last, dim=-1)  # Soft labels from model output
            hard_q = torch.zeros_like(p)  # Hard one-hot labels
            hard_q.scatter_(1, tgt_ids.view(-1, 1), 1.0)

            # Compute annealing coefficient (linear schedule)
            alpha = min(initial_hard_weight + (1.0 - initial_hard_weight) * (step / anneal_steps), 1.0)
            soft_q = (1.0 - alpha) * p + alpha * hard_q

            # Compute cross-entropy loss
            eps = 1e-12
            neg_log_p = -torch.log(p.clamp_min(eps))
            per_sample_loss = (soft_q * neg_log_p).sum(dim=-1)
            active_losses = per_sample_loss[loss_mask]

            forward_time += time.time() - forward_start

            # ============================================================
            # Backward Pass with Weight Decay
            # ============================================================
            backward_start = time.time()
            optimizer.zero_grad()

            total_loss = torch.tensor(0.0, device=device)
            for idx, i in enumerate(active_indices):
                loss_i = active_losses[idx]
                if batch_target_inits[i] is not None:
                    target_norm = torch.norm(batch_target_inits[i])
                    delta_norm = torch.norm(batch_deltas[i])
                    wd = self.hparams.v_weight_decay * (delta_norm / (target_norm + 1e-8))
                    loss_i = loss_i + wd
                total_loss = total_loss + loss_i

            if total_loss.item() > 0:
                total_loss.backward()
                optimizer.step()

            # Constrain delta norms
            with torch.no_grad():
                for i in active_indices:
                    if batch_target_inits[i] is not None:
                        self._constrain_delta_norm_tensor(batch_deltas[i], batch_target_inits[i])

            backward_time += time.time() - backward_start

            # ============================================================
            # Validation Check
            # ============================================================
            if step > 0 and (step % 10 == 0 or step > self.hparams.v_num_grad_steps - 10):
                validation_start_time = time.time()
                validated_count = 0
                new_active_indices = []

                for idx, i in enumerate(active_indices):
                    if batch_target_inits[i] is None:
                        new_active_indices.append(i)
                        continue

                    predicted_token = torch.argmax(last[i]).item()
                    target_token = tgt_ids[i].item()
                    loss_value = active_losses[idx].item()

                    if predicted_token == target_token and loss_value < 0.05:
                        batch_satisfied[i] = True
                        final_z_vectors[i] = (batch_target_inits[i] + batch_deltas[i]).detach().clone()
                        validated_count += 1
                        continue

                    new_active_indices.append(i)

                active_indices = new_active_indices
                validation_time += time.time() - validation_start_time

        print(f"[TIMER] Training loop total: {time.time() - training_start:.4f}s")
        print(f"[TIMER]   - Forward passes: {forward_time:.4f}s")
        print(f"[TIMER]   - Backward passes: {backward_time:.4f}s")
        print(f"[TIMER]   - Validation: {validation_time:.4f}s")

        # ============================================================
        # Phase 6: Post-Processing and Cache Update
        # ============================================================
        for i in active_indices:
            if batch_target_inits[i] is not None:
                cur_z = batch_target_inits[i] + batch_deltas[i]
                ok, _p = self._test_z_vector_effectiveness(mt, targets[i], cur_z, target_layer)
                if ok:
                    batch_satisfied[i] = True
                    final_z_vectors[i] = cur_z.detach().clone()

        return final_z_vectors, [i for i, z in enumerate(final_z_vectors) if z is None]

    def _test_z_vector_effectiveness(
            self,
            mt: ModelAndTokenizer,
            edit_target: Dict,
            z_vector: torch.Tensor,
            layer: int
    ) -> Tuple[bool, float]:
        """
        Test z-vector effectiveness by temporarily applying it.

        Args:
            mt: Model and tokenizer
            edit_target: Edit target information
            z_vector: Current z-vector
            layer: Target layer number

        Returns:
            Tuple of (is_effective, target_probability)
        """
        model = mt.model
        tokenizer = mt.tokenizer
        query = edit_target["query"]
        target_docid = edit_target["target_docid"]
        position = edit_target["position"]
        full_target_docids = edit_target.get("full_target_docids", None)

        # Get target DocID token ID
        target_docid_token = f"${target_docid}$"
        target_token_id = tokenizer.convert_tokens_to_ids(target_docid_token)

        if target_token_id == tokenizer.unk_token_id:
            return False, 0.0

        # Prepare inputs
        encoder_inputs = tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(model.device)

        decoder_input = self._build_decoder_input_with_context(
            mt, query, position, full_target_docids=full_target_docids
        )

        # Define temporary edit function
        def temp_edit_output_fn(cur_out, cur_layer):
            """Temporarily apply z-vector."""
            layer_name = f"decoder.block.{layer}.layer.2.DenseReluDense"

            if cur_layer == layer_name:
                if isinstance(cur_out, (list, tuple)):
                    tensor_out = cur_out[0]
                else:
                    tensor_out = cur_out

                # Apply z-vector at target position
                if len(tensor_out.shape) == 3:
                    tensor_out[:, -1, :] = z_vector
                elif len(tensor_out.shape) == 2:
                    tensor_out[-1, :] = z_vector

                # Update cur_out
                if isinstance(cur_out, (list, tuple)):
                    cur_out = list(cur_out)
                    cur_out[0] = tensor_out
                    cur_out = type(cur_out)(cur_out)
                else:
                    cur_out = tensor_out

            return cur_out

        # Apply temporary edit and compute probability
        with nethook.TraceDict(
                module=model,
                layers=[f"decoder.block.{layer}.layer.2.DenseReluDense"],
                retain_input=False,
                retain_output=True,
                edit_output=temp_edit_output_fn,
        ):
            with torch.no_grad():
                outputs = model(
                    input_ids=encoder_inputs["input_ids"],
                    attention_mask=encoder_inputs["attention_mask"],
                    decoder_input_ids=decoder_input,
                )

        # Compute target DocID probability
        logits = outputs.logits
        target_logits = logits[:, -1, :]
        probs = F.softmax(target_logits, dim=-1)
        target_prob = probs[0, target_token_id].item()

        # Check if target has highest probability among all DocIDs
        for i in range(128):
            if i != target_docid:
                prob = probs[0, mt.docid_token_ids[i]].item()
                if prob > target_prob:
                    return False, target_prob

        # Apply probability threshold if enabled
        if self.hparams.use_prob_threshold and target_prob > self.hparams.prob_threshold:
            return True, target_prob

        return False, target_prob

    def _constrain_delta_norm_tensor(self, delta_tensor: torch.Tensor, target_init: torch.Tensor):
        """
        Constrain delta tensor norm to maximum allowed value.

        Args:
            delta_tensor: Delta tensor to constrain (modified in-place)
            target_init: Initial target value (for reference)
        """
        if target_init is None:
            return

        with torch.no_grad():
            delta_norm = torch.norm(delta_tensor)
            max_norm = self.hparams.z_vector_max

            if delta_norm > max_norm:
                delta_tensor.mul_(max_norm / delta_norm)

    def _check_cache_for_same_docids(
            self,
            mt: ModelAndTokenizer,
            targets: List[Dict],
            target_layer: int,
            batch_satisfied: List[bool]
    ) -> Dict[int, torch.Tensor]:
        """
        Check cache for z-vectors with matching (layer, docid, position).

        Args:
            mt: Model and tokenizer
            targets: List of targets
            target_layer: Target layer index
            batch_satisfied: List tracking which targets are satisfied

        Returns:
            Dict mapping batch index to cached z-vector
        """
        hits = {}
        for i, t in enumerate(targets):
            if batch_satisfied[i]:
                continue

            # Cache key: (layer, docid, position)
            k = (target_layer, str(t["target_docid"]), t["position"])
            if k not in self.docid_z_cache:
                continue

            # Test each cached z-vector for this key
            for z in self.docid_z_cache[k]:
                flag, _prob = self._test_z_vector_effectiveness(mt, t, z, target_layer)
                if flag:
                    batch_satisfied[i] = True
                    hits[i] = z.to(mt.model.device, non_blocking=True)
                    break

        return hits

    # ============================================================
    # INPUT PREPARATION UTILITIES
    # ============================================================

    def _prepare_batch_encoder_inputs(self, mt: ModelAndTokenizer, targets: List[Dict]) -> Dict:
        """
        Prepare batch encoder inputs from targets.

        Args:
            mt: Model and tokenizer
            targets: List of edit targets

        Returns:
            Tokenized encoder inputs
        """
        queries = [target["query"] for target in targets]
        return mt.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(mt.model.device)

    def _prepare_batch_decoder_inputs(
            self,
            mt: ModelAndTokenizer,
            targets: List[Dict],
            position: int
    ) -> torch.Tensor:
        """
        Prepare batch decoder inputs with consistent sequence lengths.

        All targets at the same position will have the same sequence length.

        Args:
            mt: Model and tokenizer
            targets: List of edit targets (same position)
            position: Current position

        Returns:
            Batch of decoder input tensors
        """
        batch_decoder_inputs = []

        for target in targets:
            decoder_tokens = [mt.model.config.decoder_start_token_id or mt.tokenizer.pad_token_id]

            # Add preceding docid context
            full_target_docids = target.get("full_target_docids", [])
            for j in range(position):
                if j < len(full_target_docids):
                    docid_token = f"${full_target_docids[j]}$"
                    docid_id = mt.tokenizer.convert_tokens_to_ids(docid_token)
                    if docid_id == mt.tokenizer.unk_token_id:
                        docid_id = mt.docid_token_ids[0]
                else:
                    docid_id = mt.docid_token_ids[0]
                decoder_tokens.append(docid_id)

            batch_decoder_inputs.append(torch.tensor(decoder_tokens, device=mt.model.device))

        # Same position means same length - stack directly
        return torch.stack(batch_decoder_inputs)

    def _get_batch_target_token_ids(self, mt: ModelAndTokenizer, targets: List[Dict]) -> torch.Tensor:
        """
        Get batch target token IDs.

        Args:
            mt: Model and tokenizer
            targets: List of edit targets

        Returns:
            Tensor of target token IDs
        """
        target_token_ids = []
        for target in targets:
            target_docid = target["target_docid"]
            target_docid_token = f"${target_docid}$"
            target_token_id = mt.tokenizer.convert_tokens_to_ids(target_docid_token)
            target_token_ids.append(target_token_id)
        return torch.tensor(target_token_ids, device=mt.model.device)

    def _build_decoder_past_prefix(
            self,
            model,
            encoder_outputs,
            attention_mask,
            decoder_input_ids,
    ):
        """
        Build decoder past key values for caching.

        Performs one forward pass without editing to get past_key_values,
        which can be reused for subsequent forward passes.

        Args:
            model: T5 model
            encoder_outputs: Encoder hidden states
            attention_mask: Attention mask
            decoder_input_ids: Decoder input IDs

        Returns:
            Past key values tuple
        """
        if decoder_input_ids.size(1) < 1:
            return None

        with torch.no_grad():
            out = model(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                use_cache=True,
                output_hidden_states=False,
                output_attentions=False,
            )

        return out.past_key_values

    def _build_decoder_input_with_context(
            self,
            mt: ModelAndTokenizer,
            query: str,
            position: int,
            edit_targets: Optional[List[Dict]] = None,
            full_target_docids: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Build decoder input sequence with correct preceding docid context.

        Uses teacher forcing with real preceding docids to ensure proper
        autoregressive context.

        Args:
            mt: Model and tokenizer
            query: Query text
            position: Current prediction position
            edit_targets: All edit targets (for finding same-query positions)
            full_target_docids: Complete target docid sequence

        Returns:
            Decoder input tensor
        """
        model = mt.model
        tokenizer = mt.tokenizer

        # Start decoder sequence
        decoder_start_token_id = model.config.decoder_start_token_id or tokenizer.pad_token_id
        decoder_input = torch.tensor([[decoder_start_token_id]], device=model.device)

        # Get real preceding docids
        real_docids = []

        # Strategy: Use provided full docid sequence
        if full_target_docids is not None:
            real_docids = full_target_docids
        else:
            raise ValueError("No full_target_docids provided for building decoder input")

        # Build token sequence for preceding positions
        for i in range(position):
            if i < len(real_docids):
                docid_token = f"${real_docids[i]}$"
                docid_id = tokenizer.convert_tokens_to_ids(docid_token)

                if docid_id == tokenizer.unk_token_id:
                    print(f"WARNING: DocID token {docid_token} not found, using $0$")
                    docid_id = mt.docid_token_ids[0]
            else:
                # Fallback to $0$ if not enough docids
                docid_id = mt.docid_token_ids[0]

            decoder_input = torch.cat([
                decoder_input,
                torch.tensor([[docid_id]], device=model.device)
            ], dim=1)

        return decoder_input

    # ============================================================
    # COVARIANCE COMPUTATION
    # ============================================================

    def _compute_covariance_data(
            self,
            mt: ModelAndTokenizer,
            layer_name: str,
            cov_data_file: str
    ) -> torch.Tensor:
        """
        Compute covariance matrix from DocID dataset.

        Key features:
        1. Load data from knowns format JSON file
        2. Compute covariance for each DocID output position
        3. Use teacher forcing to simulate generation
        4. Cache results to disk

        Args:
            mt: Model and tokenizer
            layer_name: Layer name for covariance computation
            cov_data_file: Path to knowns JSON file

        Returns:
            Covariance matrix tensor
        """
        model = mt.model
        tokenizer = mt.tokenizer

        # Generate cache key
        model_name = getattr(model.config, '_name_or_path', 'unknown_model')
        cache_key = self.covariance_manager._generate_cache_key(
            model_name=model_name,
            module_name=layer_name,
            cov_data_file=cov_data_file,
            sample_size=self.hparams.mom2_n_samples,
            dtype=self.hparams.mom2_dtype
        )

        print(f"Covariance cache key: {cache_key}")

        # Try loading from cache
        cached_matrix = self.covariance_manager.load_covariance(cache_key)
        if cached_matrix is not None:
            print(f"✓ Using cached covariance matrix for {layer_name}")
            print(f"✓ Cache hit! Shape: {cached_matrix.shape}")
            return cached_matrix.to(model.device)

        # Cache miss - compute covariance
        print(f"Cache miss. Computing DocID covariance from {cov_data_file} for layer {layer_name}")
        print(f"Target covariance shape should be [3072, 3072] for T5 FFN internal activations")

        # Check file existence
        if not os.path.exists(cov_data_file):
            raise FileNotFoundError(f"Covariance data file not found: {cov_data_file}")

        # Load knowns dataset
        try:
            knowns_dataset = KnownsDataset(cov_data_file)
            print(f"Loaded {len(knowns_dataset)} items from knowns dataset")
        except Exception as e:
            raise ValueError(f"Error loading knowns dataset: {e}")

        # Extract query->docid patterns
        docid_data = []
        docid_pattern = r'\$(\d+)\$'

        for item in knowns_dataset:
            try:
                query = item.get("prompt", "").strip()
                attribute = item.get("attribute", "").strip()

                if not query or not attribute:
                    continue

                # Extract DocID sequence from attribute
                docids = re.findall(docid_pattern, attribute)

                # Validate DocIDs are in valid range
                valid_docids = []
                for docid in docids:
                    if docid.isdigit() and int(docid) < mt.id_num:
                        valid_docids.append(docid)

                if valid_docids:
                    docid_data.append({
                        "query": query,
                        "docids": valid_docids,
                        "known_id": item.get("known_id", len(docid_data))
                    })

            except Exception as e:
                print(f"Error processing item: {e}")
                continue

        print(f"Extracted {len(docid_data)} valid query->docid mappings")

        if not docid_data:
            raise ValueError("No valid DocID data found in the dataset")

        # Compute covariance using optimized statistics
        stat = self.docid_layer_stats_optimized(
            model=model,
            tokenizer=tokenizer,
            layer_name=layer_name,
            docid_data=docid_data,
            mt=mt,
            sample_size=self.hparams.mom2_n_samples,
            precision=self.hparams.mom2_dtype,
        )

        covariance_matrix = stat.mom2.covariance().float().to("cpu")
        print(f"Final covariance matrix shape: {covariance_matrix.shape} (should be [3072, 3072])")

        # Save to cache
        metadata = {
            "model_name": model_name,
            "layer_name": layer_name,
            "data_file": cov_data_file,
            "sample_size": self.hparams.mom2_n_samples,
            "dtype": self.hparams.mom2_dtype,
            "num_docid_data": len(docid_data)
        }

        self.covariance_manager.save_covariance(covariance_matrix, cache_key, metadata)

        return covariance_matrix

    def docid_layer_stats_optimized(
            self,
            model,
            tokenizer,
            layer_name: str,
            docid_data: List[Dict],
            mt: ModelAndTokenizer,
            sample_size: Optional[int] = None,
            precision: Optional[str] = None,
    ) -> "CombinedStat":
        """
        Compute layer statistics for DocID generation using optimized sampling.

        This method uses teacher forcing to simulate autoregressive DocID generation
        and collects activation statistics from the target layer.

        Key optimizations:
        1. Batch processing for efficiency
        2. Smart sampling to ensure diversity
        3. Per-position statistics collection
        4. Memory-efficient activation collection

        Args:
            model: T5 model
            tokenizer: Tokenizer
            layer_name: Target layer name
            docid_data: List of query->docid mappings
            mt: Model and tokenizer wrapper
            sample_size: Number of samples to use (None = use all)
            precision: Precision for statistics (fp32/fp16)

        Returns:
            CombinedStat object containing mean and covariance
        """
        try:
            from .layer_stats import CombinedStat, Covariance
        except ImportError:  # run as a top-level script: python DOME.py
            from layer_stats import CombinedStat, Covariance

        print(f"\n{'=' * 60}")
        print(f"Computing layer statistics for: {layer_name}")
        print(f"{'=' * 60}")

        device = model.device
        dtype = torch.float32 if precision == "fp32" else torch.float16

        # Sample data if needed
        if sample_size is not None and sample_size < len(docid_data):
            sampled_indices = torch.randperm(len(docid_data))[:sample_size].tolist()
            sampled_data = [docid_data[i] for i in sampled_indices]
            print(f"Sampled {sample_size} from {len(docid_data)} total examples")
        else:
            sampled_data = docid_data
            print(f"Using all {len(sampled_data)} examples")

        # Initialize statistics collector
        combined_stat = CombinedStat(model.config.d_model)
        total_tokens_processed = 0

        # Process in batches
        batch_size = self.hparams.stats_batch_size
        num_batches = (len(sampled_data) + batch_size - 1) // batch_size

        print(f"Processing {num_batches} batches (batch_size={batch_size})")

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sampled_data))
            batch_data = sampled_data[start_idx:end_idx]

            # Process this batch
            activations = self._collect_docid_activations_batch(
                model=model,
                tokenizer=tokenizer,
                layer_name=layer_name,
                batch_data=batch_data,
                mt=mt,
            )

            # Update statistics with collected activations
            for act in activations:
                act_tensor = act.to(dtype).to(device)
                combined_stat.add(act_tensor)
                total_tokens_processed += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed batch {batch_idx + 1}/{num_batches} "
                      f"({total_tokens_processed} tokens)")

        print(f"\n✓ Statistics computation complete")
        print(f"  Total tokens processed: {total_tokens_processed}")
        print(f"  Mean shape: {combined_stat.mean().shape}")
        print(f"  Covariance shape: {combined_stat.mom2.covariance().shape}")
        print(f"{'=' * 60}\n")

        return combined_stat

    def _collect_docid_activations_batch(
            self,
            model,
            tokenizer,
            layer_name: str,
            batch_data: List[Dict],
            mt: ModelAndTokenizer,
    ) -> List[torch.Tensor]:
        """
        Collect activations for a batch of DocID generation examples.

        Uses teacher forcing: feeds correct preceding DocIDs to model
        to collect intermediate activations at each position.

        Args:
            model: T5 model
            tokenizer: Tokenizer
            layer_name: Target layer name
            batch_data: Batch of query->docid mappings
            mt: Model and tokenizer wrapper

        Returns:
            List of activation tensors
        """
        all_activations = []

        for data_item in batch_data:
            query = data_item["query"]
            docids = data_item["docids"]

            # Encode query
            encoder_inputs = tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            # Generate DocID sequence with teacher forcing
            # For each position, feed correct preceding DocIDs
            for pos in range(len(docids)):
                # Build decoder input up to current position
                decoder_tokens = [
                    model.config.decoder_start_token_id or tokenizer.pad_token_id
                ]

                # Add preceding DocIDs (teacher forcing)
                for j in range(pos):
                    docid_token = f"${docids[j]}$"
                    docid_id = tokenizer.convert_tokens_to_ids(docid_token)
                    if docid_id == tokenizer.unk_token_id:
                        docid_id = mt.docid_token_ids[0]
                    decoder_tokens.append(docid_id)

                decoder_input = torch.tensor(
                    [decoder_tokens],
                    device=model.device
                )

                # Collect activation from target layer
                with torch.no_grad():
                    with nethook.TraceDict(
                            module=model,
                            layers=[layer_name],
                            retain_input=False,
                            retain_output=True,
                    ) as traces:
                        _ = model(
                            input_ids=encoder_inputs["input_ids"],
                            attention_mask=encoder_inputs["attention_mask"],
                            decoder_input_ids=decoder_input,
                        )

                    # Extract activation from target layer
                    layer_output = traces[layer_name].output

                    if isinstance(layer_output, (tuple, list)):
                        activation = layer_output[0][:, -1, :].squeeze()
                    else:
                        activation = layer_output[:, -1, :].squeeze()

                    all_activations.append(activation.cpu())

        return all_activations

    # ============================================================
    # WEIGHT UPDATE COMPUTATION (DOME Algorithm Core)
    # ============================================================

    def _compute_weight_update(
            self,
            mt: ModelAndTokenizer,
            layer_idx: int,
            z_vectors: List[torch.Tensor],
            key_vectors: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weight updates using DOME algorithm.

        The core DOME equation:
            W_new = W_old + λ * (Z - W_old @ K^T) @ K @ C^(-1)

        Where:
            - Z: Desired output representations (z-vectors)
            - K: Key vectors from intermediate layer
            - C: Covariance matrix of key vectors
            - λ: Update coefficient (mom2_update_weight)

        Args:
            mt: Model and tokenizer
            layer_idx: Target layer index
            z_vectors: List of desired output vectors
            key_vectors: List of corresponding key vectors

        Returns:
            Dictionary of weight updates for this layer
        """
        model = mt.model
        device = model.device

        # Get layer modules
        layer_name = f"decoder.block.{layer_idx}.layer.2.DenseReluDense"
        wi_name = f"{layer_name}.wi.weight"
        wo_name = f"{layer_name}.wo.weight"

        print(f"\n{'=' * 60}")
        print(f"Computing weight update for layer {layer_idx}")
        print(f"{'=' * 60}")

        # Stack vectors into matrices
        Z = torch.stack(z_vectors).to(device)  # [N, d_model]
        K = torch.stack(key_vectors).to(device)  # [N, d_model]

        print(f"Z matrix shape: {Z.shape}")
        print(f"K matrix shape: {K.shape}")

        # Load covariance matrix
        cov_data_file = self.hparams.covariance_data_file
        C = self._compute_covariance_data(mt, layer_name, cov_data_file)
        C = C.to(device)

        print(f"Covariance matrix shape: {C.shape}")

        # Add regularization to covariance matrix
        C_reg = C + self.hparams.mom2_eps * torch.eye(
            C.shape[0], device=device
        )

        # Compute C^(-1) using Cholesky decomposition for stability
        try:
            L = torch.linalg.cholesky(C_reg)
            C_inv = torch.cholesky_inverse(L)
        except RuntimeError:
            print("Warning: Cholesky decomposition failed, using pseudo-inverse")
            C_inv = torch.pinverse(C_reg)

        # Get current weights
        wi_weight = nethook.get_parameter(model, wi_name)
        wo_weight = nethook.get_parameter(model, wo_name)

        # Compute weight updates
        # For input projection: ΔW_i = λ * (Z - W_i @ K^T) @ K @ C^(-1)
        K_C_inv = K @ C_inv  # [N, d_model]

        residual = Z - (wi_weight @ K.T).T  # [N, d_model]
        delta_wi = self.hparams.mom2_update_weight * (
                residual.T @ K_C_inv
        )  # [d_model, d_model]

        # For output projection: similar computation
        delta_wo = self.hparams.mom2_update_weight * (
                (Z - (wo_weight @ K.T).T).T @ K_C_inv
        )  # [d_model, d_model]

        print(f"ΔW_i norm: {torch.norm(delta_wi).item():.6f}")
        print(f"ΔW_o norm: {torch.norm(delta_wo).item():.6f}")
        print(f"{'=' * 60}\n")

        return {
            wi_name: delta_wi,
            wo_name: delta_wo,
        }

    # ============================================================
    # MODEL EDITING AND APPLICATION
    # ============================================================

    def apply_dome_edit(
            self,
            mt: ModelAndTokenizer,
            edit_targets: List[Dict],
            keep_original_weights: bool = True,
    ) -> ModelAndTokenizer:
        """
        Apply DOME edits to model.

        High-level workflow:
        1. Group targets by position
        2. For each position:
           a. Compute z-vectors (desired outputs)
           b. Extract key vectors (intermediate activations)
           c. Compute weight updates using DOME algorithm
           d. Apply updates to model weights
        3. Validate edits

        Args:
            mt: Model and tokenizer to edit
            edit_targets: List of edit specifications
            keep_original_weights: Whether to save original weights

        Returns:
            Edited ModelAndTokenizer object
        """
        model = mt.model

        print(f"\n{'=' * 70}")
        print(f"Starting DOME Edit Process")
        print(f"{'=' * 70}")
        print(f"Total edit targets: {len(edit_targets)}")

        # Save original weights if requested
        if keep_original_weights:
            print("Saving original model weights...")
            self.original_weights = {}
            for name, param in model.named_parameters():
                if "DenseReluDense" in name:
                    self.original_weights[name] = param.data.clone()

        # Group targets by position for batch processing
        position_groups = self._group_targets_by_position(edit_targets)
        print(f"Grouped into {len(position_groups)} position groups")

        # Process each position group
        all_weight_updates = {}

        for position, targets in sorted(position_groups.items()):
            print(f"\n{'=' * 70}")
            print(f"Processing Position {position} ({len(targets)} targets)")
            print(f"{'=' * 70}")

            # Determine target layer for this position
            layer_idx = self._get_layer_for_position(position)
            print(f"Target layer: {layer_idx}")

            # Compute z-vectors for all targets at this position
            z_vectors, failed_indices = self._compute_z_vectors_batch(
                mt=mt,
                targets=targets,
                target_layer=layer_idx,
                position=position,
            )

            successful_count = len([z for z in z_vectors if z is not None])
            print(f"Successfully computed {successful_count}/{len(targets)} z-vectors")

            # Filter out failed targets
            valid_targets = []
            valid_z_vectors = []
            for i, z in enumerate(z_vectors):
                if z is not None:
                    valid_targets.append(targets[i])
                    valid_z_vectors.append(z)

            if not valid_z_vectors:
                print(f"Warning: No valid z-vectors for position {position}, skipping")
                continue

            # Extract key vectors for valid targets
            print(f"Extracting key vectors...")
            key_vectors = self._extract_key_vectors_batch(
                mt=mt,
                targets=valid_targets,
                layer_idx=layer_idx,
                position=position,
            )

            # Compute weight updates
            weight_updates = self._compute_weight_update(
                mt=mt,
                layer_idx=layer_idx,
                z_vectors=valid_z_vectors,
                key_vectors=key_vectors,
            )

            # Accumulate updates
            for param_name, delta in weight_updates.items():
                if param_name not in all_weight_updates:
                    all_weight_updates[param_name] = delta
                else:
                    all_weight_updates[param_name] += delta

        # Apply accumulated weight updates
        print(f"\n{'=' * 70}")
        print(f"Applying Weight Updates")
        print(f"{'=' * 70}")

        with torch.no_grad():
            for param_name, delta in all_weight_updates.items():
                current_param = nethook.get_parameter(model, param_name)
                current_param.add_(delta)
                print(f"✓ Updated {param_name}")

        print(f"\n{'=' * 70}")
        print(f"DOME Edit Complete")
        print(f"{'=' * 70}\n")

        return mt

    def _extract_key_vectors_batch(
            self,
            mt: ModelAndTokenizer,
            targets: List[Dict],
            layer_idx: int,
            position: int,
    ) -> List[torch.Tensor]:
        """
        Extract key vectors (intermediate activations) for targets.

        Key vectors are the inputs to the target layer, which will be used
        in the DOME weight update equation.

        Args:
            mt: Model and tokenizer
            targets: List of edit targets
            layer_idx: Target layer index
            position: Current position

        Returns:
            List of key vector tensors
        """
        model = mt.model
        tokenizer = mt.tokenizer
        layer_name = f"decoder.block.{layer_idx}.layer.2.DenseReluDense"

        key_vectors = []

        for target in targets:
            query = target["query"]
            full_target_docids = target.get("full_target_docids", [])

            # Prepare inputs
            encoder_inputs = tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            decoder_input = self._build_decoder_input_with_context(
                mt=mt,
                query=query,
                position=position,
                full_target_docids=full_target_docids,
            )

            # Extract key vector (input to target layer)
            with torch.no_grad():
                with nethook.TraceDict(
                        module=model,
                        layers=[layer_name],
                        retain_input=True,
                        retain_output=False,
                ) as traces:
                    _ = model(
                        input_ids=encoder_inputs["input_ids"],
                        attention_mask=encoder_inputs["attention_mask"],
                        decoder_input_ids=decoder_input,
                    )

                # Get input to target layer
                layer_input = traces[layer_name].input

                if isinstance(layer_input, (tuple, list)):
                    key_vec = layer_input[0][:, -1, :].squeeze()
                else:
                    key_vec = layer_input[:, -1, :].squeeze()

                key_vectors.append(key_vec.cpu())

        return key_vectors

    def _get_layer_for_position(self, position: int) -> int:
        """
        Determine which layer to edit for a given DocID position.

        Strategy: Map positions to layers in a way that spreads edits
        across multiple layers to minimize interference.

        Args:
            position: DocID position (0-indexed)

        Returns:
            Layer index to edit
        """
        # Simple strategy: use later layers for later positions
        # T5-base has 12 decoder layers, use layers 6-11 for editing
        num_layers = 12
        editing_layers = list(range(6, num_layers))

        # Map position to layer cyclically
        layer_idx = editing_layers[position % len(editing_layers)]

        return layer_idx

    def restore_original_weights(self, mt: ModelAndTokenizer):
        """
        Restore model to original weights before editing.

        Args:
            mt: Model and tokenizer to restore
        """
        if not hasattr(self, 'original_weights'):
            print("Warning: No original weights saved")
            return

        print("Restoring original weights...")
        model = mt.model

        with torch.no_grad():
            for name, original_weight in self.original_weights.items():
                current_param = nethook.get_parameter(model, name)
                current_param.copy_(original_weight)

        print("✓ Weights restored")

    # ============================================================
    # EVALUATION AND VALIDATION
    # ============================================================

    def evaluate_edit_success(
            self,
            mt: ModelAndTokenizer,
            edit_targets: List[Dict],
    ) -> Dict[str, float]:
        """
        Evaluate success rate of applied edits.

        Args:
            mt: Edited model and tokenizer
            edit_targets: List of edit targets to validate

        Returns:
            Dictionary of evaluation metrics
        """
        model = mt.model
        tokenizer = mt.tokenizer

        total_targets = len(edit_targets)
        successful_edits = 0
        position_success = {}

        print(f"\n{'=' * 70}")
        print(f"Evaluating Edit Success")
        print(f"{'=' * 70}")

        for target in edit_targets:
            query = target["query"]
            target_docid = target["target_docid"]
            position = target["position"]
            full_target_docids = target.get("full_target_docids", [])

            # Build decoder input up to target position
            decoder_input = self._build_decoder_input_with_context(
                mt=mt,
                query=query,
                position=position,
                full_target_docids=full_target_docids,
            )

            # Encode query
            encoder_inputs = tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)

            # Get model prediction
            with torch.no_grad():
                outputs = model(
                    input_ids=encoder_inputs["input_ids"],
                    attention_mask=encoder_inputs["attention_mask"],
                    decoder_input_ids=decoder_input,
                )

            logits = outputs.logits[:, -1, :]
            predicted_token_id = torch.argmax(logits, dim=-1).item()

            # Check if prediction matches target
            target_token = f"${target_docid}$"
            target_token_id = tokenizer.convert_tokens_to_ids(target_token)

            if predicted_token_id == target_token_id:
                successful_edits += 1
                position_success[position] = position_success.get(position, 0) + 1

        success_rate = successful_edits / total_targets if total_targets > 0 else 0

        print(f"\nOverall Success Rate: {success_rate:.2%}")
        print(f"Successful: {successful_edits}/{total_targets}")

        print(f"\nSuccess by Position:")
        for pos in sorted(position_success.keys()):
            pos_total = sum(1 for t in edit_targets if t["position"] == pos)
            pos_rate = position_success[pos] / pos_total if pos_total > 0 else 0
            print(f"  Position {pos}: {pos_rate:.2%} ({position_success[pos]}/{pos_total})")

        print(f"{'=' * 70}\n")

        return {
            "success_rate": success_rate,
            "successful_edits": successful_edits,
            "total_targets": total_targets,
            "position_success": position_success,
        }


# ============================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================

def parse_arguments():
    """
    Parse command line arguments for T5-DOME editing.

    Returns:
        argparse.Namespace: Parsed arguments with all configuration parameters
    """
    parser = argparse.ArgumentParser(
        description="T5-DOME for DocID Editing in Generative Retrieval",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ========================================
    # Model Configuration
    # ========================================
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--model_name",
        type=str,
        default="../huggingface/t5-large",
        help="T5 model name or local path"
    )
    model_group.add_argument(
        "--gr_checkpoint",
        type=str,
        default="data/output/t5-large-rq-ckpts/34.pt",
        help="Checkpoint of the GR model trained by GR/training.py. Pass '' to skip "
             "(only for debugging - editing an untrained model is not a valid run)."
    )
    model_group.add_argument(
        "--id_num",
        type=int,
        default=128,
        help="Number of DocID tokens (e.g., $0$, $1$, ..., $127$)"
    )

    # ========================================
    # Data Files
    # ========================================
    data_group = parser.add_argument_group('Data Files')
    data_group.add_argument(
        "--cov_data_file",
        type=str,
        default="Edit/data/query2docid_full_rq.json",
        help="JSON file for covariance matrix computation"
    )
    data_group.add_argument(
        "--fact_file",
        type=str,
        default="Edit/data/query2docid2_pseudo_rq.json",
        help="JSON file with pseudo query-to-DocID mappings"
    )
    data_group.add_argument(
        "--doc_file",
        type=str,
        default="Edit/data/mem2docid2_rq.json",
        help="JSON file with document-to-DocID mappings"
    )
    data_group.add_argument(
        "--mem_file",
        type=str,
        default="Edit/data/query2docid2_rq.json",
        help="JSON file with memory query-to-DocID mappings"
    )

    # ========================================
    # Output Configuration
    # ========================================
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="results/t5-large-rq-edit-ckpts/",
        help="Base directory for saving results"
    )
    output_group.add_argument(
        "--num_cases",
        type=int,
        default=500,
        help="Number of document cases to process"
    )
    output_group.add_argument(
        "--checkpoint_id",
        type=int,
        default=4,
        help="Checkpoint identifier for organizing outputs"
    )

    # ========================================
    # Editing Configuration
    # ========================================
    edit_group = parser.add_argument_group('Editing Configuration')
    edit_group.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[14, 15, 16, 17, 18],
        help="List of decoder layer indices to edit"
    )
    edit_group.add_argument(
        "--pseudo_num",
        type=int,
        default=10,
        help="Number of pseudo examples per document"
    )

    # ========================================
    # Processing Configuration
    # ========================================
    proc_group = parser.add_argument_group('Processing Configuration')
    proc_group.add_argument(
        "--use_multiprocessing",
        action="store_true",
        default=True,
        help="Enable multiprocessing for parallel computation"
    )
    proc_group.add_argument(
        "--no_multiprocessing",
        dest="use_multiprocessing",
        action="store_false",
        help="Disable multiprocessing (use single process)"
    )

    # ========================================
    # Cache Management
    # ========================================
    add_cache_management_args(parser)

    return parser.parse_args()


def add_cache_management_args(parser):
    """
    Add cache management arguments to parser.

    Args:
        parser: ArgumentParser instance
    """
    cache_group = parser.add_argument_group('Cache Management')
    cache_group.add_argument(
        "--covariance_cache_dir",
        type=str,
        default="cache/covariance",
        help="Directory for caching covariance matrices"
    )
    cache_group.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear all cached covariance matrices and exit"
    )
    cache_group.add_argument(
        "--list_cache",
        action="store_true",
        help="List all cached covariance matrices and exit"
    )


# ============================================================
# INITIALIZATION FUNCTIONS
# ============================================================

def initialize_multiprocessing():
    """
    Initialize multiprocessing with spawn method for CUDA compatibility.

    The spawn method is necessary when using CUDA in multiprocessing
    to avoid issues with CUDA context inheritance.
    """
    try:
        mp.set_start_method('spawn', force=True)
        print("✓ Multiprocessing initialized with 'spawn' method")
    except RuntimeError as e:
        # Already set, this is fine
        print(f"Multiprocessing method already set: {e}")


def initialize_model(args):
    """
    Initialize T5 model and tokenizer with DocID token support.

    Args:
        args: Parsed command line arguments

    Returns:
        ModelAndTokenizer: Initialized model wrapper
    """
    print(f"\n{'=' * 70}")
    print(f"Initializing Model")
    print(f"{'=' * 70}")
    print(f"Model: {args.model_name}")
    print(f"DocID tokens: {args.id_num}")

    mt = ModelAndTokenizer(
        model_name=args.model_name,
        id_num=args.id_num,
        epoch_path=args.gr_checkpoint or None
    )

    print(f"✓ Model loaded successfully")
    print(f"  - Decoder layers: {mt.num_decoder_layers}")
    print(f"  - Vocabulary size: {len(mt.tokenizer)}")
    print(f"{'=' * 70}\n")

    return mt


def initialize_hyperparameters(args):
    """
    Initialize hyperparameters for T5-DOME editing.

    Args:
        args: Parsed command line arguments

    Returns:
        T5DocIDEditHyperParams: Configured hyperparameters
    """
    print(f"\n{'=' * 70}")
    print(f"Initializing Hyperparameters")
    print(f"{'=' * 70}")

    hparams = T5DocIDEditHyperParams(
        layers=args.layers,
        pseudo_num=args.pseudo_num,
        covariance_cache_dir=args.covariance_cache_dir
    )

    print(f"Editing layers: {hparams.layers}")
    print(f"Pseudo examples per doc: {hparams.pseudo_num}")
    print(f"Learning rate: {hparams.v_lr}")
    print(f"Gradient steps: {hparams.v_num_grad_steps}")
    print(f"{'=' * 70}\n")

    return hparams


# ============================================================
# CACHE MANAGEMENT
# ============================================================

def handle_cache_management(args):
    """
    Handle cache management operations (clear/list) and exit.

    Args:
        args: Parsed command line arguments
    """
    print(f"\n{'=' * 70}")
    print(f"Cache Management")
    print(f"{'=' * 70}\n")

    # Create temporary hyperparameters for cache manager
    hparams = T5DocIDEditHyperParams(
        layers=args.layers,
        covariance_cache_dir=args.covariance_cache_dir
    )
    editor = T5DocIDDOME(hparams)

    if args.clear_cache:
        print("Clearing all covariance cache...")
        editor.covariance_manager.clear_cache()
        print("✓ Cache cleared successfully!\n")

    if args.list_cache:
        print("Listing covariance cache...")
        editor.covariance_manager.list_cache()
        print()


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def load_edit_requests(file_path: str) -> List[Dict]:
    """
    Load edit requests from JSON file in knowns format.

    Expected format:
    {
        "prompt": "query text",
        "attribute": "$0$ $5$ $12$",
        "known_id": 1
    }

    Args:
        file_path: Path to JSON file

    Returns:
        List of edit request dictionaries with keys:
        - query: Query text
        - target_docids: List of DocID strings
        - case_id: Unique identifier
    """
    print(f"\n{'=' * 70}")
    print(f"Loading Edit Requests")
    print(f"{'=' * 70}")
    print(f"Source file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load knowns dataset
    knowns = KnownsDataset(file_path)
    edit_requests = []

    # Regular expression to extract DocIDs in format "$N$"
    docid_pattern = r'\$(\d+)\$'

    # Convert knowns format to edit request format
    for item in knowns:
        query = item.get("prompt", "").strip()
        attribute = item.get("attribute", "").strip()

        if not query or not attribute:
            continue

        # Extract DocID sequence from attribute string
        docids = re.findall(docid_pattern, attribute)

        if docids:
            edit_requests.append({
                "query": query,
                "target_docids": docids,
                "case_id": str(item.get("known_id", len(edit_requests)))
            })

    print(f"✓ Loaded {len(edit_requests)} edit requests")
    print(f"{'=' * 70}\n")

    return edit_requests


def prepare_test_requests(
        edit_requests: List[Dict],
        doc_edit_requests: List[Dict],
        num_cases: int,
        pseudo_num: int
) -> List[Dict]:
    """
    Prepare test requests by interleaving document and pseudo requests.

    Structure:
    - For each document (num_cases total):
        1. Add document request
        2. Add pseudo_num pseudo examples

    This creates a pattern: [doc1, pseudo1_1, ..., pseudo1_10, doc2, pseudo2_1, ...]

    Args:
        edit_requests: List of pseudo edit requests
        doc_edit_requests: List of document edit requests
        num_cases: Number of documents to process
        pseudo_num: Number of pseudo examples per document

    Returns:
        Combined list of test requests
    """
    print(f"\n{'=' * 70}")
    print(f"Preparing Test Requests")
    print(f"{'=' * 70}")
    print(f"Documents: {num_cases}")
    print(f"Pseudo examples per document: {pseudo_num}")

    test_requests = []

    # Calculate total pseudo requests needed
    total_pseudo_needed = num_cases * pseudo_num
    pseudo_subset = edit_requests[:total_pseudo_needed]

    # Interleave document and pseudo requests
    for i in range(0, len(pseudo_subset), pseudo_num):
        doc_index = i // pseudo_num

        # Add document request if available
        if doc_index < len(doc_edit_requests):
            test_requests.append(doc_edit_requests[doc_index])

        # Add pseudo examples for this document
        pseudo_batch = pseudo_subset[i:i + pseudo_num]
        test_requests.extend(pseudo_batch)

    print(f"✓ Prepared {len(test_requests)} total test requests")
    print(f"  - Document requests: {min(num_cases, len(doc_edit_requests))}")
    print(f"  - Pseudo requests: {len(pseudo_subset)}")
    print(f"{'=' * 70}\n")

    return test_requests


# ============================================================
# SAVING AND OUTPUT FUNCTIONS
# ============================================================

def save_edited_model(
        edited_model: torch.nn.Module,
        hparams: T5DocIDEditHyperParams,
        args,
        elapsed_time: float
):
    """
    Save edited model, hyperparameters, and metadata to disk.

    Creates directory structure:
    output_dir/
        {num_cases}cases/
            {checkpoint_id}/
                edited_t5_docid.pt       # Model weights
                t5_docid_edit_hparams.json  # Hyperparameters
                edit_metadata.json       # Metadata

    Args:
        edited_model: The edited PyTorch model
        hparams: Hyperparameters used for editing
        args: Command line arguments
        elapsed_time: Total editing time in seconds
    """
    print(f"\n{'=' * 70}")
    print(f"Saving Edited Model")
    print(f"{'=' * 70}")

    # Construct output directory path
    output_dir = Path(args.output_dir) / f"{args.num_cases}cases" / str(args.checkpoint_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # ========================================
    # Save Model Weights
    # ========================================
    model_path = output_dir / "edited_t5_docid.pt"
    print(f"Saving model weights to: {model_path}")
    torch.save(edited_model.state_dict(), model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✓ Model saved ({model_size_mb:.1f} MB)")

    # ========================================
    # Save Hyperparameters
    # ========================================
    hparams_path = output_dir / "t5_docid_edit_hparams.json"
    print(f"Saving hyperparameters to: {hparams_path}")
    hparams.save(str(hparams_path))
    print(f"✓ Hyperparameters saved")

    # ========================================
    # Save Metadata
    # ========================================
    metadata = {
        "model_name": args.model_name,
        "num_cases": args.num_cases,
        "pseudo_num": args.pseudo_num,
        "layers_edited": args.layers,
        "id_num": args.id_num,
        "editing_time_seconds": elapsed_time,
        "editing_time_formatted": f"{elapsed_time:.2f}s",
        "timestamp": str(pd.Timestamp.now()) if 'pandas' in sys.modules else time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_files": {
            "covariance": args.cov_data_file,
            "facts": args.fact_file,
            "documents": args.doc_file,
            "memory": args.mem_file,
        },
        "multiprocessing_enabled": args.use_multiprocessing,
    }

    metadata_path = output_dir / "edit_metadata.json"
    print(f"Saving metadata to: {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved")

    print(f"\n{'=' * 70}")
    print(f"All outputs saved to: {output_dir}")
    print(f"{'=' * 70}\n")

# ============================================================
# SHARED PIPELINE
# ============================================================

def run(
        editor_cls: type,
        apply_edit: Callable[[Any, ModelAndTokenizer, List[Dict], Any], ModelAndTokenizer],
        banner: str,
        done_banner: str,
) -> None:
    """The editing pipeline both DOME and MOSE run.

    Every step here is identical for the two methods: parse args, init
    multiprocessing, handle cache, load the model and hyperparameters, load and
    interleave the edit requests, then save and report. Only stage 7 differs,
    which is why it arrives as `apply_edit` rather than being written twice.

    Args:
        editor_cls: Editor class to instantiate with the hyperparameters.
        apply_edit: Stage 7. Receives (editor, mt, test_requests, args) and
            returns the edited ModelAndTokenizer.
        banner: Title printed at start.
        done_banner: Title printed on completion.
    """
    start_time: float = time.time()

    print(f"\n{'#' * 70}")
    print(f"#{banner.center(68)}#")
    print(f"#{'=' * 68}#")
    print(f"{'#' * 70}\n")

    args: Any = parse_arguments()
    initialize_multiprocessing()

    if args.clear_cache or args.list_cache:
        handle_cache_management(args)
        return

    mt: ModelAndTokenizer = initialize_model(args)
    hparams: T5DocIDEditHyperParams = initialize_hyperparameters(args)

    print(f"{'=' * 70}")
    print(f"Loading Data Files")
    print(f"{'=' * 70}\n")

    edit_requests: List[Dict] = load_edit_requests(args.fact_file)
    doc_edit_requests: List[Dict] = load_edit_requests(args.doc_file)

    print(f"Data loading summary:")
    print(f"  - Pseudo examples: {len(edit_requests)}")
    print(f"  - Document examples: {len(doc_edit_requests)}")
    print(f"{'=' * 70}\n")

    test_requests: List[Dict] = prepare_test_requests(
        edit_requests=edit_requests,
        doc_edit_requests=doc_edit_requests,
        num_cases=args.num_cases,
        pseudo_num=args.pseudo_num
    )

    print(f"\n{'=' * 70}")
    print(f"{banner}")
    print(f"{'=' * 70}\n")

    editor: Any = editor_cls(hparams)
    edited_model: ModelAndTokenizer = apply_edit(editor, mt, test_requests, args)

    elapsed_time: float = time.time() - start_time

    save_edited_model(
        edited_model=edited_model,
        hparams=hparams,
        args=args,
        elapsed_time=elapsed_time
    )

    print(f"\n{'#' * 70}")
    print(f"#{done_banner.center(68)}#")
    print(f"#{'=' * 68}#")
    print(f"#  Total time: {elapsed_time:.2f}s{' ' * (55 - len(f'{elapsed_time:.2f}s'))}#")
    print(f"#  Cases processed: {args.num_cases}{' ' * (51 - len(str(args.num_cases)))}#")
    print(f"#  Layers edited: {len(args.layers)}{' ' * (53 - len(str(len(args.layers))))}#")
    print(f"{'#' * 70}\n")
