"""MOSE: multiplicative orthogonal DocID editing.

Everything MOSE shares with DOME lives in `common`. This file holds only what
is unique to MOSE: seven methods that replace DOME's additive stage 3 with an
orthogonal Procrustes rotation, and the two lines that wire them into the
shared pipeline.

The two MOSE hyperparameters, `preserve_lambda` and `mose_svd_reg`, live on
`T5DocIDEditHyperParams` in `common` because the shared argument parser
constructs it.

Read this file top to bottom to read the contribution. Nothing here is DOME.
"""

import os
import re
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from util import nethook
from dsets import KnownsDataset

from common import ModelAndTokenizer, T5DocIDDOME, run


class T5DocIDMOSE(T5DocIDDOME):
    """DOME's editor with stage 3 replaced by a multiplicative orthogonal update.

    Stages 1 and 2 are inherited unchanged: group targets by DocID position,
    pick the layer per position, optimize z-vectors with the hybrid
    soft/hard-label strategy. Only the weight update differs.
    """
    # ============================================================
    # MOSE WEIGHT UPDATE (multiplicative orthogonal edit)
    # ============================================================

    def _mose_module_name(self, layer_idx: int) -> str:
        """Name of the linear map MOSE rotates: the FFN output projection.

        MOSE edits one genuine linear map, the one `decoder_module_tmp`
        already names (`...DenseReluDense.wo`). Its keys live in `d_ff` and
        its outputs live in `d_model`, which is the space the z-vectors from
        `_compute_z_vectors_batch` are optimized in. `wi` is not touched: a
        rotation is only defined for the map whose keys and values MOSE has.
        """
        return self.hparams.decoder_module_tmp.format(layer_idx)

    def _extract_mose_key_vectors_batch(
            self,
            mt: ModelAndTokenizer,
            targets: List[Dict],
            layer_idx: int,
            position: int,
    ) -> List[torch.Tensor]:
        """Extract MOSE keys: the inputs to `wo`, i.e. post-activation d_ff.

        Differs from `_extract_key_vectors_batch`, which traces the whole
        `DenseReluDense` block and so returns d_model vectors. Those are the
        block's input, not the keys of any single linear map. MOSE needs
        k such that `W0 @ k` is the quantity the z-vector replaces.

        Args:
            mt: Model and tokenizer
            targets: Edit targets sharing this position
            layer_idx: Decoder block being edited
            position: DocID position being edited

        Returns:
            One d_ff key tensor per target, on CPU.
        """
        model: Any = mt.model
        tokenizer: Any = mt.tokenizer
        module_name: str = self._mose_module_name(layer_idx)
        key_vectors: List[torch.Tensor] = []

        target: Dict
        for target in targets:
            query: str = target["query"]
            full_target_docids: List = target.get("full_target_docids", [])

            encoder_inputs: Any = tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            decoder_input: torch.Tensor = self._build_decoder_input_with_context(
                mt=mt,
                query=query,
                position=position,
                full_target_docids=full_target_docids,
            )

            with torch.no_grad():
                traces: Any
                with nethook.TraceDict(
                        module=model,
                        layers=[module_name],
                        retain_input=True,
                        retain_output=False,
                ) as traces:
                    _out: Any = model(
                        input_ids=encoder_inputs["input_ids"],
                        attention_mask=encoder_inputs["attention_mask"],
                        decoder_input_ids=decoder_input,
                    )

                layer_input: Any = traces[module_name].input
                key_vec: torch.Tensor
                if isinstance(layer_input, (tuple, list)):
                    key_vec = layer_input[0][:, -1, :].squeeze()
                else:
                    key_vec = layer_input[:, -1, :].squeeze()

                key_vectors.append(key_vec.detach().float().cpu())

        return key_vectors

    def _load_mose_docid_data(self, cov_data_file: str, mt: ModelAndTokenizer) -> List[Dict]:
        """Parse a knowns-format file into query -> docID-sequence pairs.

        Same extraction `_compute_covariance_data` performs, kept separate so
        the MOSE path does not inherit that method's d_model accumulator.

        Args:
            cov_data_file: Path to the knowns JSON file (the preserved corpus)
            mt: Model and tokenizer, for the docID vocabulary bound

        Returns:
            List of {"query", "docids"} dicts.
        """
        if not os.path.exists(cov_data_file):
            raise FileNotFoundError(f"[MOSE] covariance data file not found: {cov_data_file}")

        knowns_dataset: Any = KnownsDataset(cov_data_file)
        docid_pattern: str = r'\$(\d+)\$'
        docid_data: List[Dict] = []

        item: Any
        for item in knowns_dataset:
            query: str = item.get("prompt", "").strip()
            attribute: str = item.get("attribute", "").strip()
            if not query or not attribute:
                continue

            docids: List[str] = re.findall(docid_pattern, attribute)
            valid_docids: List[str] = []
            docid: str
            for docid in docids:
                if docid.isdigit() and int(docid) < mt.id_num:
                    valid_docids.append(docid)

            if valid_docids:
                docid_data.append({"query": query, "docids": valid_docids})

        if not docid_data:
            raise ValueError(f"[MOSE] no valid docID data found in {cov_data_file}")

        print(f"[MOSE] parsed {len(docid_data)} query->docID pairs from {cov_data_file}")
        return docid_data

    def _collect_mose_preservation_keys(
            self,
            mt: ModelAndTokenizer,
            module_name: str,
            docid_data: List[Dict],
            sample_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Accumulate C = E[k kT] over the preserved corpus, in key space.

        Teacher-forces each preserved query's docID sequence and records the
        input to `module_name` at every docID position. The accumulator is
        sized to `d_ff` because that is the dimension of a `wo` key; the DOME
        path sizes its accumulator to `d_model`, which does not match the
        [3072, 3072] shape its own log line announces.

        Args:
            mt: Model and tokenizer
            module_name: Module whose input is the key (the `wo` projection)
            docid_data: Preserved query -> docID pairs
            sample_size: Cap on queries used; None uses all

        Returns:
            [d_ff, d_ff] uncentered second moment, on CPU in float32.
        """
        try:
            from .layer_stats import CombinedStat
        except ImportError:  # run as a top-level script: python MOSE.py
            from layer_stats import CombinedStat

        model: Any = mt.model
        tokenizer: Any = mt.tokenizer
        key_dim: int = model.config.d_ff

        sampled_data: List[Dict] = docid_data
        if sample_size is not None and sample_size < len(docid_data):
            sampled_indices: List[int] = torch.randperm(len(docid_data))[:sample_size].tolist()
            sampled_data = [docid_data[i] for i in sampled_indices]

        print(f"[MOSE] collecting preservation keys at {module_name} (dim {key_dim})")
        print(f"[MOSE] using {len(sampled_data)} preserved queries")

        stat: Any = CombinedStat(key_dim)
        total_keys: int = 0

        data_item: Dict
        for data_item in tqdm(sampled_data, desc="[MOSE] preservation keys"):
            query: str = data_item["query"]
            docids: List[str] = data_item["docids"]

            encoder_inputs: Any = tokenizer(
                query,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            pos: int
            for pos in range(len(docids)):
                decoder_tokens: List[int] = [
                    model.config.decoder_start_token_id or tokenizer.pad_token_id
                ]
                j: int
                for j in range(pos):
                    docid_token: str = f"${docids[j]}$"
                    docid_id: int = tokenizer.convert_tokens_to_ids(docid_token)
                    if docid_id == tokenizer.unk_token_id:
                        docid_id = mt.docid_token_ids[0]
                    decoder_tokens.append(docid_id)

                decoder_input: torch.Tensor = torch.tensor(
                    [decoder_tokens], device=model.device
                )

                with torch.no_grad():
                    traces: Any
                    with nethook.TraceDict(
                            module=model,
                            layers=[module_name],
                            retain_input=True,
                            retain_output=False,
                    ) as traces:
                        _out: Any = model(
                            input_ids=encoder_inputs["input_ids"],
                            attention_mask=encoder_inputs["attention_mask"],
                            decoder_input_ids=decoder_input,
                        )

                    layer_input: Any = traces[module_name].input
                    key_vec: torch.Tensor
                    if isinstance(layer_input, (tuple, list)):
                        key_vec = layer_input[0][:, -1, :].squeeze()
                    else:
                        key_vec = layer_input[:, -1, :].squeeze()

                    stat.add(key_vec.float())
                    total_keys += 1

        print(f"[MOSE] accumulated {total_keys} preservation keys")
        second_moment: torch.Tensor = stat.mom2.second_moment().float().cpu()
        print(f"[MOSE] C shape {tuple(second_moment.shape)}")
        return second_moment

    def _get_mose_covariance(
            self,
            mt: ModelAndTokenizer,
            layer_idx: int,
            cov_data_file: str,
    ) -> torch.Tensor:
        """Load or compute the key-space second moment for one layer.

        Cached through the existing `CovarianceManager`. The cache key carries
        the `wo` module name, so MOSE statistics never collide with DOME's.
        """
        module_name: str = self._mose_module_name(layer_idx)

        if layer_idx in self.cov_cache:
            return self.cov_cache[layer_idx]

        model_name: str = getattr(mt.model.config, "_name_or_path", "unknown_model")
        cache_key: str = self.covariance_manager._generate_cache_key(
            model_name=model_name,
            module_name=module_name,
            cov_data_file=cov_data_file,
            sample_size=self.hparams.mom2_n_samples,
            dtype=self.hparams.mom2_dtype,
        )

        cached_matrix: Optional[torch.Tensor] = self.covariance_manager.load_covariance(cache_key)
        if cached_matrix is not None:
            print(f"[MOSE] covariance cache hit for {module_name}: {tuple(cached_matrix.shape)}")
            self.cov_cache[layer_idx] = cached_matrix
            return cached_matrix

        print(f"[MOSE] covariance cache miss for {module_name}, computing")
        docid_data: List[Dict] = self._load_mose_docid_data(cov_data_file, mt)
        second_moment: torch.Tensor = self._collect_mose_preservation_keys(
            mt=mt,
            module_name=module_name,
            docid_data=docid_data,
            sample_size=self.hparams.mom2_n_samples,
        )

        metadata: Dict = {
            "model_name": model_name,
            "module_name": module_name,
            "data_file": cov_data_file,
            "sample_size": self.hparams.mom2_n_samples,
            "dtype": self.hparams.mom2_dtype,
            "space": "wo_input_d_ff",
            "statistic": "uncentered_second_moment",
        }
        self.covariance_manager.save_covariance(second_moment, cache_key, metadata)
        self.cov_cache[layer_idx] = second_moment
        return second_moment

    def _compute_mose_rotation(
            self,
            mt: ModelAndTokenizer,
            layer_idx: int,
            z_vectors: List[torch.Tensor],
            key_vectors: List[torch.Tensor],
            cov: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Solve MOSE's orthogonal Procrustes problem for one layer.

        The update is multiplicative, not additive. With W0 the current
        projection, K_E the edit keys, V_E the target values and C the
        preserved-key second moment:

            M = lam * (W0 C W0^T) + V_E K_E^T W0^T
            R = U V^T,   where  M = U S V^T
            W_new = R W0

        R is the orthogonal matrix maximizing tr(R^T M), i.e. the nearest
        orthogonal matrix to M. Written as a stacked alignment the two terms
        are the two column blocks of a single Procrustes problem:

            A = [sqrt(lam) W0 K0 | W0 K_E],  B = [sqrt(lam) W0 K0 | V_E]

        since B A^T = lam W0 (K0 K0^T) W0^T + V_E K_E^T W0^T, and
        C = K0 K0^T / n with the count folded into lam. The preserved block
        pins R to the identity in directions the model actually uses; the edit
        block turns it where the new docIDs need it.

        Because R is orthogonal, no singular value of W is amplified, which is
        the property that is supposed to hold up over a stream of rounds where
        an additive delta accumulates without bound.

        Args:
            mt: Model and tokenizer
            layer_idx: Decoder block being edited
            z_vectors: Desired outputs, d_model each
            key_vectors: Edit keys, d_ff each
            cov: [d_ff, d_ff] preserved-key second moment

        Returns:
            {weight_name: W_new} holding the full new weight, not a delta.
        """
        model: Any = mt.model
        module_name: str = self._mose_module_name(layer_idx)
        weight_name: str = f"{module_name}.weight"

        W_param: torch.Tensor = nethook.get_parameter(model, weight_name)
        device: Any = W_param.device
        W0: torch.Tensor = W_param.detach().to(device=device, dtype=torch.float32)

        K_E: torch.Tensor = torch.stack(
            [k.flatten() for k in key_vectors], dim=1
        ).to(device=device, dtype=torch.float32)
        V_E: torch.Tensor = torch.stack(
            [z.flatten() for z in z_vectors], dim=1
        ).to(device=device, dtype=torch.float32)
        C: torch.Tensor = cov.to(device=device, dtype=torch.float32)

        print(f"\n{'=' * 60}")
        print(f"[MOSE] rotating {weight_name}")
        print(f"{'=' * 60}")
        print(f"[MOSE] W0 {tuple(W0.shape)}  K_E {tuple(K_E.shape)}  V_E {tuple(V_E.shape)}  C {tuple(C.shape)}")

        if W0.shape[1] != K_E.shape[0]:
            raise RuntimeError(
                f"[MOSE] key dim {K_E.shape[0]} does not match {weight_name} "
                f"input dim {W0.shape[1]}; keys must be inputs to this projection"
            )
        if W0.shape[0] != V_E.shape[0]:
            raise RuntimeError(
                f"[MOSE] z dim {V_E.shape[0]} does not match {weight_name} "
                f"output dim {W0.shape[0]}"
            )
        if C.shape[0] != W0.shape[1] or C.shape[1] != W0.shape[1]:
            raise RuntimeError(
                f"[MOSE] covariance {tuple(C.shape)} is not in key space "
                f"[{W0.shape[1]}, {W0.shape[1]}]"
            )

        lam: float = float(self.hparams.preserve_lambda) * float(self.hparams.mom2_update_weight)
        preserve_term: torch.Tensor = lam * (W0 @ C @ W0.T)
        edit_term: torch.Tensor = V_E @ K_E.T @ W0.T
        M: torch.Tensor = preserve_term + edit_term

        print(f"[MOSE] lam {lam:.4g}  ||preserve|| {torch.linalg.matrix_norm(preserve_term).item():.4g}"
              f"  ||edit|| {torch.linalg.matrix_norm(edit_term).item():.4g}")

        if self.hparams.mose_svd_reg > 0:
            M = M + self.hparams.mose_svd_reg * torch.eye(
                M.shape[0], device=device, dtype=M.dtype
            )

        U: torch.Tensor
        S: torch.Tensor
        Vh: torch.Tensor
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        R: torch.Tensor = U @ Vh
        W_new: torch.Tensor = R @ W0

        eye: torch.Tensor = torch.eye(R.shape[0], device=device, dtype=R.dtype)
        before: float = torch.linalg.matrix_norm(W0 @ K_E - V_E).item()
        after: float = torch.linalg.matrix_norm(W_new @ K_E - V_E).item()
        print(f"[MOSE] ||R - I|| {torch.linalg.matrix_norm(R - eye).item():.6f}"
              f"  orthogonality err {torch.linalg.matrix_norm(R.T @ R - eye).item():.2e}"
              f"  det sign {torch.sign(torch.linalg.det(R)).item():+.0f}")
        print(f"[MOSE] edit residual {before:.4f} -> {after:.4f}")
        print(f"[MOSE] smallest singular value of M {S[-1].item():.4g}")
        print(f"{'=' * 60}\n")

        return {weight_name: W_new.to(W_param.dtype)}

    def apply_mose_edit(
            self,
            mt: ModelAndTokenizer,
            edit_targets: List[Dict],
            cov_data_file: str,
            keep_original_weights: bool = True,
    ) -> ModelAndTokenizer:
        """Apply MOSE edits to the model.

        Stages 1 and 2 are DOME's, unchanged: group targets by docID position,
        pick the layer per position, optimize z-vectors with the hybrid
        soft/hard-label strategy. Stage 3 is replaced by the multiplicative
        orthogonal update.

        Each position group is written into the model as soon as its rotation
        is solved, rather than accumulated and applied at the end the way
        `apply_dome_edit` does. Rotations compose multiplicatively, so summing
        them would leave the product non-orthogonal and throw away the
        property the method exists for. Writing in sequence also means a later
        group that lands on the same layer solves against the already-rotated
        W0, which is how MOSE is specified for sequential edits.

        `cov_data_file` is an explicit argument rather than a hyperparameter
        because `T5DocIDEditHyperParams` has no `covariance_data_file` field;
        for the streaming diagnostic it stays pinned to the initial corpus.

        Args:
            mt: Model and tokenizer to edit
            edit_targets: Edit specifications
            cov_data_file: Knowns file defining the preserved corpus
            keep_original_weights: Save pre-edit FFN weights for restore

        Returns:
            The edited ModelAndTokenizer.
        """
        model: Any = mt.model

        print(f"\n{'=' * 70}")
        print(f"[MOSE] starting edit process")
        print(f"{'=' * 70}")
        print(f"[MOSE] total edit targets: {len(edit_targets)}")
        print(f"[MOSE] preserve_lambda {self.hparams.preserve_lambda} "
              f"mom2_update_weight {self.hparams.mom2_update_weight} "
              f"svd_reg {self.hparams.mose_svd_reg}")

        if keep_original_weights:
            print("[MOSE] saving original model weights")
            self.original_weights = {}
            name: str
            param: torch.Tensor
            for name, param in model.named_parameters():
                if "DenseReluDense" in name:
                    self.original_weights[name] = param.data.clone()

        position_groups: Dict[int, List[Dict]] = self._group_targets_by_position(edit_targets)
        print(f"[MOSE] grouped into {len(position_groups)} position groups")

        rotated_layers: List[int] = []

        position: int
        targets: List[Dict]
        for position, targets in sorted(position_groups.items()):
            print(f"\n{'=' * 70}")
            print(f"[MOSE] position {position} ({len(targets)} targets)")
            print(f"{'=' * 70}")

            layer_idx: int = self._get_layer_for_position(position)
            print(f"[MOSE] target layer: {layer_idx}")

            z_vectors: List[Optional[torch.Tensor]]
            failed_indices: List[int]
            z_vectors, failed_indices = self._compute_z_vectors_batch(
                mt=mt,
                targets=targets,
                target_layer=layer_idx,
                position=position,
            )

            valid_targets: List[Dict] = []
            valid_z_vectors: List[torch.Tensor] = []
            i: int
            z: Optional[torch.Tensor]
            for i, z in enumerate(z_vectors):
                if z is not None:
                    valid_targets.append(targets[i])
                    valid_z_vectors.append(z)

            print(f"[MOSE] {len(valid_z_vectors)}/{len(targets)} z-vectors computed")

            if not valid_z_vectors:
                print(f"[MOSE] no valid z-vectors for position {position}, skipping")
                continue

            print(f"[MOSE] extracting key vectors")
            key_vectors: List[torch.Tensor] = self._extract_mose_key_vectors_batch(
                mt=mt,
                targets=valid_targets,
                layer_idx=layer_idx,
                position=position,
            )

            cov: torch.Tensor = self._get_mose_covariance(
                mt=mt,
                layer_idx=layer_idx,
                cov_data_file=cov_data_file,
            )

            new_weights: Dict[str, torch.Tensor] = self._compute_mose_rotation(
                mt=mt,
                layer_idx=layer_idx,
                z_vectors=valid_z_vectors,
                key_vectors=key_vectors,
                cov=cov,
            )

            with torch.no_grad():
                param_name: str
                new_weight: torch.Tensor
                for param_name, new_weight in new_weights.items():
                    current_param: torch.Tensor = nethook.get_parameter(model, param_name)
                    current_param.copy_(new_weight.to(current_param.device))
                    print(f"[MOSE] wrote {param_name}")

            rotated_layers.append(layer_idx)

        print(f"\n{'=' * 70}")
        print(f"[MOSE] edit complete, rotated layers {rotated_layers}")
        print(f"{'=' * 70}\n")

        return mt


# ============================================================
# ENTRY POINT
# ============================================================

def apply_mose(
        editor: T5DocIDMOSE,
        mt: ModelAndTokenizer,
        requests: List[Dict],
        args: Any,
) -> ModelAndTokenizer:
    """Stage 7 for MOSE: rotate each edited layer instead of adding a delta.

    `cov_data_file` is passed through because the preserved-key second moment
    is defined by the initial corpus, not by the edit batch.
    """
    return editor.apply_mose_edit(
        mt=mt,
        edit_targets=requests,
        cov_data_file=args.cov_data_file,
    )


if __name__ == "__main__":
    run(
        editor_cls=T5DocIDMOSE,
        apply_edit=apply_mose,
        banner="T5-MOSE: Orthogonal DocID Editing for Generative Retrieval",
        done_banner="MOSE EDITING COMPLETE",
    )
