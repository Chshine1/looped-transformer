import random
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch.nn import functional

from looped_transformer.config import ModelOptions, RuntimeOptions
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.tokenizer import MNLITokenizer
from looped_transformer.types.data import MNLIBatch
from looped_transformer.types.mnli import get_logic_relation
from looped_transformer.types.model import ModelForwardOutput
from looped_transformer.utils.torch import resolve_device
from looped_transformer.utils.torch import resolve_dtype


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    raw_cross_entropy: float
    smooth_cross_entropy: float
    accuracy: float


TimeReduction = Literal["last", "uniform", "max_confidence", "min_entropy"]


@dataclass
class BatchDiagnostics:
    batch_index: int
    batch_size: int

    official_raw_ce: float
    official_smooth_ce: float
    official_acc: float

    last_raw_ce: float
    last_smooth_ce: float
    last_acc: float
    uniform_raw_ce: float
    uniform_smooth_ce: float
    uniform_acc: float

    oracle_raw_ce: float
    oracle_smooth_ce: float
    oracle_acc: float
    shuffled_oracle_raw_ce: float
    shuffled_oracle_smooth_ce: float
    shuffled_oracle_acc: float

    per_t_raw_ce: list[float]
    per_t_smooth_ce: list[float]
    per_t_acc: list[float]

    best_t_hist: list[int]

    gain_acc_mean: float
    gain_acc_median: float
    gain_acc_min: float
    gain_acc_max: float
    gain_acc_positive_frac: float
    gain_ce_mean: float
    gain_ce_median: float
    gain_ce_positive_frac: float

    target_hist: list[int]
    pred_hist_last: list[int]
    pred_hist_official: list[int]


class EvaluationPolicy:
    def __init__(
        self,
            max_iterations: int,
        device: torch.device,
        label_smoothing: float,
        official_time_reduction: TimeReduction = "last",
        shuffle_oracle_repeats: int = 3,
        verbose: bool = True,
    ):
        self._max_iterations = max_iterations
        self._device = device
        self._label_smoothing = label_smoothing
        self._official_time_reduction = official_time_reduction
        self._shuffle_oracle_repeats = shuffle_oracle_repeats
        self._verbose = verbose

        self._batch_index = 0
        self._total_count = 0
        self._history: list[BatchDiagnostics] = []

        self._official_raw_ce_avg = 0.0
        self._official_smooth_ce_avg = 0.0
        self._official_correct = 0

        self._per_t_raw_ce_sum = torch.zeros(max_iterations, device=self._device)
        self._per_t_smooth_ce_sum = torch.zeros(max_iterations, device=self._device)
        self._per_t_acc_sum = torch.zeros(max_iterations, device=self._device)
        self._per_t_sample_count = 0

        self._best_t_hist_sum = torch.zeros(max_iterations, dtype=torch.long, device=self._device)

        self._last_raw_ce_avg = 0.0
        self._last_smooth_ce_avg = 0.0
        self._last_correct = 0

        self._uniform_raw_ce_avg = 0.0
        self._uniform_smooth_ce_avg = 0.0
        self._uniform_correct = 0.0

        self._oracle_raw_ce_avg = 0.0
        self._oracle_smooth_ce_avg = 0.0
        self._oracle_correct = 0.0

        self._shuffled_oracle_raw_ce_avg = 0.0
        self._shuffled_oracle_smooth_ce_avg = 0.0
        self._shuffled_oracle_correct = 0.0

        self._gain_acc_sum = 0.0
        self._gain_ce_sum = 0.0
        self._gain_acc_positive = 0
        self._gain_ce_positive = 0

        self._batch_acc_list: list[float] = []

        if official_time_reduction not in {"last", "uniform"}:
            raise ValueError(
                "official_time_reduction must be 'last' or 'uniform' "
                "(max_confidence/min_entropy are diagnostics only)"
            )

    def reset(self) -> None:
        self.__init__(
            max_iterations=self._max_iterations,
            device=self._device,
            label_smoothing=self._label_smoothing,
            official_time_reduction=self._official_time_reduction,
            shuffle_oracle_repeats=self._shuffle_oracle_repeats,
            verbose=self._verbose,
        )

    @staticmethod
    def _per_sample_ce_by_time(
        timed_logits: torch.Tensor,
        label_tensor: torch.Tensor,
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        T, B, C = timed_logits.shape
        flat_logits = timed_logits.reshape(T * B, C)
        flat_target = label_tensor.unsqueeze(0).expand(T, B).reshape(T * B)
        ce = functional.cross_entropy(
            flat_logits,
            flat_target,
            reduction="none",
            label_smoothing=label_smoothing,
        )
        return ce.reshape(T, B)

    @staticmethod
    def _per_t_metrics(
        timed_logits: torch.Tensor,
        label_tensor: torch.Tensor,
        label_smoothing: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_ce_tb = EvaluationPolicy._per_sample_ce_by_time(
            timed_logits, label_tensor, 0.0
        )
        smooth_ce_tb = EvaluationPolicy._per_sample_ce_by_time(
            timed_logits, label_tensor, label_smoothing
        )
        pred_tb = timed_logits.argmax(dim=-1)
        acc_tb = (pred_tb == label_tensor.unsqueeze(0)).float()

        return raw_ce_tb.mean(dim=1), smooth_ce_tb.mean(dim=1), acc_tb.mean(dim=1)

    @staticmethod
    def _select_by_t_idx(
        timed_logits: torch.Tensor,
        t_idx: torch.Tensor,
    ) -> torch.Tensor:
        B = timed_logits.shape[1]
        return timed_logits[t_idx, torch.arange(B, device=timed_logits.device)]

    def _select_logits_for_reduction(
        self,
        timed_logits: torch.Tensor,
        reduction: TimeReduction,
    ) -> torch.Tensor:
        if reduction == "last":
            return timed_logits[-1]

        if reduction == "max_confidence":
            probs = functional.softmax(timed_logits, dim=-1)
            t_idx = probs.max(dim=-1).values.argmax(dim=0)
            return self._select_by_t_idx(timed_logits, t_idx)

        if reduction == "min_entropy":
            log_probs = functional.log_softmax(timed_logits, dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            t_idx = entropy.argmin(dim=0)
            return self._select_by_t_idx(timed_logits, t_idx)

        if reduction == "uniform":
            return timed_logits

        raise ValueError(f"Unknown reduction={reduction!r}")

    def _oracle_metrics(
        self,
        timed_logits: torch.Tensor,
        label_tensor: torch.Tensor,
        label_smoothing: float,
    ) -> dict[str, torch.Tensor]:
        T, B, C = timed_logits.shape
        raw_ce_tb = self._per_sample_ce_by_time(timed_logits, label_tensor, 0.0)
        smooth_ce_tb = self._per_sample_ce_by_time(
            timed_logits, label_tensor, label_smoothing
        )
        best_t = raw_ce_tb.argmin(dim=0)  # B
        arange_b = torch.arange(B, device=timed_logits.device)
        oracle_raw_ce = raw_ce_tb[best_t, arange_b].mean()
        oracle_smooth_ce = smooth_ce_tb[best_t, arange_b].mean()
        oracle_pred = timed_logits[best_t, arange_b].argmax(dim=-1)
        oracle_acc = (oracle_pred == label_tensor).float().mean()
        return {
            "raw_ce": oracle_raw_ce,
            "smooth_ce": oracle_smooth_ce,
            "acc": oracle_acc,
            "best_t": best_t,
        }

    def _shuffled_oracle_metrics(
        self,
        timed_logits: torch.Tensor,
        label_tensor: torch.Tensor,
        label_smoothing: float,
    ) -> dict[str, float]:
        B = label_tensor.numel()
        accum_raw = 0.0
        accum_smooth = 0.0
        accum_acc = 0.0
        for _ in range(self._shuffle_oracle_repeats):
            perm = torch.randperm(B, device=label_tensor.device)
            shuffled = label_tensor[perm]
            m = self._oracle_metrics(timed_logits, shuffled, label_smoothing)
            accum_raw += m["raw_ce"].item()
            accum_smooth += m["smooth_ce"].item()
            accum_acc += m["acc"].item()
        k = self._shuffle_oracle_repeats
        return {
            "raw_ce": accum_raw / k,
            "smooth_ce": accum_smooth / k,
            "acc": accum_acc / k,
        }

    def _hidden_state_stats(
        self,
        hidden_states: torch.Tensor,
    ) -> dict[str, Any]:
        """
        假设 hidden_states: T, B, D（或 T, B, ... 可 flatten 到 D）。
        返回相邻 t 余弦相似度、各 t 范数、T x T 余弦矩阵均值。
        """
        if hidden_states.ndim < 3:
            return {"error": f"hidden_states ndim={hidden_states.ndim} < 3"}

        T = hidden_states.shape[0]
        B = hidden_states.shape[1]
        h = hidden_states.reshape(T, B, -1)
        h_norm = functional.normalize(h, dim=-1)

        # 相邻 t 余弦
        cos_adj = (h_norm[1:] * h_norm[:-1]).sum(dim=-1)  # T-1, B
        cos_adj_mean = cos_adj.mean().item()
        cos_adj_per_t = cos_adj.mean(dim=1).detach().cpu().tolist()

        # 各 t 范数
        norm_per_t = h.norm(dim=-1).mean(dim=1).detach().cpu().tolist()

        # T x T 余弦矩阵（按样本平均）
        cos_matrix = torch.einsum("tbd,sbd->tsb", h_norm, h_norm).mean(dim=-1)
        cos_matrix_mean = cos_matrix.detach().cpu().tolist()

        # 非对角平均
        mask = ~torch.eye(T, dtype=torch.bool, device=cos_matrix.device)
        off_diag_mean = cos_matrix[mask].mean().item()

        return {
            "cos_adjacent_mean": round(cos_adj_mean, 6),
            "cos_adjacent_per_t": [round(float(v), 6) for v in cos_adj_per_t],
            "norm_per_t": [round(float(v), 6) for v in norm_per_t],
            "cos_matrix_mean": [
                [round(float(v), 6) for v in row] for row in cos_matrix_mean
            ],
            "cos_off_diag_mean": round(off_diag_mean, 6),
        }

    # ---------- 主入口 ----------

    def attend_batch(self, batch: MNLIBatch, output: ModelForwardOutput) -> None:
        timed_logits = output["timed_logits"]
        if timed_logits.ndim != 3:
            raise ValueError(
                f"Expected timed_logits T,B,C, got {tuple(timed_logits.shape)}"
            )
        T, B, C = timed_logits.shape

        label_tensor = torch.tensor(
            [get_logic_relation(label).value for label in batch.labels],
            dtype=torch.long,
            device=self._device,
        )
        if label_tensor.numel() != B:
            raise ValueError(
                f"Batch size mismatch: labels={label_tensor.numel()}, B={B}"
            )

        per_t_raw_ce, per_t_smooth_ce, per_t_acc = self._per_t_metrics(
            timed_logits, label_tensor, self._label_smoothing
        )

        last_logits = timed_logits[-1]
        last_raw_ce = functional.cross_entropy(last_logits, label_tensor, reduction="mean")
        last_smooth_ce = functional.cross_entropy(
            last_logits,
            label_tensor,
            reduction="mean",
            label_smoothing=self._label_smoothing,
        )
        last_pred = last_logits.argmax(dim=-1)
        last_correct = int((last_pred == label_tensor).sum().item())

        uniform_raw_ce = per_t_raw_ce.mean()
        uniform_smooth_ce = per_t_smooth_ce.mean()
        uniform_acc = per_t_acc.mean()

        oracle = self._oracle_metrics(
            timed_logits, label_tensor, self._label_smoothing
        )
        shuffled = self._shuffled_oracle_metrics(
            timed_logits, label_tensor, self._label_smoothing
        )

        official_logits = self._select_logits_for_reduction(
            timed_logits, self._official_time_reduction
        )
        if self._official_time_reduction == "uniform":
            official_raw_ce = uniform_raw_ce
            official_smooth_ce = uniform_smooth_ce
            official_correct = int(uniform_acc.item() * B)
            official_pred_hist = (
                torch.bincount(
                    timed_logits.argmax(dim=-1).reshape(-1),
                    minlength=C,
                )
                .div(T)
                .round()
                .long()
            )
        else:
            official_raw_ce = functional.cross_entropy(
                official_logits, label_tensor, reduction="mean"
            )
            official_smooth_ce = functional.cross_entropy(
                official_logits,
                label_tensor,
                reduction="mean",
                label_smoothing=self._label_smoothing,
            )
            official_pred = official_logits.argmax(dim=-1)
            official_correct = int((official_pred == label_tensor).sum().item())
            official_pred_hist = torch.bincount(official_pred, minlength=C)

        per_sample_last_ce = self._per_sample_ce_by_time(
            timed_logits[-1:], label_tensor, 0.0
        )[0]
        per_sample_oracle_ce = self._per_sample_ce_by_time(
            timed_logits, label_tensor, 0.0
        ).gather(0, oracle["best_t"].unsqueeze(0)).squeeze(0)
        gain_ce = per_sample_last_ce - per_sample_oracle_ce  # 正数表示 oracle 更好

        last_acc_per_sample = (last_pred == label_tensor).float()
        oracle_acc_per_sample = (
            timed_logits[
                oracle["best_t"], torch.arange(B, device=timed_logits.device)
            ].argmax(dim=-1)
            == label_tensor
        ).float()
        gain_acc = oracle_acc_per_sample - last_acc_per_sample

        new_count = B
        total_count = self._total_count + new_count

        self._official_raw_ce_avg += (
            official_raw_ce.item() - self._official_raw_ce_avg
        ) * (new_count / total_count)
        self._official_smooth_ce_avg += (
            official_smooth_ce.item() - self._official_smooth_ce_avg
        ) * (new_count / total_count)
        self._official_correct += official_correct

        self._last_raw_ce_avg += (last_raw_ce.item() - self._last_raw_ce_avg) * (
            new_count / total_count
        )
        self._last_smooth_ce_avg += (
            last_smooth_ce.item() - self._last_smooth_ce_avg
        ) * (new_count / total_count)
        self._last_correct += last_correct

        self._uniform_raw_ce_avg += (
            uniform_raw_ce.item() - self._uniform_raw_ce_avg
        ) * (new_count / total_count)
        self._uniform_smooth_ce_avg += (
            uniform_smooth_ce.item() - self._uniform_smooth_ce_avg
        ) * (new_count / total_count)
        self._uniform_correct += float(uniform_acc.item() * B)

        self._oracle_raw_ce_avg += (
            oracle["raw_ce"].item() - self._oracle_raw_ce_avg
        ) * (new_count / total_count)
        self._oracle_smooth_ce_avg += (
            oracle["smooth_ce"].item() - self._oracle_smooth_ce_avg
        ) * (new_count / total_count)
        self._oracle_correct += float(oracle["acc"].item() * B)

        self._shuffled_oracle_raw_ce_avg += (
            shuffled["raw_ce"] - self._shuffled_oracle_raw_ce_avg
        ) * (new_count / total_count)
        self._shuffled_oracle_smooth_ce_avg += (
            shuffled["smooth_ce"] - self._shuffled_oracle_smooth_ce_avg
        ) * (new_count / total_count)
        self._shuffled_oracle_correct += shuffled["acc"] * B

        self._gain_acc_sum += gain_acc.sum().item()
        self._gain_ce_sum += gain_ce.sum().item()
        self._gain_acc_positive += int((gain_acc > 0).sum().item())
        self._gain_ce_positive += int((gain_ce > 0).sum().item())

        self._per_t_raw_ce_sum += per_t_raw_ce.detach() * B
        self._per_t_smooth_ce_sum += per_t_smooth_ce.detach() * B
        self._per_t_acc_sum += per_t_acc.detach() * B
        self._per_t_sample_count += B

        best_t_hist = torch.bincount(oracle["best_t"], minlength=T)
        self._best_t_hist_sum += best_t_hist

        self._total_count = total_count
        self._batch_index += 1

        target_hist = torch.bincount(label_tensor, minlength=C).detach().cpu().tolist()
        last_pred_hist = torch.bincount(last_pred, minlength=C).detach().cpu().tolist()

        diag = BatchDiagnostics(
            batch_index=self._batch_index,
            batch_size=B,
            official_raw_ce=round(float(official_raw_ce.item()), 6),
            official_smooth_ce=round(float(official_smooth_ce.item()), 6),
            official_acc=round(official_correct / B, 6),
            last_raw_ce=round(float(last_raw_ce.item()), 6),
            last_smooth_ce=round(float(last_smooth_ce.item()), 6),
            last_acc=round(last_correct / B, 6),
            uniform_raw_ce=round(float(uniform_raw_ce.item()), 6),
            uniform_smooth_ce=round(float(uniform_smooth_ce.item()), 6),
            uniform_acc=round(float(uniform_acc.item()), 6),
            oracle_raw_ce=round(float(oracle["raw_ce"].item()), 6),
            oracle_smooth_ce=round(float(oracle["smooth_ce"].item()), 6),
            oracle_acc=round(float(oracle["acc"].item()), 6),
            shuffled_oracle_raw_ce=round(shuffled["raw_ce"], 6),
            shuffled_oracle_smooth_ce=round(shuffled["smooth_ce"], 6),
            shuffled_oracle_acc=round(shuffled["acc"], 6),
            per_t_raw_ce=[round(float(v), 6) for v in per_t_raw_ce.detach().cpu().tolist()],
            per_t_smooth_ce=[round(float(v), 6) for v in per_t_smooth_ce.detach().cpu().tolist()],
            per_t_acc=[round(float(v), 6) for v in per_t_acc.detach().cpu().tolist()],
            best_t_hist=best_t_hist.detach().cpu().tolist(),
            gain_acc_mean=round(float(gain_acc.mean().item()), 6),
            gain_acc_median=round(float(gain_acc.median().item()), 6),
            gain_acc_min=round(float(gain_acc.min().item()), 6),
            gain_acc_max=round(float(gain_acc.max().item()), 6),
            gain_acc_positive_frac=round(float((gain_acc > 0).float().mean().item()), 6),
            gain_ce_mean=round(float(gain_ce.mean().item()), 6),
            gain_ce_median=round(float(gain_ce.median().item()), 6),
            gain_ce_positive_frac=round(float((gain_ce > 0).float().mean().item()), 6),
            target_hist=target_hist,
            pred_hist_last=last_pred_hist,
            pred_hist_official=official_pred_hist.detach().cpu().tolist(),
        )
        self._history.append(diag)
        self._batch_acc_list.append(diag.official_acc)

        if self._verbose:
            self._print_batch(diag)

    def _print_batch(self, d: BatchDiagnostics) -> None:
        lines = [
            f"[Eval][batch {d.batch_index}] size={d.batch_size} "
            f"target={d.target_hist} pred_last={d.pred_hist_last}",
            f"  official({self._official_time_reduction}): "
            f"raw_ce={d.official_raw_ce:.4f} "
            f"smooth_ce={d.official_smooth_ce:.4f} acc={d.official_acc:.4f}",
            f"  last:    raw_ce={d.last_raw_ce:.4f} "
            f"smooth_ce={d.last_smooth_ce:.4f} acc={d.last_acc:.4f}",
            f"  uniform: raw_ce={d.uniform_raw_ce:.4f} "
            f"smooth_ce={d.uniform_smooth_ce:.4f} acc={d.uniform_acc:.4f}",
            f"  oracle:  raw_ce={d.oracle_raw_ce:.4f} "
            f"smooth_ce={d.oracle_smooth_ce:.4f} acc={d.oracle_acc:.4f}",
            f"  shuffled_oracle: raw_ce={d.shuffled_oracle_raw_ce:.4f} "
            f"smooth_ce={d.shuffled_oracle_smooth_ce:.4f} "
            f"acc={d.shuffled_oracle_acc:.4f}",
            f"  gain(oracle-last): acc_mean={d.gain_acc_mean:+.4f} "
            f"acc_pos={d.gain_acc_positive_frac:.2%} "
            f"ce_mean={d.gain_ce_mean:+.4f} ce_pos={d.gain_ce_positive_frac:.2%}",
            f"  best_t_hist={d.best_t_hist}",
            f"  per_t_raw_ce={d.per_t_raw_ce}",
            f"  per_t_acc={d.per_t_acc}",
        ]
        print("\n".join(lines))

    def collect(self) -> EvaluationResult:
        if self._total_count == 0:
            raise ValueError("Evaluation data contains no recognized labels")
        return EvaluationResult(
            raw_cross_entropy=self._official_raw_ce_avg,
            smooth_cross_entropy=self._official_smooth_ce_avg,
            accuracy=self._official_correct / self._total_count,
        )

    def collect_diagnostics(self) -> dict[str, Any]:
        if self._total_count == 0:
            raise ValueError("Evaluation data contains no recognized labels")
        n = self._total_count

        per_t_raw = (self._per_t_raw_ce_sum / self._per_t_sample_count).detach().cpu().tolist()
        per_t_smooth = (self._per_t_smooth_ce_sum / self._per_t_sample_count).detach().cpu().tolist()
        per_t_acc = (self._per_t_acc_sum / self._per_t_sample_count).detach().cpu().tolist()

        batch_acc_tensor = torch.tensor(self._batch_acc_list)
        return {
            "num_batches": self._batch_index,
            "total_samples": n,
            "official_time_reduction": self._official_time_reduction,
            "official": {
                "raw_ce": self._official_raw_ce_avg,
                "smooth_ce": self._official_smooth_ce_avg,
                "acc": self._official_correct / n,
            },
            "last": {
                "raw_ce": self._last_raw_ce_avg,
                "smooth_ce": self._last_smooth_ce_avg,
                "acc": self._last_correct / n,
            },
            "uniform": {
                "raw_ce": self._uniform_raw_ce_avg,
                "smooth_ce": self._uniform_smooth_ce_avg,
                "acc": self._uniform_correct / n,
            },
            "oracle_best_ce": {
                "raw_ce": self._oracle_raw_ce_avg,
                "smooth_ce": self._oracle_smooth_ce_avg,
                "acc": self._oracle_correct / n,
            },
            "shuffled_label_oracle": {
                "raw_ce": self._shuffled_oracle_raw_ce_avg,
                "smooth_ce": self._shuffled_oracle_smooth_ce_avg,
                "acc": self._shuffled_oracle_correct / n,
            },
            "oracle_minus_shuffled": {
                "raw_ce": self._oracle_raw_ce_avg - self._shuffled_oracle_raw_ce_avg,
                "smooth_ce": self._oracle_smooth_ce_avg - self._shuffled_oracle_smooth_ce_avg,
                "acc": self._oracle_correct / n - self._shuffled_oracle_correct / n,
            },
            "per_t": {
                "raw_ce": [round(float(v), 6) for v in per_t_raw],
                "smooth_ce": [round(float(v), 6) for v in per_t_smooth],
                "acc": [round(float(v), 6) for v in per_t_acc],
            },
            "best_t_hist": self._best_t_hist_sum.detach().cpu().tolist(),
            "per_sample_gain": {
                "acc_mean": self._gain_acc_sum / n,
                "acc_positive_frac": self._gain_acc_positive / n,
                "ce_mean": self._gain_ce_sum / n,
                "ce_positive_frac": self._gain_ce_positive / n,
            },
            "batch_acc": {
                "mean": float(batch_acc_tensor.mean().item()),
                "std": float(batch_acc_tensor.std().item()),
                "min": float(batch_acc_tensor.min().item()),
                "max": float(batch_acc_tensor.max().item()),
                "values": self._batch_acc_list,
            },
        }


class Evaluator:
    def __init__(
        self,
        checkpoint_path: str,
        data_reader: MNLIDataReader,
        model_options: ModelOptions,
        runtime_options: RuntimeOptions,
    ):
        self._tokenizer = MNLITokenizer(model_options.max_sequence_length)
        self._model_options = model_options
        self._runtime_options = runtime_options
        self._data_reader = data_reader
        self._device = resolve_device(runtime_options)

        self._model = LoopedMNLI(
            model_options, self._tokenizer.get_tokenizer_info()
        ).to(device=self._device, dtype=resolve_dtype(runtime_options.dtype))
        self._evaluation_policy = EvaluationPolicy(
            max_iterations=model_options.max_iterations,
            device=self._device,
            label_smoothing=0.0,
            official_time_reduction="last",
            shuffle_oracle_repeats=5,
            verbose=True,
        )

        checkpoint: dict = torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=True,
        )
        self._model.load_state_dict(checkpoint["model"])

        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])

        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())

        if self._device.type == "cuda" and checkpoint.get("cuda_random_state"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_random_state"]]
            )

    def evaluate(self) -> EvaluationResult:
        training_modes = [(module, module.training) for module in self._model.modules()]

        try:
            self._model.eval()

            for batch in self._data_reader.get_batches_iterator(shuffle=False):
                encoding = self._tokenizer.encode(batch.premises, batch.hypotheses)
                input_ids = encoding.input_ids.to(self._device)
                attention_mask = encoding.attention_mask.to(self._device)

                with torch.inference_mode(), self._autocast():
                    output: ModelForwardOutput = self._model(
                        input_ids,
                        attention_mask,
                    )

                self._evaluation_policy.attend_batch(batch, output)
        finally:
            for module, training in training_modes:
                module.train(training)

        result = self._evaluation_policy.collect()
        self._evaluation_policy.reset()
        return result

    def _autocast(self):
        amp_dtype = resolve_dtype(self._runtime_options.amp_dtype)
        if (
            self._runtime_options.amp_enabled
            and self._device.type == "cpu"
            and amp_dtype == torch.float16
        ):
            raise ValueError(
                "float16 AMP is unsupported on CPU; use bfloat16 or disable AMP"
            )
        return (
            torch.autocast(self._device.type, dtype=amp_dtype)
            if self._runtime_options.amp_enabled
            else nullcontext()
        )
