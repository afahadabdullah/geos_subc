"""Training-only extreme sampling and threshold artifacts for flow-multi v9.4."""

from collections import Counter
import math
import os

import torch
from torch.utils.data import Dataset, Sampler


CATEGORY_NAMES = ("ordinary", "heavy_pr", "warm_t2m", "cold_t2m")


def _sample_target_raw(dataset, index):
    if getattr(dataset, "preload", False) and getattr(dataset, "data_cache", None):
        sample = dataset.data_cache[index]
    else:
        sample = dataset[index]
    return sample["target_raw"].detach().float().cpu()


def _field_score(target_raw):
    pr = target_raw[0].reshape(-1)
    t2m = target_raw[1].reshape(-1)
    return torch.stack(
        [
            torch.quantile(pr, 0.95),
            torch.quantile(t2m, 0.90),
            torch.quantile(t2m, 0.10),
        ]
    )


def build_or_load_extreme_artifact(dataset, path, force=False):
    """Build month/lead thresholds using only the supplied training dataset."""
    if os.path.exists(path) and not force:
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if int(artifact.get("num_samples", -1)) != len(dataset):
            raise ValueError(
                f"Extreme artifact {path} contains {artifact.get('num_samples')} samples, "
                f"but the training dataset contains {len(dataset)}. Rebuild the artifact."
            )
        return artifact

    targets = []
    scores = []
    months = []
    leads = []
    for index, meta in enumerate(dataset.samples):
        raw = _sample_target_raw(dataset, index)
        targets.append(raw)
        scores.append(_field_score(raw))
        months.append(int(meta["date"].month) - 1)
        leads.append(int(meta["lead_idx"]))

    targets = torch.stack(targets)
    scores = torch.stack(scores)
    months = torch.tensor(months, dtype=torch.long)
    leads = torch.tensor(leads, dtype=torch.long)
    height, width = targets.shape[-2:]

    grid_thresholds = {
        name: torch.empty((12, 4, height, width), dtype=torch.float32)
        for name in ("pr_q90", "pr_q95", "t2m_q05", "t2m_q10", "t2m_q90", "t2m_q95")
    }
    score_thresholds = torch.empty((12, 4, 3), dtype=torch.float32)

    for month in range(12):
        for lead in range(4):
            mask = (months == month) & (leads == lead)
            if not mask.any():
                raise RuntimeError(f"No training samples for month={month + 1}, lead={lead + 1}")
            group_targets = targets[mask]
            group_scores = scores[mask]
            grid_thresholds["pr_q90"][month, lead] = torch.quantile(
                group_targets[:, 0], 0.90, dim=0
            )
            grid_thresholds["pr_q95"][month, lead] = torch.quantile(
                group_targets[:, 0], 0.95, dim=0
            )
            grid_thresholds["t2m_q05"][month, lead] = torch.quantile(
                group_targets[:, 1], 0.05, dim=0
            )
            grid_thresholds["t2m_q10"][month, lead] = torch.quantile(
                group_targets[:, 1], 0.10, dim=0
            )
            grid_thresholds["t2m_q90"][month, lead] = torch.quantile(
                group_targets[:, 1], 0.90, dim=0
            )
            grid_thresholds["t2m_q95"][month, lead] = torch.quantile(
                group_targets[:, 1], 0.95, dim=0
            )
            score_thresholds[month, lead, 0] = torch.quantile(group_scores[:, 0], 0.90)
            score_thresholds[month, lead, 1] = torch.quantile(group_scores[:, 1], 0.90)
            score_thresholds[month, lead, 2] = torch.quantile(group_scores[:, 2], 0.10)

    labels = torch.zeros(len(dataset), dtype=torch.long)
    for index in range(len(dataset)):
        month = int(months[index])
        lead = int(leads[index])
        threshold = score_thresholds[month, lead]
        score = scores[index]
        severity = torch.tensor(
            [
                -float("inf"),
                float((score[0] - threshold[0]) / (threshold[0].abs() + 1e-6)),
                float((score[1] - threshold[1]) / (threshold[1].abs() + 1e-6)),
                float((threshold[2] - score[2]) / (threshold[2].abs() + 1e-6)),
            ]
        )
        active = torch.tensor(
            [
                True,
                bool(score[0] >= threshold[0]),
                bool(score[1] >= threshold[1]),
                bool(score[2] <= threshold[2]),
            ]
        )
        severity[~active] = -float("inf")
        labels[index] = int(torch.argmax(severity)) if active[1:].any() else 0

    counts = torch.bincount(labels, minlength=len(CATEGORY_NAMES))
    natural_fractions = counts.float() / max(1, len(dataset))
    artifact = {
        "version": "v9.4",
        "num_samples": len(dataset),
        "category_names": list(CATEGORY_NAMES),
        "labels": labels,
        "counts": counts,
        "natural_fractions": natural_fractions,
        "months": months,
        "leads": leads,
        "score_thresholds": score_thresholds,
        "grid_thresholds": grid_thresholds,
        "training_year_start": min(int(meta["year"]) for meta in dataset.samples),
        "training_year_end": max(int(meta["year"]) for meta in dataset.samples),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(artifact, path)
    return artifact


def classify_dataset(dataset, artifact):
    """Classify another dataset using training-only scalar thresholds."""
    thresholds = artifact["score_thresholds"]
    labels = torch.zeros(len(dataset), dtype=torch.long)
    for index, meta in enumerate(dataset.samples):
        score = _field_score(_sample_target_raw(dataset, index))
        threshold = thresholds[int(meta["date"].month) - 1, int(meta["lead_idx"])]
        severity = torch.tensor(
            [
                -float("inf"),
                float((score[0] - threshold[0]) / (threshold[0].abs() + 1e-6)),
                float((score[1] - threshold[1]) / (threshold[1].abs() + 1e-6)),
                float((threshold[2] - score[2]) / (threshold[2].abs() + 1e-6)),
            ]
        )
        active = torch.tensor(
            [
                True,
                bool(score[0] >= threshold[0]),
                bool(score[1] >= threshold[1]),
                bool(score[2] <= threshold[2]),
            ]
        )
        severity[~active] = -float("inf")
        labels[index] = int(torch.argmax(severity)) if active[1:].any() else 0
    return labels


def select_extreme_validation_indices(dataset, artifact, max_per_category=64, seed=941):
    labels = classify_dataset(dataset, artifact)
    generator = torch.Generator().manual_seed(int(seed))
    key_to_indices = {}
    for index, meta in enumerate(dataset.samples):
        key = (int(meta["year"]), int(meta["s_idx"]))
        key_to_indices.setdefault(key, []).append(index)
    selected_keys = set()
    counts = {}
    for category in range(1, len(CATEGORY_NAMES)):
        indices = torch.where(labels == category)[0]
        keys = []
        seen = set()
        for index in indices.tolist():
            meta = dataset.samples[index]
            key = (int(meta["year"]), int(meta["s_idx"]))
            if key not in seen:
                seen.add(key)
                keys.append(key)
        if max_per_category > 0 and len(keys) > max_per_category:
            order = torch.randperm(len(keys), generator=generator)[:max_per_category]
            keys = [keys[int(item)] for item in order]
        selected_keys.update(keys)
        counts[CATEGORY_NAMES[category]] = len(keys)
    selected = [
        index
        for key in sorted(selected_keys)
        for index in sorted(key_to_indices[key], key=lambda item: dataset.samples[item]["lead_idx"])
    ]
    return selected, counts


class ExtremeSamplingDataset(Dataset):
    """Inject sampling metadata while leaving the underlying v9.3 data unchanged."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            index, category, correction = key
        else:
            index, category, correction = int(key), 0, 1.0
        sample = dict(self.dataset[int(index)])
        sample["sample_index"] = torch.tensor(int(index), dtype=torch.long)
        sample["sampling_category"] = torch.tensor(int(category), dtype=torch.long)
        sample["sampling_weight"] = torch.tensor(float(correction), dtype=torch.float32)
        return sample

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class ExtremeBalancedBatchSampler(Sampler):
    """Deterministic epoch-level category mixture with no duplicate inside a batch."""

    def __init__(
        self,
        labels,
        batch_size,
        target_fractions=(0.65, 0.15, 0.10, 0.10),
        uniform_fraction=0.0,
        correction_min=0.25,
        correction_max=4.0,
        seed=941,
        drop_last=False,
    ):
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        target = torch.tensor(target_fractions, dtype=torch.float64)
        if len(target) != len(CATEGORY_NAMES) or (target < 0).any():
            raise ValueError(f"Invalid extreme target fractions: {target_fractions}")
        target = target / target.sum()
        natural = torch.bincount(self.labels, minlength=len(CATEGORY_NAMES)).double()
        natural = natural / natural.sum()
        uniform_fraction = float(uniform_fraction)
        if not 0.0 <= uniform_fraction <= 1.0:
            raise ValueError("uniform_fraction must be in [0, 1]")
        self.sample_fractions = (1.0 - uniform_fraction) * target + uniform_fraction * natural
        correction = natural / self.sample_fractions.clamp_min(1e-12)
        correction = correction.clamp(float(correction_min), float(correction_max))
        correction = correction / torch.sum(self.sample_fractions * correction)
        self.correction_weights = correction.float()
        self.pools = [torch.where(self.labels == category)[0] for category in range(len(CATEGORY_NAMES))]
        if any(len(pool) == 0 for pool in self.pools):
            missing = [CATEGORY_NAMES[i] for i, pool in enumerate(self.pools) if len(pool) == 0]
            raise RuntimeError(f"Extreme sampler has empty category pools: {missing}")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.drop_last:
            return len(self.labels) // self.batch_size
        return math.ceil(len(self.labels) / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        draw_count = len(self) * self.batch_size
        exact = self.sample_fractions * draw_count
        counts = torch.floor(exact).long()
        remainder = draw_count - int(counts.sum())
        if remainder:
            order = torch.argsort(exact - counts, descending=True)
            counts[order[:remainder]] += 1
        categories = torch.cat(
            [torch.full((int(count),), category, dtype=torch.long) for category, count in enumerate(counts)]
        )
        categories = categories[torch.randperm(len(categories), generator=generator)]

        shuffled_pools = []
        cursors = [0] * len(CATEGORY_NAMES)
        for pool in self.pools:
            shuffled_pools.append(pool[torch.randperm(len(pool), generator=generator)])

        for batch_number in range(len(self)):
            start = batch_number * self.batch_size
            stop = min(start + self.batch_size, len(categories))
            if self.drop_last and stop - start < self.batch_size:
                break
            chosen = set()
            batch = []
            for category_tensor in categories[start:stop]:
                category = int(category_tensor)
                pool = shuffled_pools[category]
                attempts = 0
                while True:
                    if cursors[category] >= len(pool):
                        pool = self.pools[category][
                            torch.randperm(len(self.pools[category]), generator=generator)
                        ]
                        shuffled_pools[category] = pool
                        cursors[category] = 0
                    index = int(pool[cursors[category]])
                    cursors[category] += 1
                    attempts += 1
                    if index not in chosen or attempts > len(pool):
                        break
                if index in chosen:
                    fallback = torch.randperm(len(self.labels), generator=generator)
                    index = next(int(item) for item in fallback if int(item) not in chosen)
                    category = int(self.labels[index])
                chosen.add(index)
                batch.append((index, category, float(self.correction_weights[category])))
            yield batch

    def summary(self):
        return {
            "sample_fractions": {
                CATEGORY_NAMES[i]: float(self.sample_fractions[i]) for i in range(len(CATEGORY_NAMES))
            },
            "correction_weights": {
                CATEGORY_NAMES[i]: float(self.correction_weights[i]) for i in range(len(CATEGORY_NAMES))
            },
            "natural_counts": dict(Counter(CATEGORY_NAMES[int(item)] for item in self.labels)),
        }
