import random

import numpy as np


class CropAndAugment:
    def __init__(
        self,
        crop_length,
        split,
        random_crop=True,
        augment=None,
        seed=None,
    ):
        self.crop_length = crop_length
        self.split = split

        self.augment = augment
        self.random_crop = random_crop

        # Reproducibility (documented deviation): use seeded generators instead of
        # numpy/python global RNGs so augmentation is seed-controlled. seed=None
        # preserves the original unseeded behaviour.
        self._np_rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

        assert augment in [None, "all_shuffle", "timewise_shuffle"], augment

    def __call__(self, tl, ts):
        tl, ts = self._maybe_crop_query(tl, ts)

        if self.split == "train" and self.augment is not None:
            tl, ts = self._augment(tl, ts)

        return tl, ts

    def _augment(self, tl, ts):
        if self.augment == "all_shuffle":
            indices = self._np_rng.permutation(len(tl))
            tl = tl[indices]
            ts = ts[indices]

        elif self.augment == "timewise_shuffle":
            for _t in np.unique(ts):
                mask = ts == _t
                if mask.sum() > 1:  # Only shuffle if there are multiple entries
                    tl[mask] = tl[mask][self._np_rng.permutation(mask.sum())]

        return tl, ts

    def _maybe_crop_query(self, tl, ts):
        if (
            self.split != "train" or not self.random_crop
        ):  # valid or test, keep last samples
            if len(tl) > self.crop_length:
                tl = tl[-self.crop_length :]
                ts = ts[-self.crop_length :]
            return tl, ts

        elif self.split == "train" and self.random_crop:
            # How many steps to select
            len_to_select = min(len(tl), self.crop_length)

            # Where to select
            if len(tl) > len_to_select:
                offset = self._py_rng.randint(0, len(tl) - len_to_select)
            else:
                offset = 0

            # Select the steps
            tl = tl[offset : offset + len_to_select]
            ts = ts[offset : offset + len_to_select]

        return tl, ts
