import numpy as np
import ray

from src.models import SpecialTokens

from .augment import CropAndAugment


class Remap:
    def __init__(self, L, K):
        self.L = L
        self.K = K

        self.levels_offset = np.arange(L) * K  # Broadcasted
        self.special_tokens_offset = len(SpecialTokens)

    def __call__(self, quantized):
        return quantized + self.levels_offset + self.special_tokens_offset


class MARIUSPrePro:
    def __init__(
        self,
        *,  # Pass all arguments as keyword arguments
        quantizer_ref,
        split,
        crop_length,
        augment,
        random_crop=True,
        **kwargs,
    ):
        self.split = split

        # Context parameters
        self.split = split
        # Context parameters
        self.crop_and_augment = CropAndAugment(
            crop_length=crop_length,
            augment=augment,
            random_crop=random_crop,
            split=split,
        )

        self.quant_df = ray.get(quantizer_ref)

        self.L = len(self.quant_df.columns)
        self.K = self.quant_df.values.max() + 1
        print(f"Quantized table has {self.L} levels and {self.K} codes per level.")

        self.remap = Remap(L=self.L, K=self.K)

    def __call__(self, row):
        # Quantize the timeline
        tl, ts = self.crop_and_augment(row["timeline"], row["timestamp"])

        quantized_query = self.remap(self.quant_df.loc[tl].values)
        # Shape : L x K

        pad_to = self.crop_and_augment.crop_length

        # Target - All items except first.
        target = quantized_query[1:]
        target = np.concatenate(
            [
                np.full(
                    (pad_to - target.shape[0], self.L),
                    -100,
                ),
                target,
            ],
            axis=0,
        )

        # Prepare the input - Drop last item
        input = quantized_query[:-1]
        input = np.concatenate(
            [
                np.full(
                    (pad_to - input.shape[0] - 1, self.L),
                    SpecialTokens.PAD.value,
                ),
                np.full((1, self.L), SpecialTokens.BOS.value),
                input,
            ],
            axis=0,
        )

        return {"input": input, "target": target}


class MARIUSDistillPrePro(MARIUSPrePro):
    """MARIUSPrePro extended with two extra batch keys for score-KL distillation.

    Extra keys emitted on each call:
      "teacher_query"      -- same crop's item-id history in SASRec item-index
                              space, left-padded with PAD to crop_length (int64).
      "target_catalog_idx" -- the true next item's catalog index, equal to
                              item_to_id[last_item] - len(SpecialTokens) (int64 scalar).

    The random crop is shared with the parent's input/target construction so
    teacher_query and the MARIUS input refer to the SAME sequence slice.
    """

    def __init__(self, *, items_ref, **kwargs):
        super().__init__(**kwargs)
        items_map = ray.get(items_ref)
        self.item_to_id = items_map["item_to_id"]

    def __call__(self, row):
        tl, ts = self.crop_and_augment(row["timeline"], row["timestamp"])

        quantized_query = self.remap(self.quant_df.loc[tl].values)

        pad_to = self.crop_and_augment.crop_length

        # Target - All items except first.
        target = quantized_query[1:]
        target = np.concatenate(
            [
                np.full(
                    (pad_to - target.shape[0], self.L),
                    -100,
                ),
                target,
            ],
            axis=0,
        )

        # Prepare the input - Drop last item
        input = quantized_query[:-1]
        input = np.concatenate(
            [
                np.full(
                    (pad_to - input.shape[0] - 1, self.L),
                    SpecialTokens.PAD.value,
                ),
                np.full((1, self.L), SpecialTokens.BOS.value),
                input,
            ],
            axis=0,
        )

        # Distillation extras (same crop tl as above).
        tq_ids = [self.item_to_id[it] for it in tl[:-1]]
        teacher_query = np.array(
            [SpecialTokens.PAD.value] * (pad_to - len(tq_ids)) + tq_ids,
            dtype=np.int64,
        )
        target_catalog_idx = np.int64(
            self.item_to_id[tl[-1]] - len(SpecialTokens)
        )

        return {
            "input": input,
            "target": target,
            "teacher_query": teacher_query,
            "target_catalog_idx": target_catalog_idx,
        }
