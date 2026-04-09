"""FTW dataset."""

import os
import random
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from matplotlib.figure import Figure
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

from ftw_tools.settings import ALL_COUNTRIES, TEMPORAL_OPTIONS
from ftw_tools.utils import validate_checksums


class FTW(NonGeoDataset):
    valid_splits = ["train", "val", "test"]

    def __init__(
        self,
        root: str = "data/ftw",
        countries: Sequence[str] | str | None = None,
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        checksum: bool = False,
        load_boundaries: bool = False,
        load_edges: bool = False,
        temporal_options: str = "stacked",
        swap_order: bool = False,
        num_samples: int = -1,
        ignore_sample_fn: Optional[str] = None,
        single_window_subdirs: Optional[dict[str, str]] = None,
        verbose: bool = True,
    ) -> None:
        """Initialize a new FTW dataset instance.

        Args:
            root: root directory where dataset can be found, this should contain the
                country folder
            countries: the countries to load the dataset from, e.g. "france"
            split: string specifying what split to load (e.g. "train", "val", "test")
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            checksum: if True, check the MD5 of the downloaded files (may be slow)
            load_boundaries: if True, load the 3 class masks with boundaries
            load_edges: if True, load the edge masks
            temporal_options : for ablation study, valid option are (stacked, windowA,
                windowB, median, rgb, random_window)
            swap_order: if True, swap the order of temporal data (i.e. use window A first)
            ignore_sample_fn: path to a filename with a list of samples to ignore
            single_window_subdirs: optional dict mapping country name to rgb image
                subdir (e.g. {"kenya_spot_processed": "scaled"}) for datasets that
                have only one time window instead of window_a and window_b. Use with
                temporal_options in (windowA, windowB, random_window) and in_channels: 4.
        Raises:
            AssertionError: if ``countries`` argument is invalid
            AssertionError: if ``split`` argument is invalid
            RuntimeError: if data is not found, or checksums don't match
        """
        self.root = root

        if countries is None:
            raise ValueError("Please specify the countries to load the dataset from")

        if temporal_options not in TEMPORAL_OPTIONS:
            raise ValueError(f"Invalid temporal option {temporal_options}")

        if isinstance(countries, str):
            countries = [countries]
        countries = [country.lower() for country in countries]
        for country in countries:
            assert country in ALL_COUNTRIES, f"Invalid country {country}"

        self.countries = countries
        assert split in self.valid_splits
        self.transforms = transforms
        self.checksum = checksum
        self.load_boundaries = load_boundaries
        self.load_edges = load_edges
        self.temporal_options = temporal_options
        self.num_samples = num_samples

        if swap_order:
            if temporal_options not in ("stacked", "rgb"):
                raise ValueError(
                    "Can only use swap_order with temporal_options stacked or rgb"
                )
        self.swap_order = swap_order
        self.single_window_subdirs = single_window_subdirs or {}

        if self.single_window_subdirs and temporal_options not in (
            "windowA",
            "windowB",
            "random_window",
            "rgb",
        ):
            raise ValueError(
                f"single_window_subdirs requires temporal_options in "
                f"(windowA, windowB, random_window, rgb); got {temporal_options!r}"
            )

        if verbose:
            if self.load_boundaries:
                print("Loading 3 Class Masks, with Boundaries")
            else:
                print("Loading 2 Class Masks, without Boundaries")
            print("Temporal option: ", temporal_options)
            if swap_order:
                print("Using window A first, then window B")
            else:
                print("Using window B first, then window A")
            if self.load_edges:
                print("Loading edge masks")
            if self.single_window_subdirs:
                print("Single-window countries (S2 subdir):", self.single_window_subdirs)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found at root directory or corrupted.  Download dataset with `ftw data download`"
            )

        if checksum:
            assert self._checksum(), "Checksum of dataset does not match"

        self.filenames = []
        all_filenames = []

        bad_samples = set()
        if ignore_sample_fn is not None:
            with open(ignore_sample_fn, "r") as f:
                lines = f.readlines()
                for line in lines:
                    country, sample_idx = line.split(",")
                    bad_samples.add((country, sample_idx.strip()))
            if verbose:
                print(f"Ignoring samples: {len(bad_samples)}")

        for country in self.countries:
            country_root: str = os.path.join(self.root, country)
            chips_fn = os.path.join(country_root, f"chips_{country}.parquet")
            chips_df = gpd.read_parquet(str(chips_fn))
            chips_df = chips_df[chips_df["split"] == split]
            # Parquet may use aoi_id (e.g. kenya_counties_batch) or chip_id (e.g. kenya_spot_processed)
            if "aoi_id" in chips_df.columns:
                id_col = "aoi_id"
            elif "chip_id" in chips_df.columns:
                id_col = "chip_id"
            else:
                raise KeyError(
                    f"chips parquet {chips_fn} must have column 'aoi_id' or 'chip_id'; "
                    f"got {list(chips_df.columns)}"
                )
            aoi_ids = chips_df[id_col].values

            is_single_window = country in self.single_window_subdirs
            s2_subdir = self.single_window_subdirs.get(country)

            for idx in aoi_ids:
                if (country, idx) in bad_samples:
                    continue

                if is_single_window and s2_subdir:
                    # Single-window layout: one S2 folder (e.g. scaled /)
                    s2_fn = Path(
                        os.path.join(country_root, s2_subdir, f"{idx}.tif")
                    )
                    masks_2c_fn = Path(
                        os.path.join(
                            country_root, "label_masks/semantic_2class", f"{idx}.tif"
                        )
                    )
                    masks_3c_fn = Path(
                        os.path.join(
                            country_root, "label_masks/semantic_3class", f"{idx}.tif"
                        )
                    )
                    edge_fn = Path(
                        os.path.join(country_root, "label_masks/edges", f"{idx}.tif")
                    )
                    if not (s2_fn.exists() and masks_2c_fn.exists() and masks_3c_fn.exists()):
                        continue
                    if self.load_edges and not edge_fn.exists():
                        raise ValueError(
                            "ERROR: Missing edge files! Run ./scripts/add_edges_to_dataset.py"
                        )
                    mask_fn = masks_3c_fn if self.load_boundaries else masks_2c_fn
                    file_record = {"s2_image": str(s2_fn), "mask": str(mask_fn)}
                    if self.load_edges:
                        file_record["edge"] = str(edge_fn)
                    all_filenames.append(file_record)
                    continue

                window_b_fn = Path(
                    os.path.join(country_root, "s2_images/window_b", f"{idx}.tif")
                )
                window_a_fn = Path(
                    os.path.join(country_root, "s2_images/window_a", f"{idx}.tif")
                )
                masks_2c_fn = Path(
                    os.path.join(
                        country_root, "label_masks/semantic_2class", f"{idx}.tif"
                    )
                )
                masks_3c_fn = Path(
                    os.path.join(
                        country_root, "label_masks/semantic_3class", f"{idx}.tif"
                    )
                )
                edge_fn = Path(
                    os.path.join(country_root, "label_masks/edges", f"{idx}.tif")
                )

                # Skip the image AOI's which does not have all four corresponding files
                if not (
                    window_b_fn.exists()
                    and window_a_fn.exists()
                    and masks_2c_fn.exists()
                    and masks_3c_fn.exists()
                ):
                    continue

                if self.load_edges and not edge_fn.exists():
                    raise ValueError(
                        "ERROR: Missing edge files! Run ./scripts/add_edges_to_dataset.py"
                    )

                if self.load_boundaries:
                    mask_fn = masks_3c_fn
                else:
                    mask_fn = masks_2c_fn

                file_record = {
                    "window_b": str(window_b_fn),
                    "window_a": str(window_a_fn),
                    "mask": str(mask_fn),
                }
                if self.load_edges:
                    file_record["edge"] = str(edge_fn)
                all_filenames.append(file_record)

        if self.num_samples == -1:  # select all samples
            self.filenames = all_filenames
        else:
            self.filenames = random.sample(
                all_filenames, min(self.num_samples, len(all_filenames))
            )

        if verbose:
            print(f"Selecting: {len(self.filenames)} samples")

    def _checksum(self) -> bool:
        """Check the checksum of the dataset.

        Returns:
            True if the checksum matches, else False
        """
        for country in ALL_COUNTRIES:
            print(f"Validating checksums for {country}")
            for checksum_file in [
                "distances_checksums.md5",
                "masks_checksums.md5",
                "window_b_checksums.md5",
                "window_a_checksums.md5",
            ]:
                checksum_file = os.path.join(self.root, country, checksum_file)
                if not os.path.exists(checksum_file):
                    print(f"Checksum file {checksum_file} not found")
                    return False
                if not validate_checksums(checksum_file, self.root):
                    return False
        return True

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset structure.

        Returns:
            True if the dataset directories and split files are found, else False
        """

        for country in self.countries:
            if country not in ALL_COUNTRIES:
                print(f"Invalid country {country}")
                return False

            country_dir = os.path.join(self.root, country)
            if not os.path.exists(country_dir):
                print(f"Country directory {country_dir} not found")
                return False

            chips_fns = list(Path(country_dir).glob("chips_*.parquet"))
            # boundaries_fns = list(Path(country_dir).glob(f"boundaries_*.parquet"))
            if len(chips_fns) != 1:
                print(f"Country {country} does not have chips file")
                return False

            if country in self.single_window_subdirs:
                s2_subdir = self.single_window_subdirs[country]
                required = [
                    os.path.exists(os.path.join(country_dir, s2_subdir)),
                    os.path.exists(
                        os.path.join(country_dir, "label_masks/semantic_3class")
                    )
                    if self.load_boundaries
                    else os.path.exists(
                        os.path.join(country_dir, "label_masks/semantic_2class")
                    ),
                ]
                if not all(required):
                    print(
                        f"Country {country} (single-window) does not have "
                        f"required directories: {s2_subdir}, label_masks"
                    )
                    return False
            elif self.load_boundaries:
                if not all(
                    [
                        os.path.exists(os.path.join(country_dir, "s2_images/window_b")),
                        os.path.exists(os.path.join(country_dir, "s2_images/window_a")),
                        os.path.exists(
                            os.path.join(country_dir, "label_masks/semantic_3class")
                        ),
                    ]
                ):
                    print(f"Country {country} does not have all required directories")
                    return False
            else:
                if not all(
                    [
                        os.path.exists(os.path.join(country_dir, "s2_images/window_b")),
                        os.path.exists(os.path.join(country_dir, "s2_images/window_a")),
                        os.path.exists(
                            os.path.join(country_dir, "label_masks/semantic_2class")
                        ),
                    ]
                ):
                    print(f"Country {country} does not have all required directories")
                    return False
        return True

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            dictionary containing "image" and "mask" PyTorch tensors
        """
        filenames = self.filenames[index]

        images = []
        if "s2_image" in filenames:
            # Single-window dataset: one 4-channel S2 image
            with rasterio.open(filenames["s2_image"]) as f:
                img = f.read()
            images.append(img)
        else:
            if self.temporal_options in ("stacked", "median", "windowB", "rgb"):
                with rasterio.open(filenames["window_b"]) as f:
                    window_b_img = f.read()
                    if self.temporal_options == "rgb":  # select 3 channels only
                        window_b_img = window_b_img[:3]
                    images.append(window_b_img)

            if self.temporal_options in ("stacked", "median", "windowA", "rgb"):
                with rasterio.open(filenames["window_a"]) as f:
                    window_a_img = f.read()
                    if self.temporal_options == "rgb":  # select 3 channels only
                        window_a_img = window_a_img[:3]
                    images.append(window_a_img)

            if self.temporal_options == "random_window":
                if random.random() < 0.5:
                    with rasterio.open(filenames["window_a"]) as f:
                        window_a_img = f.read()
                    images.append(window_a_img)
                else:
                    with rasterio.open(filenames["window_b"]) as f:
                        window_b_img = f.read()
                    images.append(window_b_img)

        if self.swap_order and len(images) == 2:
            images = [images[1], images[0]]

        if self.temporal_options == "median":
            images = np.array(images).astype(np.int32)
            image = np.median(images, axis=0).astype(np.int32)
        else:
            image = np.concatenate(images, axis=0).astype(np.int32)

        image = torch.from_numpy(image).float()

        with rasterio.open(filenames["mask"]) as f:
            mask = f.read(1)
        mask = torch.from_numpy(mask).long()

        sample = {"image": image, "mask": mask}

        if self.load_edges:
            with rasterio.open(filenames["edge"]) as f:
                edge = f.read(1)
            edge = torch.from_numpy(edge).long()
            sample["edge"] = edge

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def plot(self, sample: dict[str, Tensor], suptitle: Optional[str] = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample return by :meth:`__getitem__`
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample
        """
        img1 = sample["image"][0:3].numpy().transpose(1, 2, 0)

        if self.temporal_options in (
            "stacked",
            "rgb",
        ):  # only two option where we will have more than 4 channels in input image
            img2 = (
                sample["image"][3:6]
                if self.temporal_options == "rgb"
                else sample["image"][4:7]
            )
            img2 = img2.numpy().transpose(1, 2, 0)

        mask = sample["mask"].numpy().squeeze()
        num_panels = 3 if self.temporal_options in ("stacked", "rgb") else 2
        if "prediction" in sample:
            num_panels += 1
            predictions = sample["prediction"].numpy()

        fig, axs = plt.subplots(1, num_panels, figsize=(num_panels * 5, 8))
        axs = axs.flatten()
        axs[0].imshow(np.clip(img1, 0, 1))
        axs[0].axis("off")

        panel_id = 1
        if self.temporal_options in ("stacked", "rgb"):
            axs[panel_id].imshow(np.clip(img2, 0, 1))
            axs[panel_id].axis("off")
            axs[panel_id + 1].imshow(mask, vmin=0, vmax=2, cmap="gray")
            axs[panel_id + 1].axis("off")
            panel_id += 2
        else:
            axs[panel_id].imshow(mask, vmin=0, vmax=2, cmap="gray")
            axs[panel_id].axis("off")
            panel_id += 1

        if "prediction" in sample:
            axs[panel_id].imshow(predictions, vmin=0, vmax=2, cmap="gray")
            axs[panel_id].axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
