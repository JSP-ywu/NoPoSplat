import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization


@dataclass
class DatasetCambridgeCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    scenes: list[str]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    skip_bad_shape: bool


@dataclass
class DatasetCambridgeCfgWrapper:
    cambridge: DatasetCambridgeCfg

class DatasetCambridge(IterableDataset):
    cfg: DatasetCambridgeCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    seqs: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetCambridgeCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # TODO:
        # Intrinsics are given seperately per images (Under colmap calculation)
        # Cambridge dataset uses only 1 camera (assumption)
        '''
        Expected structure for intrinsics (in the case of single camera, obtained by COLMAP)

        # Camera list with one line of data per camera:
        #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # Number of cameras: 1
        1 CAMERA_MODEL w h fx fy cx cy

        '''
        with open(cfg.roots / "cameras.txt", 'r') as c:
            print('Loading intrinsics from cameras.txt')
            camera = c.readlines()[-1]
            self.intrinsics = torch.tensor(list(map(float, camera.strip().split()[4:])), dtype=torch.float32)

        # Collect seqs
        # Chunk = seq
        self.seqs = []
        for root in cfg.roots:
            for scene in cfg.scenes:
                root = root / scene
                # Load label files
                root_seqs = []
                for seq_path in root.glob(f"dataset_{stage}_seq*.txt"):
                    if seq_path.name != f"dataset_{stage}.txt":
                        root_seqs.append(seq_path)
                self.seqs.extend(root_seqs)
            if self.cfg.overfit_to_scene is not None:
                seq_path = self.index[self.cfg.overfit_to_scene]
                self.seqs = [seq_path] * len(self.seqs)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # seqs must be shuffled here (not inside __init__) for validation to show
        # random seqs.
        if self.stage in ("train", "test"):
            self.seqs = self.shuffle(self.seqs)

        # When testing, the data loaders alternate seqs.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.seqs = [
                seq
                for seq_index, seq in enumerate(self.seqs)
                if seq_index % worker_info.num_workers == worker_info.id
            ]
        # No need to shuffle seqs(there is only 1 chunk), shuffle in advance
        self.seqs = self.shuffle(self.seqs)

        for example in self.seqs:
            # In this case, example should be path of each sequences
            # Load the seq.
            # 1 seq is composed of 1 GT text(.txt file)
            
            '''
            GT file format
            Image path(under the scene directory) [x y z] [w x y z] (8 components per each line)
            Intrinsic is given seperately in the dataset
            '''

            seq = open(example, 'r')

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in seq if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                seq = item * len(seq)

            # TODO:
            # Prepare validation set seperately(train / test / validation)

            # if self.stage in ("train", "test", "validation"):
            #     seq = self.shuffle(seq)

            # example is originally 1 video data
            # for example in seq.readlines():
            # Remove newline character and split
            example = to_data_dict(self.cfg.roots, example)
            example = example.strip().split()
            extrinsics, intrinsics = self.convert_poses(example["cameras"], self.intrinsics)
            scene = example["key"]

            try:
                context_indices, target_indices, overlap = self.view_sampler.sample(
                    scene,
                    extrinsics,
                    intrinsics,
                )
            except ValueError:
                # Skip because the example doesn't have enough frames.
                continue

            # Skip the example if the field of view is too wide.
            if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                continue

            # Load the images.
            try:
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)
            except IndexError:
                continue
            except OSError:
                print(f"Skipped bad example {example['key']}.")  # DL3DV-Full have some bad images
                continue

            # Skip the example if the images don't have the right shape.
            context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
            target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
            if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                print(
                    f"Skipped bad example {example['key']}. Context shape was "
                    f"{context_images.shape} and target shape was "
                    f"{target_images.shape}."
                )
                continue

            # Resize the world to make the baseline 1.
            context_extrinsics = extrinsics[context_indices]
            if self.cfg.make_baseline_1:
                a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                scale = (a - b).norm()
                if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                    print(
                        f"Skipped {scene} because of baseline out of range: "
                        f"{scale:.6f}"
                    )
                    continue
                extrinsics[:, :3, 3] /= scale
            else:
                scale = 1

            if self.cfg.relative_pose:
                extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

            example = {
                "context": {
                    "extrinsics": extrinsics[context_indices],
                    "intrinsics": intrinsics[context_indices],
                    "image": context_images,
                    "near": self.get_bound("near", len(context_indices)) / scale,
                    "far": self.get_bound("far", len(context_indices)) / scale,
                    "index": context_indices,
                    "overlap": overlap,
                },
                "target": {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / scale,
                    "far": self.get_bound("far", len(target_indices)) / scale,
                    "index": target_indices,
                },
                "scene": scene,
            }
            if self.stage == "train" and self.cfg.augment:
                example = apply_augmentation_shim(example)
            yield apply_crop_shim(example, tuple(self.cfg.input_image_shape))

    # Is existance of batch means that overfit to scene is enabled?
    # Or input contains all information from 1 trajectories?
    def convert_poses(
        self,
        poses: Float[Tensor, "batch 7"],
        intr: Float[Tensor, "batch 4"]
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = intr
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
    
    def to_data_dict(
        self,
        root: Path,
        seq_path: str,
    )-> dict:
        scene = seq_path.split('/')[-2]
        cameras = torch.empty((0,11))
        images = []
        with open(seq_path, 'r') as seq:
            data_lines = np.asarray([line.strip() for line in seq.readlines()])
        for line in data_lines:
            line = line.split()
            images.append(str(root / scene / line[0]))
            cameras = torch.cat((cameras, torch.tensor([line[1:]])), dim=0)

        return {
            "key": scene,
            "cameras": cameras,
            "images": images,
        }

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    # def __len__(self) -> int:
    #     return len(self.index.keys())
