import argparse
from pathlib import Path
from time import time
from typing import Optional
import pprofile

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage

from torchvideo.datasets import (
    GulpVideoDataset,
    DummyLabelSet,
    VideoDataset,
    VideoFolderDataset,
    ImageFolderVideoDataset,
)
from torchvideo.samplers import (
    FullVideoSampler,
    ClipSampler,
    FrameSampler,
    TemporalSegmentSampler,
)
from torchvideo.transforms import (
    CenterCropVideo,
    TimeApply,
    CollectFrames,
    PILVideoToTensor,
)

parser = argparse.ArgumentParser(
    description="Benchmark data loading",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("dataset_root", type=Path, help="Path to the root of the dataset")
parser.add_argument("-n", "--max-iterations", type=int, default=-1)
parser.add_argument("-b", "--batch-size", type=int, default=64)
parser.add_argument("--shuffle", action="store_true")
parser.add_argument("--pin-memory", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument(
    "--profile-callgrind", type=Path, help="Path to store callgrind output"
)
parser.add_argument(
    "--dataset-type", type=str, default="gulp", choices=["gulp", "image", "video"]
)
parser.add_argument("--image-filename-template", default="frame_{:05d}.jpg")
parser.add_argument(
    "--sampler", type=str, default="clip", choices=["full", "clip", "tsn"]
)
parser.add_argument("--sampler-clip-length", type=int, default=10)
parser.add_argument("--sampler-tsn-segment-count", type=int, default=3)
parser.add_argument("--sampler-tsn-segment-length", type=int, default=1)
parser.add_argument("-j", "--workers", type=int, default=0)


def benchmark_dataloader(
    loader: DataLoader,
    max_iterations: int = -1,
    profile: bool = False,
    profile_callgrind: Path = None,
) -> None:
    dataloader_iter = iter(loader)

    def run_dataloader():
        end_of_iter_time = time()
        total_iterations = (
            len(dataloader_iter)
            if max_iterations < 0
            else min(len(dataloader_iter), max_iterations)
        )
        for i, batch in enumerate(dataloader_iter):
            if 0 < max_iterations <= i:
                break
            start_of_iter_time = time()
            dataloader_duration_s = start_of_iter_time - end_of_iter_time
            examples_per_second = loader.batch_size / dataloader_duration_s

            print(
                "batch[{}/{}] {:.2f} examples/s".format(
                    i + 1, total_iterations, examples_per_second
                )
            )
            end_of_iter_time = start_of_iter_time

    if profile:
        prof = pprofile.Profile()
        with prof():
            run_dataloader()
        if profile_callgrind is not None:
            with open(str(profile_callgrind), "w", encoding="utf8") as f:
                prof.callgrind(f)
            print("Wrote callgrind profile log to {}".format(profile_callgrind))
        else:
            prof.print_stats()
    else:
        run_dataloader()


def main(args) -> None:
    sampler = make_sampler(args)
    dataset = make_dataset(
        args,
        sampler=sampler,
        transform=Compose([CenterCropVideo(100), CollectFrames(), PILVideoToTensor()]),
    )
    loader = DataLoader(
        dataset,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
    )
    benchmark_dataloader(
        loader,
        max_iterations=args.max_iterations,
        profile=args.profile,
        profile_callgrind=args.profile_callgrind,
    )


def make_dataset(
    args, sampler: Optional[FrameSampler] = None, transform=None
) -> VideoDataset:
    dataset_type = args.dataset_type.lower()
    if dataset_type == "gulp":
        if transform is not None:
            transform = Compose([TimeApply(ToPILImage()), transform])
        return GulpVideoDataset(
            args.dataset_root,
            label_set=DummyLabelSet(),
            sampler=sampler,
            transform=transform,
        )
    elif dataset_type == "image":
        return ImageFolderVideoDataset(
            args.dataset_root,
            args.image_filename_template,
            label_set=DummyLabelSet(),
            sampler=sampler,
            transform=transform,
        )
    elif dataset_type == "video":
        return VideoFolderDataset(
            args.dataset_root,
            label_set=DummyLabelSet(),
            sampler=sampler,
            transform=transform,
        )
    else:
        raise ValueError("Unknown dataset type '{}'".format(args.dataset_type))


def make_sampler(args):
    if args.sampler == "full":
        return FullVideoSampler()
    elif args.sampler == "clip":
        return ClipSampler(args.sampler_clip_length)
    elif args.sampler == "tsn":
        return TemporalSegmentSampler(
            args.sampler_tsn_segment_count, args.sampler_tsn_segment_length
        )
    else:
        raise ValueError("Expected --sampler to be one of 'full', 'clip', or 'tsn'")


if __name__ == "__main__":
    main(parser.parse_args())
