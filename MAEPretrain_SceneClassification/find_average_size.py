# Copyright contributors to the neural-embedding-compression project

import argparse
from typing import Iterable

import models_mae
import numpy as np
import torch
import torchvision.transforms as transforms
import zarr
from mmseg.datasets import build_dataloader
from numcodecs import Blosc
from tqdm import tqdm
from util.datasets import UCMDataset, build_dataset
from util.pos_embed import interpolate_pos_embed

from MAEPretrain_SceneClassification.util.compression import findEmbeddingSize


def get_args_parser():
    parser = argparse.ArgumentParser("Compression evaluation")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_compress_adapter",
        type=str,
        metavar="MODEL",
        help="Name of model",
    )
    parser.add_argument("--model_path", type=str, required=True, help="path of model")

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    # Dataset parameters
    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        help="dataset name",
        choices=["ucm", "MillionAid", "potsdam"],
    )

    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--eval_size",
        type=int,
        help="Number of samples to use to estimate embedding size. If none, use the whole dataset",
    )
    parser.add_argument(
        "--entropy",
        action="store_true",
        help="If model has been trained with entropy compression, additionally evaluate with this",
    )
    parser.add_argument(
        "--data_path",
        required=False,
        type=str,
        help="Path to data",
    )

    return parser


def quantize_data(data, bits):
    latent_min = data.min()
    latent_max = data.max()
    upper_range = (2**bits) - 1
    scaled = ((data - latent_min) / (latent_max - latent_min)) * upper_range
    quantized_data = scaled.round().astype(np.uint8)
    assert quantized_data.max() <= upper_range and quantized_data.min() >= 0
    return quantized_data


def main(args):
    if "mae_vit_" in args.model:
        print("MAE pretraining ViT series model")
        model = models_mae.__dict__[args.model](0, img_size=args.input_size)
    else:
        try:
            model = models_mae.__dict__[args.model]
        except Exception as e:
            raise NotImplementedError from e

    model = model.cuda()
    checkpoint = torch.load(args.model_path)["model"]
    interpolate_pos_embed(model, checkpoint)
    model.load_state_dict(checkpoint, strict=False)

    # data path is ignored for potsdam, uses one in config files
    dataset_eval = build_dataset("test", args=args, data_path=args.data_path)

    if args.eval_size is not None:
        if len(dataset_eval) > args.eval_size:
            dataset_eval = torch.utils.data.Subset(
                dataset_eval,
                torch.arange(args.eval_size),
            )
        else:
            args.eval_size = len(dataset_eval)

    args.eval_size = len(dataset_eval)
    print(f"Loaded {args.eval_size} samples")

    if args.dataset == "potsdam":
        data_loader_eval = build_dataloader(
            dataset_eval, args.batch_size, args.num_workers, drop_last=False
        )
    else:
        data_loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
        )

    if args.entropy:
        if args.dataset == "potsdam":
            ld = lambda x: x["img"][0]
        else:
            ld = lambda x: x[0]
        embedding_sizes = findEmbeddingSize(model, data_loader_eval, extract_data_ld=ld)
        av_size = (embedding_sizes.sum() / args.eval_size) * 8

    embeddings = []
    image_bits = []
    image_bits_8 = []
    image_bits_16 = []
    bits = []
    bits_2 = []
    bits_3 = []
    bits_4 = []
    bits_5 = []
    bits_8 = []
    bits_16 = []

    image_bits_compressed = []
    image_bits_compressed_8 = []
    image_bits_compressed_16 = []
    bits_compressed = []
    bits_2_compressed = []
    bits_3_compressed = []
    bits_4_compressed = []
    bits_5_compressed = []
    bits_8_compressed = []
    bits_16_compressed = []
    with torch.no_grad():
        for data in tqdm(data_loader_eval):
            if args.dataset == "potsdam":
                data = data["img"]
            # else:
            data = data[0]
            data = data.to("cuda")
            embedding, _, _ = model.forward_encoder(data, 0)

            compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

            raw_data = data.cpu().numpy()
            z_raw = zarr.array(raw_data, compressor=compressor)

            raw_data_16 = data.cpu().numpy()
            latent_min = raw_data_16.min()
            latent_max = raw_data_16.max()
            scaled = ((raw_data_16 - latent_min) / (latent_max - latent_min)) * (
                (2**16) - 1
            )
            raw_data_16 = scaled.round().astype(np.uint16)
            z_raw_16 = zarr.array(raw_data_16, compressor=compressor)

            raw_data_8 = quantize_data(data.cpu().numpy(), 8)
            z_raw_8 = zarr.array(raw_data_8, compressor=compressor)

            embeddings = embedding.cpu().numpy()
            z = zarr.array(embeddings, compressor=compressor)

            embeddings8 = quantize_data(np.copy(embeddings), 8)
            z8 = zarr.array(embeddings8, compressor=compressor)

            embeddings4 = quantize_data(np.copy(embeddings), 4)
            z4 = zarr.array(embeddings4, compressor=compressor)

            embeddings5 = quantize_data(np.copy(embeddings), 5)
            z5 = zarr.array(embeddings5, compressor=compressor)

            embeddings3 = quantize_data(np.copy(embeddings), 3)
            z3 = zarr.array(embeddings3, compressor=compressor)

            embeddings2 = quantize_data(np.copy(embeddings), 2)
            z2 = zarr.array(embeddings2, compressor=compressor)

            embeddings16 = embeddings.astype(np.float16)
            z16 = zarr.array(embeddings16, compressor=compressor)

            image_bits.append(z_raw.nbytes * 8)
            image_bits_16.append(z_raw_16.nbytes * 8)
            image_bits_8.append(z_raw_8.nbytes * 8)
            bits.append(z.nbytes * 8)
            bits_2.append(z2.nbytes * 8)
            bits_3.append(z3.nbytes * 8)
            bits_4.append(z4.nbytes * 8)
            bits_8.append(z8.nbytes * 8)
            bits_5.append(z5.nbytes * 8)
            bits_16.append(z16.nbytes * 8)

            image_bits_compressed.append((z_raw.nbytes_stored) * 8)
            image_bits_compressed_16.append((z_raw_16.nbytes_stored) * 8)
            image_bits_compressed_8.append((z_raw_8.nbytes_stored) * 8)
            bits_compressed.append((z.nbytes_stored) * 8)
            bits_2_compressed.append((z2.nbytes_stored) * 8)
            bits_3_compressed.append((z3.nbytes_stored) * 8)
            bits_4_compressed.append((z4.nbytes_stored) * 8)
            bits_8_compressed.append((z8.nbytes_stored) * 8)
            bits_5_compressed.append((z5.nbytes_stored) * 8)
            bits_16_compressed.append((z16.nbytes_stored) * 8)

    
    print(args.eval_size)
    print("RAW IMAGE -----------")
    print(
        f"Average bits per image that would need to be stored: {np.array(image_bits).sum() / args.eval_size}"
    )
    print(
        f"Average bits per image that would need to be stored when compressed: {np.array(image_bits_compressed).sum() / args.eval_size}"
    )
    # include the two 16-bit floats for max and min
    print(
        f"Average bits per image that would need to be stored if quantized (8 bits): {(np.array(image_bits_8).sum() + 16*2) / args.eval_size}"
    )
    print(
        f"Average bits per image that would need to be stored when compressed (8 bits): {(np.array(image_bits_compressed_8).sum() + 16*2) / args.eval_size}"
    )
    print(
        f"Average bits per image that would need to be stored if quantized (16 bits): {np.array(image_bits_16).sum() / args.eval_size}"
    )
    print(
        f"Average bits per image that would need to be stored when compressed (16 bits): {np.array(image_bits_compressed_16).sum() / args.eval_size}"
    )

    print("EMBEDDING -----------")
    print(
        f"Average bits per image that would need to be stored: {np.array(bits).sum() / args.eval_size}"
    )
    # include the two 16-bit floats for max and min
    print(
        f"Average bits per image that would need to be stored if quantized (8 bits): {(np.array(bits_8).sum() + 2*16) / args.eval_size}"
    )

    # include the two 16-bit floats for max and min
    print(
        f"Average bits stored per image when compressed (zip, lossless compression): {np.array(bits_compressed).sum() / args.eval_size}"
    )
    print(
        f"Average bits stored per image when compressed (zip, 8 bits quantization): {(np.array(bits_8_compressed).sum() + 2*16) / args.eval_size}"
    )
    print(
        f"Average bits stored per image when compressed (zip, 5 bits quantization): {(np.array(bits_5_compressed).sum() + 2*16) / args.eval_size}"
    )
    print(
        f"Average bits stored per image when compressed (zip, 4 bits quantization): {(np.array(bits_4_compressed).sum() + 2*16) / args.eval_size}"
    )
    print(
        f"Average bits stored per image when compressed (zip, 3 bits quantization): {(np.array(bits_3_compressed).sum() + 2*16) / args.eval_size}"
    )
    print(
        f"Average bits stored per image when compressed (zip, 2 bits quantization): {(np.array(bits_2_compressed).sum() + 2*16) / args.eval_size}"
    )
    print(
        f"Average bits stored per image when compressed (zip, 16 bits float): {np.array(bits_16_compressed).sum() / args.eval_size}"
    )

    if args.entropy:
        print(f"Average bits per image with learned compression: {av_size}")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
