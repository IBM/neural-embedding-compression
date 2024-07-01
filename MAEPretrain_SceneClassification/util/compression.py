# Copyright contributors to the neural-embedding-compression project

from typing import Any, Dict, Mapping, cast

import torch
import torch.distributed as dist
from compressai.registry import OPTIMIZERS
from einops import rearrange
from torch import nn
from tqdm import tqdm

from MAEPretrain_SceneClassification.util.misc import is_dist_avail_and_initialized


def net_aux_optimizer(
    net: nn.Module, conf: Mapping[str, Any]
) -> Dict[str, torch.optim.Optimizer]:
    """
    Slightly modified version from CompressAI, allowing frozen parameters
    Returns separate optimizers for net and auxiliary losses.

    Each optimizer operates on a mutually exclusive set of parameters.
    """
    parameters = {
        "net": {
            name
            for name, param in net.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles")
        },
        "aux": {
            name
            for name, param in net.named_parameters()
            if param.requires_grad and name.endswith(".quantiles")
        },
    }

    # Make sure we don't have an intersection of parameters
    params_dict = {name: param for name, param in net.named_parameters() if param.requires_grad}
    inter_params = parameters["net"] & parameters["aux"]
    union_params = parameters["net"] | parameters["aux"]
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0
    
    def make_optimizer(key):
        kwargs = dict(conf[key])
        del kwargs["type"]
        params = (params_dict[name] for name in sorted(parameters[key]))
        return OPTIMIZERS[conf[key]["type"]](params, **kwargs)

    optimizer = {key: make_optimizer(key) for key in ["net", "aux"]}

    return cast(Dict[str, torch.optim.Optimizer], optimizer)

def findEmbeddingSize(model, dataloader, extract_data_ld=None, device=0):
    """
    Given a compression model and a dataloader, return the size of the embeddings in bits

        Parameters:
            model: A valid Compression model
            dataloader: A valid dataloader

        Returns:
            bytes_used (List): List where each element is the number of bytes to store the embedding of that batch
    """
    model.update()
    model.eval()
    bytes_used = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            if extract_data_ld:
                data = extract_data_ld(data)
            else:
                data = data[0]
            data = data.to(next(model.parameters()).device)
            compressed_data = model.compress(data)

            # make sure you can recover the data exactly
            recovered = model.decompress(
                compressed_data["strings"],
                compressed_data["shape"],
            )

            uncompressed_embedding = model.quantized_embedding(data)
            assert (recovered == uncompressed_embedding).all()
            # bytes used + 4 for the size (int32) for each string

            total_bytes_batch = 0
            for component in compressed_data["strings"]:
                total_bytes_batch += sum(len(s) for s in component)
                total_bytes_batch += 4
                
            bytes_used.append(total_bytes_batch)

    bytes_used = torch.tensor(bytes_used)
    if not is_dist_avail_and_initialized():
        return bytes_used
    bytes_used = bytes_used.to(device=device)
    dist.barrier()
    all_bytes_used = [
        torch.zeros_like(bytes_used) for _ in range(dist.get_world_size())
    ]  # world_size is the total number of gpu processes you are running. 4 in your case.
    print(dist.get_world_size())
    all_bytes_used = torch.cat(dist.all_gather(all_bytes_used, bytes_used))
    return all_bytes_used
