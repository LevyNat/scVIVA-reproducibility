import argparse
import os
from typing import Any, Dict, Optional

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

# from ..data.dataset import NicheformerDataset
# from ..models.nicheformer import Nicheformer
from nicheformer.data import NicheformerDataset
from nicheformer.models import Nicheformer


def get_embeddings(config: Optional[dict[str, Any]] = None) -> None:
    """
    Extract embeddings from a pre-trained Nicheformer model and store them in AnnData.

    Args:
        config (dict): Configuration dictionary containing all necessary parameters
    """
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Load data
    adata = ad.read_h5ad(config["data_path"])
    technology_mean = np.load(config["technology_mean_path"])

    adata.obs["modality"] = 4  # spatial
    adata.obs["specie"] = 6  # mouse
    adata.obs["technology"] = 7  # mouse
    adata.obs["nicheformer_split"] = "train"

    print(adata)

    # Create dataset for all cells (no train/val/test split needed)
    dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        # split=None,  # Use all cells
        split="train",  # Use all cells
        max_seq_len=config.get("max_seq_len", 4096),
        aux_tokens=config.get("aux_tokens", 30),
        chunk_size=config.get("chunk_size", 1000),
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    # Load pre-trained model
    model = Nicheformer.load_from_checkpoint(
        checkpoint_path=config["checkpoint_path"],
        map_location=lambda storage, loc: storage.cuda(0) if torch.cuda.is_available() else storage,
    )

    model.eval()  # Set to evaluation mode

    # # Configure trainer
    # trainer = pl.Trainer(
    #     accelerator="gpu" if torch.cuda.is_available() else "cpu",
    #     devices=1,
    #     default_root_dir=config["output_dir"],
    #     precision=config.get("precision", 32),
    # )

    # Get embeddings
    print("Extracting embeddings...")
    embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Get embeddings from the model
            emb = model.get_embeddings(
                batch=batch,
                layer=config.get("embedding_layer", -1),  # Default to last layer
            )
            embeddings.append(emb.cpu().numpy())

    # Concatenate all embeddings
    embeddings = np.concatenate(embeddings, axis=0)

    # Store embeddings in AnnData object
    embedding_key = f"X_niche_{config.get('embedding_name', 'embeddings')}"
    adata.obsm[embedding_key] = embeddings

    # Save updated AnnData
    adata.write_h5ad(config["output_path"])

    print(f"Embeddings saved to {config['output_path']} in obsm['{embedding_key}']")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Extract embeddings from a pre-trained Nicheformer model')
    # parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    # args = parser.parse_args()

    # with open(args.config) as f:
    #     config = yaml.safe_load(f)

    config = {
        "data_path": "/home/labs/nyosef/nathanl/nicheformer-benchmark/data/adata_M1_M2_core_6_sections_nicheformer.h5ad",  #'path/to/your/data.h5ad',  # Path to your AnnData file
        "technology_mean_path": "/home/labs/nyosef/nathanl/nicheformer-benchmark/data/model_means/merfish_mean_script.npy",  #'path/to/technology_mean.npy',  # Path to technology mean file
        "checkpoint_path": "/home/labs/nyosef/nathanl/nicheformer-benchmark/src/nicheformer/experiments/checkpoints/nicheformer-epoch=00-val_loss=0.42.ckpt",  # Path to model checkpoint
        "output_path": "data_with_embeddings_epoch=00.h5ad",  # Where to save the result, it is a new h5ad
        "output_dir": ".",  # Directory for any intermediate outputs
        "batch_size": 32,
        "max_seq_len": 1500,
        "aux_tokens": 30,
        "chunk_size": 1000,  # to prevent OOM
        "num_workers": 4,
        "precision": 32,
        "embedding_layer": -1,  # Which layer to extract embeddings from (-1 for last layer)
        "embedding_name": "embeddings",  # Name suffix for the embedding key in adata.obsm
    }

    get_embeddings(config=config)
