import argparse
import os
from typing import Any, Dict, Optional

import anndata as ad
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

# from ..data.dataset import NicheformerDataset
# from ..models.nicheformer import Nicheformer
from nicheformer.data import NicheformerDataset
from nicheformer.models import Nicheformer


def fine_tune_pretraining(config: Optional[dict[str, Any]] = None) -> None:
    """
    Fine-tune a pre-trained Nicheformer model and save the checkpoint.

    Args:
        config (dict): Configuration dictionary containing all necessary parameters
    """
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Load data
    adata = ad.read_h5ad(config["data_path"])
    technology_mean = np.load(config["technology_mean_path"])

    # Ensure reproducibility
    np.random.seed(42)

    # Create an 80-10-10 split
    adata.obs["nicheformer_split"] = np.random.choice(["train", "val", "test"], size=adata.shape[0], p=[0.8, 0.1, 0.1])

    # Create datasets
    train_dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split="train",
        max_seq_len=config.get("max_seq_len", 4096),
        aux_tokens=config.get("aux_tokens", 30),
        chunk_size=config.get("chunk_size", 1000),
    )

    val_dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split="val",
        max_seq_len=config.get("max_seq_len", 4096),
        aux_tokens=config.get("aux_tokens", 30),
        chunk_size=config.get("chunk_size", 1000),
    )

    test_dataset = NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split="test",
        max_seq_len=config.get("max_seq_len", 4096),
        aux_tokens=config.get("aux_tokens", 30),
        chunk_size=config.get("chunk_size", 1000),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    # Load pre-trained model and modify for fine-tuning
    model = Nicheformer.load_from_checkpoint(
        checkpoint_path=config["checkpoint_path"], strict=False, autoregressive=config.get("autoregressive", False)
    )

    # Ensure `autoregressive` exists
    if not hasattr(model.hparams, "autoregressive"):
        model.hparams.autoregressive = config.get("autoregressive", False)

    # Update model parameters for fine-tuning
    model.lr = config["lr"]
    model.warmup = config["warmup"]
    model.max_epochs = config["max_epochs"]

    # Set up model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["output_dir"], "checkpoints"),
        filename="nicheformer-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    # Configure trainer
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=[1],  # Use GPU 1
        callbacks=[checkpoint_callback],  # Enable checkpoint callback
        default_root_dir=config["output_dir"],
        precision=config.get("precision", 32),
        gradient_clip_val=config.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=config.get("accumulate_grad_batches", 10),
    )

    # Train the model
    print("Training the model...")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Fine-tune a pre-trained Nicheformer model')
    # parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    # args = parser.parse_args()

    # with open(args.config) as f:
    #     config = yaml.safe_load(f)

    config = {
        "data_path": "/home/labs/nyosef/nathanl/nicheformer-benchmark/data/adata_M1_M2_core_6_sections_nicheformer.h5ad",  #'path/to/your/data.h5ad',  # Path to your AnnData file
        "technology_mean_path": "/home/labs/nyosef/nathanl/nicheformer-benchmark/data/model_means/merfish_mean_script.npy",  #'path/to/technology_mean.npy',  # Path to technology mean file
        "checkpoint_path": "/home/labs/nyosef/nathanl/nicheformer-benchmark/nicheformer.ckpt",  # Path to model checkpoint
        "output_path": "data_with_embeddings.h5ad",  # Where to save the result, it is a new h5ad
        "output_dir": ".",  # Directory for any intermediate outputs
        "batch_size": 12,
        "max_seq_len": 1500,
        "aux_tokens": 30,
        "chunk_size": 100,  # to prevent OOM
        "num_workers": 4,
        "precision": 32,
        "embedding_layer": -1,  # Which layer to extract embeddings from (-1 for last layer)
        "embedding_name": "embeddings",  # Name suffix for the embedding key in adata.obsm
        "lr": 1e-5,
        "warmup": 1,
        "max_epochs": 1,
        "autoregressive": False,
    }

    fine_tune_pretraining(config=config)
