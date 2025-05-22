import scanpy as sc

# import squidpy as sq

import numpy as np
import pandas as pd
import os

import scvi
import matplotlib.pyplot as plt


import torch


BATCH = "brain_section_label"
# BATCH = "sample"

LABEL = "cell_type"

SAMPLE = "brain_section_label"
# SAMPLE = "sample"


N_EPOCHS = 400

K_NN = 20

simvi_is_trained = False

TRAIN_SIZE = 0.8
TRAIN_SIZE_STR = str(TRAIN_SIZE).replace(".", "")

folder = "/simvi_t" + TRAIN_SIZE_STR + "_r5_k20_" + str(N_EPOCHS) + "epochs"


data_dir = "/home/nathanlevy/sda/Data/"  # Liver_VIZGEN/"
data_file_name = "adata_M1_M2_core_6_sections.h5ad"
# data_file_name = "xenium_breast_cancer_S1_R1_2.h5ad"


path_to_save = os.path.join("checkpoints", data_file_name)
os.makedirs(path_to_save, exist_ok=True)
adata = sc.read_h5ad(os.path.join(data_dir, data_file_name))
print(adata)


if adata.raw:
    adata.layers["counts"] = adata.raw.X.copy()


from simvi.model import SimVI

SimVI.setup_anndata(adata, layer="counts", batch_key=BATCH)
edge_index = SimVI.extract_edge_index(
    adata, batch_key=SAMPLE, spatial_key="spatial", n_neighbors=K_NN
)


if not simvi_is_trained:
    model = SimVI(
        adata,
        n_latent=10,
        dropout_rate=0.1,
        kl_weight=1,
        kl_gatweight=1,
        lam_mi=5,
    )
    train_loss, val_loss, train_mask, val_mask = model.train(
        edge_index,
        max_epochs=N_EPOCHS,
        train_size=TRAIN_SIZE,
        validation_size=1 - TRAIN_SIZE,
    )

    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()
    plt.yscale("log")

    plt.savefig(path_to_save + folder + "_loss.png")

    model.save(path_to_save + folder + ".pt")

    adata.uns["simvi_train_mask_" + TRAIN_SIZE_STR] = train_mask
    adata.uns["simvi_val_mask_" + TRAIN_SIZE_STR] = val_mask

    print(adata.uns["simvi_val_mask"])


else:
    model = SimVI.load(
        dir_path=path_to_save + folder + ".pt",
        adata=adata,
    )


adata.obsm["simvi_intrinsic_" + TRAIN_SIZE_STR] = model.get_latent_representation(
    edge_index, representation_kind="intrinsic"
)
adata.obsm["simvi_interact_" + TRAIN_SIZE_STR] = model.get_latent_representation(
    edge_index, representation_kind="interaction"
)


adata.obsm["simvi_both_" + TRAIN_SIZE_STR] = np.concatenate(
    [
        adata.obsm["simvi_interact_" + TRAIN_SIZE_STR],
        adata.obsm["simvi_intrinsic_" + TRAIN_SIZE_STR],
    ],
    axis=1,
)


# SCVI_MDE_KEY = "simvi_interact_MDE_" + TRAIN_SIZE_STR
# adata.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(
#     adata.obsm["simvi_interact_" + TRAIN_SIZE_STR]
# )
# SCVI_MDE_KEY = "simvi_intrinsic_MDE_" + TRAIN_SIZE_STR
# adata.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(
#     adata.obsm["simvi_intrinsic_" + TRAIN_SIZE_STR]
# )
# SCVI_MDE_KEY = "simvi_both_MDE_" + TRAIN_SIZE_STR
# adata.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(
#     adata.obsm["simvi_both_" + TRAIN_SIZE_STR]
# )


print(adata.obsm.keys())

adata.write_h5ad(os.path.join(data_dir, data_file_name))


# model = SimVI.load(
#     dir_path=path_to_save + "/simvi_r5_k20_400epochs.pt",
#     adata=adata,
# )
