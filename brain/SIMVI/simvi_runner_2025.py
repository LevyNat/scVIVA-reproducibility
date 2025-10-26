import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
from simvi.model import SimVI

BATCH = "brain_section_label"
# BATCH = "sample"
LABEL = "cell_type"
SAMPLE = "brain_section_label"
# SAMPLE = "sample"
N_EPOCHS = 500
BATCH_SIZE = 1024
MAE_EPOCHS = 80

K_NN = 20

simvi_is_trained = True

TRAIN_SIZE = 0.8
TRAIN_SIZE_STR = str(TRAIN_SIZE).replace(".", "")

folder = "/simvi25_t" + TRAIN_SIZE_STR + str(N_EPOCHS) + "epochs_mae_" + str(MAE_EPOCHS)


data_dir = "/home/nathanl/Data/"  # Liver_VIZGEN/"
data_file_name = "adata_M1_M2_core_6_sections.h5ad"
# data_file_name = "xenium_breast_cancer_S1_R1_2.h5ad"


path_to_save = os.path.join("checkpoints", data_file_name)
os.makedirs(path_to_save, exist_ok=True)
adata = sc.read_h5ad(os.path.join(data_dir, data_file_name))
print(adata)


if adata.raw:
    adata.layers["counts"] = adata.raw.X.copy()

SimVI.setup_anndata(adata, layer="counts", batch_key=BATCH)
edge_index = SimVI.extract_edge_index(
    adata, batch_key=SAMPLE, spatial_key="spatial", n_neighbors=K_NN
)


if not simvi_is_trained:
    model = SimVI(
        adata,
        # n_latent=10,
        # dropout_rate=0.1,
        # kl_weight=1,
        # kl_gatweight=1,
        # lam_mi=5,
        n_hidden=128,
        n_intrinsic=10,
        n_spatial=10,
        n_layers=1,
        dropout_rate=0,
        use_observed_lib_size=True,
        lam_mi=1000,
        reg_to_use="mmd",
        noising_mode="sampling",
        dis_to_use="zinb",
        permutation_rate=0.25,
        var_eps=1e-4,
        kl_weight=1,
        kl_gatweight=0.01,
        attention_heads=1,
    )
    train_loss, val_loss = model.train(
        edge_index,
        max_epochs=N_EPOCHS,
        train_size=TRAIN_SIZE,
        validation_size=1 - TRAIN_SIZE,
        anneal_epochs=50,
        mae_epochs=MAE_EPOCHS,
        lr=1e-3,
        weight_decay=1e-4,
        use_gpu=True,
        batch_size=BATCH_SIZE,
    )

    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.legend()
    plt.yscale("log")

    plt.savefig(path_to_save + folder + "mae_" + str(MAE_EPOCHS) + "_loss25.png")

    model.save(path_to_save + folder + "mae_" + str(MAE_EPOCHS) + ".pt")


else:
    model = SimVI.load(
        dir_path=path_to_save + folder + "mae_" + str(MAE_EPOCHS) + ".pt",
        adata=adata,
    )


adata.obsm["simvi25_intrinsic_mae_" + str(MAE_EPOCHS)] = model.get_latent_representation(
    edge_index, representation_kind="intrinsic", give_mean=True
)
adata.obsm["simvi25_interact_mae_" + str(MAE_EPOCHS)] = model.get_latent_representation(
    edge_index, representation_kind="interaction", give_mean=True
)
adata.obsm["simvi25_all_mae_" + str(MAE_EPOCHS)] = model.get_latent_representation(
    edge_index, representation_kind="all", give_mean=True
)

adata.obsm["simvi25_both_mae_" + str(MAE_EPOCHS)] = np.concatenate(
    [
        adata.obsm["simvi25_interact_mae_" + str(MAE_EPOCHS)],
        adata.obsm["simvi25_intrinsic_mae_" + str(MAE_EPOCHS)],
    ],
    axis=1,
)


print(adata.obsm.keys())

adata.write_h5ad(os.path.join(data_dir, data_file_name))
