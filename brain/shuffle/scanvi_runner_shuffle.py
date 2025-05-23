import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix

from rich import print
import os
import matplotlib.pyplot as plt
import scvi
import nichevi
import numpy as np
import scanpy as sc

from params import setup, niche_setup

from scib_metrics.benchmark import Benchmarker, BatchCorrection, BioConservation

from nichevi.spatial_analysis import _lisi_per_cell_type

scvi.settings.seed = 34


save_scvi = True
compute_mde = False

# ----------load data----------#

data_dir, data_file = setup.DATA_FOLDER, setup.DATA_FILE

data_file_name = os.path.splitext(data_file)[0]
print(data_file_name)

path_to_save = os.path.join("../checkpoints", data_file_name)
os.makedirs(path_to_save, exist_ok=True)
os.makedirs(setup.FIGURES_FOLDER_SCANVI, exist_ok=True)

adata = ad.read_h5ad(os.path.join(data_dir, data_file))


adata.layers["counts"] = adata.raw.X.copy()
print(adata)

print(
    "Will save the adata as: ",
    os.path.join(path_to_save, data_file_name + "_scanVI.h5ad"),
)


categories = adata.obs[setup.CELL_TYPE].astype(str)

categories[categories.isin(["oligodendrocyte precursor cell"])] = (
    "oligodendrocyte_precursor cell"
)

adata.obs[setup.CELL_TYPE] = categories.astype("category")

types_to_shuffle = [
    "astrocyte",
    "endothelial cell",
    "oligodendrocyte",
    "microglial cell",
    "pericyte",
    "oligodendrocyte_precursor cell",
    "GABAergic neuron",
    "glutamatergic neuron",
]

history_setup = {}

embedding_obsm_keys = []

for type_to_shuffle in types_to_shuffle:
    print(f"Shuffling {type_to_shuffle}")

    # Extract the obsm array and the 'region' column
    obsm_array = adata.obsm[setup.COORDS]
    region_array = adata.obs[setup.NICHE].values

    # Find the indices of cells matching the condition
    neuron_indices = np.where(adata.obs[setup.CELL_TYPE] == type_to_shuffle)[0]

    # Store the initial state
    initial_obsm = obsm_array.copy()
    initial_region = region_array.copy()

    # Extract the subset of obsm and region arrays for these indices
    neuron_obsm = obsm_array[neuron_indices]
    neuron_region = region_array[neuron_indices]

    # Shuffle the neuron indices
    shuffled_indices = np.random.permutation(len(neuron_indices))

    # Apply the shuffled indices to the neuron subset
    shuffled_neuron_obsm = neuron_obsm[shuffled_indices]
    shuffled_neuron_region = neuron_region[shuffled_indices]

    # Replace the original data with the shuffled subset for these cells
    obsm_array[neuron_indices] = shuffled_neuron_obsm
    region_array[neuron_indices] = shuffled_neuron_region

    # Assign the shuffled data back to adata
    adata.obsm[setup.COORDS] = obsm_array
    adata.obs[setup.NICHE] = region_array

    print(
        "Creating new region obs:",
        setup.NICHE + "_scanvi_" + type_to_shuffle.split()[0],
    )

    adata.obs[setup.NICHE + "_scanvi_" + type_to_shuffle.split()[0]] = region_array

    if setup.EXPRESSION_MODEL == "scanvi":
        scvi.model.SCANVI.setup_anndata(
            adata,
            layer="counts",
            unlabeled_category="ignore",
            batch_key=setup.BATCH,
            labels_key=setup.CELL_TYPE,
        )

        scvivae = scvi.model.SCANVI(
            adata,
            gene_likelihood=setup.LIKELIHOOD,
            n_layers=setup.N_LAYERS,
            n_latent=setup.N_LATENT,
            linear_classifier=True,  # seems to improve scib?
        )

    if setup.EXPRESSION_MODEL == "scvi":
        scvi.model.SCVI.setup_anndata(
            adata,
            layer="counts",
            batch_key=setup.BATCH,
        )

        scvivae = scvi.model.SCVI(
            adata,
            gene_likelihood=setup.LIKELIHOOD,
            n_layers=setup.N_LAYERS,
            n_latent=setup.N_LATENT,
        )

    scvivae.train(
        max_epochs=setup.N_EPOCHS_SCVI,
        train_size=0.8,
        validation_size=0.2,
        batch_size=setup.BATCH_SIZE_SCVI,
        plan_kwargs=dict(
            lr=setup.LR_SCVI,
            n_epochs_kl_warmup=setup.KL_WARMUP,
            weight_decay=setup.WEIGHT_DECAY,
            optimizer=setup.OPTIMIZER,
        ),
        # trainer_kwargs=dict(check_val_every_n_epoch=1),
        early_stopping=True,
    )

    if save_scvi:
        scvivae.save(
            dir_path=path_to_save
            + f"/{setup.EXPRESSION_MODEL}vae_E"
            + str(setup.N_EPOCHS_SCVI)
            + "_"
            + type_to_shuffle.split()[0]
            + "_"
            + str(setup.LIKELIHOOD)
            # + "L"
            + ".pt",
            save_anndata=False,
        )

    # To revert to the initial state:
    adata.obsm[setup.COORDS] = initial_obsm
    adata.obs[setup.NICHE] = initial_region

    latent_key = type_to_shuffle.split()[0] + "_X_scanvi"

    adata.obsm[latent_key] = scvivae.get_latent_representation(batch_size=1024)

    history_setup[latent_key] = scvivae.history
    # if compute_mde:
    #     NICHEVI_MDE_KEY = latent_key + "_MDE"
    #     adata.obsm[NICHEVI_MDE_KEY] = scvi.model.utils.mde(adata.obsm[latent_key])



    TRESHOLD = setup.TRESHOLD

    adata.obs["cell_type_niche"] = adata.obs[setup.CELL_TYPE].astype(str) + "_" + adata.obs[setup.NICHE].astype(str)
    value_counts = adata.obs["cell_type_niche"].value_counts()
    cell_types_to_keep = value_counts[value_counts >= TRESHOLD].index
    print("n_obs: ", adata.n_obs)
    adata_filtered = adata[adata.obs["cell_type_niche"].isin(cell_types_to_keep)].copy()
    print("n_obs_filtered: ", adata_filtered.n_obs)

    adata_subset = adata_filtered[adata_filtered.obs[setup.CELL_TYPE] == type_to_shuffle].copy()

    import warnings

    warnings.filterwarnings("ignore")

    batchcorr = BatchCorrection(
        silhouette_batch=True,
        ilisi_knn=True,
        kbet_per_label=True,
        graph_connectivity=True,
        pcr_comparison=False,  # not reliable I think!
    )

    biocons = BioConservation(
        isolated_labels=True,
        nmi_ari_cluster_labels_leiden=True,
        nmi_ari_cluster_labels_kmeans=True,
        silhouette_label=True,
        clisi_knn=True,
    )

    embedding_obsm_keys = [latent_key, "X_pca"]

    bm = Benchmarker(
        adata_subset,
        batch_key=setup.BATCH,
        label_key=setup.NICHE + "_scanvi_" + type_to_shuffle.split()[0],
        embedding_obsm_keys=embedding_obsm_keys,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcorr,
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, show=False)

    plt.savefig(
        f"{setup.FIGURES_FOLDER_SCANVI}scib_scanvi_{type_to_shuffle}", bbox_inches="tight", dpi=80
    )

    scanvi_clisi = _lisi_per_cell_type(
            adata_subset,
            latent_key,
            setup.NICHE + "_scanvi_" + type_to_shuffle.split()[0],
        )


    np.save(f"{setup.FIGURES_FOLDER_SCANVI}cLISI_{type_to_shuffle.split()[0]}.npy", scanvi_clisi)


    adata_subset.obsm['MDE'] = scvi.model.utils.mde(adata_subset.obsm[latent_key])

    sc.pp.neighbors(adata_subset, use_rep=latent_key)
    sc.tl.leiden(adata_subset, resolution=0.5, key_added=f"leiden_scanvi{type_to_shuffle.split()[0]}")

    sc.pl.embedding(
        adata_subset,
        basis="MDE",
        color=["cell_type_niche", f"leiden_scanvi{type_to_shuffle.split()[0]}"],
        frameon=False,
        ncols=1,
        palette="tab20",
        show=False,
    )

    plt.savefig(
        f"{setup.FIGURES_FOLDER_SCANVI}leiden_{type_to_shuffle.split()[0]}.png", bbox_inches="tight", dpi=100
    )


from plot_history import plot_history

plot_history(figure_folder=setup.FIGURES_FOLDER_SCANVI)


# Save the dictionary to a PKL file
with open(path_to_save + "/models_history.pkl", "wb") as pickle_file:
    pd.to_pickle(history_setup, pickle_file)

adata.write_h5ad(os.path.join(path_to_save, data_file_name + "_scanVI.h5ad"))



