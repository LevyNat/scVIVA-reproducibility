import os
import numpy as np
import pandas as pd
from IPython.display import display
import warnings

import scvi

warnings.filterwarnings("ignore")

import scipy.sparse as sparse
from scipy.io import mmread
from scipy.stats import pearsonr, pointbiserialr

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

import seaborn as sns
import scanpy as sc


from dataclasses import dataclass
from typing import Optional
from rich import print

sc.logging.print_header()
sc.set_figure_params(facecolor="white", figsize=(8, 8))
sc.settings.verbosity = 1  # errors (0), warnings (1), info (2), hints (3)
plt.rcParams["font.family"] = "Arial"
sns.set_style("white")

import random

from monitor_cpu import monitor_cpu_and_memory

# Note that BANKSY itself is deterministic, here the seeds affect the umap clusters and leiden partition
# seed = 1234
# np.random.seed(seed)
# random.seed(seed)


@dataclass(frozen=True)
class PARAMS:
    "set all params for our analysis"

    DATA_FOLDER: str = "/home/nathanlevy/sda/Data/"
    # DATA_FOLDER: str = "/home/labs/nyosef/Collaboration/SpatialDatasets/merfish/whole_brain_atlas_zhuang/"
    DATA_FILE: str = "adata_M1_M2_core_6_sections.h5ad"
    # DATA_FILE: str = "proseg_resolvi_scanvi_xenium_breast_cancer_S1_R1_2.h5ad"

    FIGURES_FOLDER: str = "figures/"

    # for the data
    COORDS: str = "spatial"
    BATCH: str = "brain_section_label"
    SAMPLE: str = "brain_section_label"
    CELL_TYPE: str = "cell_type"
    NICHE: str = "major_brain_region"

    # for the data
    # COORDS: str = "spatial"
    # BATCH: str = "sample"
    # SAMPLE: str = "sample"
    # CELL_TYPE: str = "predictions_resolvi_proseg_coarse_corrected"
    # NICHE: str = "kmeans_alpha_5"

    # for nichetype scib
    TRESHOLD: int = 400


setup = PARAMS()

# data_path = "/home/nathanlevy/sda/Data/adata_M1_M2_core_10_sections.h5ad"
# data_path = "/home/labs/nyosef/Collaboration/SpatialDatasets/merfish/whole_brain_atlas_zhuang/adata_M1_M2.h5ad"
data_path = os.path.join(setup.DATA_FOLDER, setup.DATA_FILE)
# data_path = "/home/nathanlevy/Data/xenium_breast_cancer_S1_R1_2.h5ad"

adata = sc.read_h5ad(data_path)
adata


categories = adata.obs[setup.CELL_TYPE].astype(str)
categories[categories.isin(["oligodendrocyte precursor cell"])] = (
    "oligodendrocyte_precursor cell"
)
adata.obs[setup.CELL_TYPE] = categories.astype("category")


def run_banksy(adata=adata, type_to_shuffle=None, seed=1234, setup=setup):
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")

    np.random.seed(seed)
    random.seed(seed)

    if type_to_shuffle:
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

        adata.obs[setup.NICHE + "_banksy_" + type_to_shuffle.split()[0]] = region_array

    # Calulates QC metrics and put them in place to the adata object
    # sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], log1p=True, inplace=True)
    from banksy.main import median_dist_to_nearest_neighbour

    # set params
    # ==========
    plot_graph_weights = True
    k_geom = 20  # number of spatial neighbours
    max_m = 1  # use both mean and AFT
    nbr_weight_decay = (
        "scaled_gaussian"  # can also choose "reciprocal", "uniform" or "ranked"
    )

    # Find median distance to closest neighbours
    nbrs = median_dist_to_nearest_neighbour(adata, key="spatial")
    from banksy.initialize_banksy import initialize_banksy

    banksy_dict = initialize_banksy(
        adata,
        "spatial",
        k_geom,
        nbr_weight_decay=nbr_weight_decay,
        max_m=max_m,
        plt_edge_hist=False,
        plt_nbr_weights=False,
        plt_agf_angles=False,  # takes long time to plot
        plt_theta=False,
    )

    from banksy.embed_banksy import generate_banksy_matrix

    # The following are the main hyperparameters for BANKSY
    # -----------------------------------------------------
    resolutions = [0.7]  # clustering resolution for UMAP
    pca_dims = [20]  # Dimensionality in which PCA reduces to
    # lambda_list = [0.2, 0.5]  # list of lambda parameters
    lambda_list = [0.2]  # list of lambda parameters

    banksy_dict, banksy_matrix = generate_banksy_matrix(
        adata, banksy_dict, lambda_list, max_m
    )

    from banksy.main import concatenate_all

    banksy_dict["nonspatial"] = {
        # Here we simply append the nonspatial matrix (adata.X) to obtain the nonspatial clustering results
        0.0: {
            "adata": concatenate_all([adata.X], 0, adata=adata),
        }
    }

    print(banksy_dict["nonspatial"][0.0]["adata"])

    from banksy_utils.umap_pca import pca_umap

    pca_umap(
        adata,
        banksy_dict,
        pca_dims=pca_dims,
        add_umap=False,
        plt_remaining_var=False,
        type_to_shuffle=type_to_shuffle,
    )

    # adata.write(data_path)

    from harmony import harmonize

    if not type_to_shuffle:
        adata.obsm["X_banksy_02_harmony"] = harmonize(
            adata.obsm["banksy_pc_20_lambda_0.2"],
            adata.obs,
            batch_key=setup.BATCH,
        )

    # adata.obsm["X_banksy_04_harmony"] = harmonize(
    #     adata.obsm["banksy_pc_20_lambda_0.4"],
    #     adata.obs,
    #     batch_key=setup.BATCH,
    # )

    # adata.obsm["X_banksy_06_harmony"] = harmonize(
    #     adata.obsm["banksy_pc_20_lambda_0.6"], adata.obs, batch_key="brain_section_label"
    # )

    if type_to_shuffle:
        adata.obsm[f"X_banksy_02_harmony_{type_to_shuffle.split()[0]}"] = harmonize(
            adata.obsm[f"banksy_pc_20_lambda_0.2_{type_to_shuffle.split()[0]}"],
            adata.obs,
            batch_key=setup.BATCH,
        )

        # To revert to the initial state:
        adata.obsm[setup.COORDS] = initial_obsm
        adata.obs[setup.NICHE] = initial_region

    print(adata.obsm.keys())

    adata.write(data_path)


def _lisi_per_cell_type(
    adatype, embedding_key, label_key, n_neighbors=90, perplexity=30
):
    from scib_metrics.nearest_neighbors import NeighborsResults
    from scib_metrics import clisi_knn
    from sklearn.neighbors import NearestNeighbors

    X, labels = adatype.obsm[embedding_key], adatype.obs[label_key]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="kd_tree").fit(X)
    dists, inds = nbrs.kneighbors(X)
    neigh_results = NeighborsResults(indices=inds, distances=dists)
    lisi_res = clisi_knn(
        neigh_results, labels, perplexity=perplexity, return_median=False
    )
    return lisi_res


def banksy_benchmark(
    adata=adata,
    type_to_shuffle=None,
    seed=1234,
    setup=setup,
    leiden_mde=False,
    return_scib=False,
    scib_per_cell=False,
):
    from scib_metrics.benchmark import Benchmarker, BatchCorrection, BioConservation

    import warnings

    warnings.filterwarnings("ignore")

    np.random.seed(seed)
    random.seed(seed)

    TRESHOLD = setup.TRESHOLD

    adata.obs["cell_type_niche"] = (
        adata.obs[setup.CELL_TYPE].astype(str)
        + "_"
        + adata.obs[setup.NICHE].astype(str)
    )
    value_counts = adata.obs["cell_type_niche"].value_counts()
    cell_types_to_keep = value_counts[value_counts >= TRESHOLD].index
    print("n_obs: ", adata.n_obs)
    adata_filtered = adata[adata.obs["cell_type_niche"].isin(cell_types_to_keep)].copy()
    print("n_obs_filtered: ", adata_filtered.n_obs)

    adata_subset = adata_filtered[
        adata_filtered.obs[setup.CELL_TYPE] == type_to_shuffle
    ].copy()

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

    latent_key = (
        f"X_banksy_02_harmony_{type_to_shuffle.split()[0]}"
        if type_to_shuffle
        else "X_banksy_02_harmony"
    )

    embedding_obsm_keys = (
        (
            latent_key,
            "X_pca",
        )
        if type_to_shuffle
        else (latent_key, "X_pca")
    )

    bm = Benchmarker(
        adata_subset,
        batch_key=setup.BATCH,
        label_key=setup.NICHE + "_banksy_" + type_to_shuffle.split()[0],
        embedding_obsm_keys=embedding_obsm_keys,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcorr,
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, show=False)

    plt.savefig(
        f"{setup.FIGURES_FOLDER}scib_{type_to_shuffle}", bbox_inches="tight", dpi=80
    )

    if leiden_mde:
        adata_subset.obsm["MDE"] = scvi.model.utils.mde(adata_subset.obsm[latent_key])

        sc.pp.neighbors(adata_subset, use_rep=latent_key)
        sc.tl.leiden(
            adata_subset,
            resolution=0.5,
            key_added=f"leiden_banksy{type_to_shuffle.split()[0]}",
        )

        sc.pl.embedding(
            adata_subset,
            basis="MDE",
            color=["cell_type_niche", f"leiden_banksy{type_to_shuffle.split()[0]}"],
            frameon=False,
            ncols=1,
            palette="tab20",
            show=False,
        )

        plt.savefig(
            f"{setup.FIGURES_FOLDER}leiden_{type_to_shuffle.split()[0]}.png",
            bbox_inches="tight",
            dpi=100,
        )

    if return_scib:
        scib_type = bm.get_results(min_max_scale=False)

        leiden_NMI, leiden_ARI, cLISI = (
            scib_type["Leiden NMI"][latent_key],
            scib_type["Leiden ARI"][latent_key],
            scib_type["cLISI"][latent_key],
        )
        leiden_NMI_pca, leiden_ARI_pca, cLISI_pca = (
            scib_type["Leiden NMI"]["X_pca"],
            scib_type["Leiden ARI"]["X_pca"],
            scib_type["cLISI"]["X_pca"],
        )
        return [leiden_NMI, leiden_ARI, cLISI], [
            leiden_NMI_pca,
            leiden_ARI_pca,
            cLISI_pca,
        ]

    if scib_per_cell:
        # adata_subset.obs[latent_key + "_cLISI"] = _lisi_per_cell_type(
        #     adata_subset,
        #     latent_key,
        #     setup.NICHE + "_banksy_" + type_to_shuffle.split()[0],
        # )
        # adata_subset.obs["X_pca_cLISI"] = _lisi_per_cell_type(
        #     adata_subset, "X_pca", setup.NICHE + "_banksy_" + type_to_shuffle.split()[0]
        # )

        # adata_subset.write(
        #     setup.DATA_FILE.split()[0] + "_" + type_to_shuffle.split()[0] + ".h5ad"
        # )

        banksy_clisi = _lisi_per_cell_type(
            adata_subset,
            latent_key,
            setup.NICHE + "_banksy_" + type_to_shuffle.split()[0],
        )

        np.save(
            f"{setup.FIGURES_FOLDER}cLISI_{type_to_shuffle.split()[0]}.npy",
            banksy_clisi,
        )


print("Running BANKSY")
# monitor_cpu_and_memory(run_banksy)


# RUNING BANKSY with different cell types---------------------------------------

for type in [
    "oligodendrocyte_precursor cell",
    "microglial cell",
    "pericyte",
    "oligodendrocyte",
    "glutamatergic neuron",
    "GABAergic neuron",
    "endothelial cell",
    "astrocyte",
]:
    print(f"Running BANKSY for {type}")
    run_banksy(adata, type_to_shuffle=type)


for type in [
    "oligodendrocyte_precursor cell",
    "microglial cell",
    "pericyte",
    "oligodendrocyte",
    "glutamatergic neuron",
    "GABAergic neuron",
    "endothelial cell",
    "astrocyte",
]:
    print(f"Running BANKSY benchmark for {type}")
    banksy_benchmark(adata, type_to_shuffle=type, setup=setup, scib_per_cell=True)

# run_banksy(adata, type_to_shuffle=None)

# RUNING BANKSY with different seeds---------------------------------------

# scib_dict = {"Banksy": np.empty((20, 3)), "PCA": np.empty((20, 3))}

# for _seed in range(20):
#     run_banksy(adata, type_to_shuffle="oligodendrocyte", seed=_seed, setup=setup)
#     banksy, pca = banksy_benchmark(
#         adata,
#         type_to_shuffle="oligodendrocyte",
#         seed=_seed,
#         setup=setup,
#         return_scib=True,
#     )

#     scib_dict["Banksy"][_seed] = banksy
#     scib_dict["PCA"][_seed] = pca


# # Save to an .npz file
# np.savez("scib_seeds_scores_oligodendrocyte.npz", **scib_dict)
