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

scvi_is_trained = True
compute_mde = False
nichevi_is_trained = False

# ----------load data----------#

data_dir, data_file = setup.DATA_FOLDER, setup.DATA_FILE

data_file_name = os.path.splitext(data_file)[0]
print(data_file_name)

path_to_save = os.path.join("../checkpoints", data_file_name)
os.makedirs(path_to_save, exist_ok=True)

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

history_setup = {}

if scvi_is_trained:
    if setup.EXPRESSION_MODEL == "scanvi":
        scvivae = scvi.model.SCANVI.load(
            dir_path=path_to_save
            + f"/{setup.EXPRESSION_MODEL}vae_E"
            + str(setup.N_EPOCHS_SCVI)
            + "_"
            + str(setup.LIKELIHOOD)
            + ".pt",
            adata=adata,
        )
    if setup.EXPRESSION_MODEL == "scvi":
        scvivae = scvi.model.SCVI.load(
            dir_path=path_to_save
            + f"/{setup.EXPRESSION_MODEL}vae_E"
            + str(setup.N_EPOCHS_SCVI)
            + "_"
            + str(setup.LIKELIHOOD)
            + ".pt",
            adata=adata,
        )

latent_key = "X_scVI" if setup.EXPRESSION_MODEL == "scvi" else "X_scanvi"

adata.obsm[latent_key] = scvivae.get_latent_representation(
    batch_size=setup.BATCH_SIZE_SCVI
)


for setting in niche_setup.keys():
    print("[bold green]" + setting + "[/bold green]")
    setup_dict = niche_setup[setting]

    # preprocessing function to populate adata.obsm with the keys 'neighborhood_composition',
    # 'qz1_m', 'qz1_var', 'niche_indexes', 'niche_distances', 'qz1_m_niche_knn', 'qz1_var_niche_knn', 'qz1_m_niche_ct',
    # 'qz1_var_niche_ct'

    path_to_save_nichevae = (
        path_to_save
        + "/nichevae_"
        + setting
        + "_"
        + str(setup.N_EPOCHS_NICHEVI)
        + ".pt"
    )

    if setup_dict["niche_expression"] == "pca":
        NICHE_LIKELIHOOD = "gaussian"
        adata.obsm["qz1_m"] = adata.obsm["X_pca"]

    if setup_dict["niche_expression"] == "scvi":
        NICHE_LIKELIHOOD = "gaussian"
        adata.obsm["qz1_m"] = adata.obsm["X_scanvi"]

    if setup_dict["niche_expression"] == "resolvi":
        NICHE_LIKELIHOOD = "gaussian"
        # adata_resolvi = ad.read_h5ad(
        #     os.path.join(path_to_save, data_file_name + "_nicheVI.h5ad")
        # )
        adata.obsm["qz1_m"] = adata.obsm["X_resolvi"].copy()

    if setup_dict["niche_expression"] == "gaussian_counts":
        NICHE_LIKELIHOOD = "gaussian"
        adata.obsm["qz1_m"] = adata.X.toarray()

    # Extract the obsm array and the 'region' column
    obsm_array = adata.obsm[setup.COORDS]
    region_array = adata.obs[setup.NICHE].values

    # Find the indices of cells matching the condition
    neuron_indices = np.where(
        adata.obs[setup.CELL_TYPE] == setup_dict["type_to_shuffle"]
    )[0]

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

    region_shuffled_obs = (
        setup.NICHE + "_nichevi_" + setup_dict["type_to_shuffle"].split()[0]
    )

    print(
        "Creating new region obs:",
        region_shuffled_obs,
    )

    adata.obs[region_shuffled_obs] = region_array

    setup_kwargs = {
        "labels_key": setup.CELL_TYPE,
        "cell_coordinates_key": setup.COORDS,
        "expression_embedding_key": "qz1_m",
        "expression_embedding_niche_key": "qz1_m_niche_ct",
        "niche_composition_key": "neighborhood_composition",
        "niche_indexes_key": "niche_indexes",
        "niche_distances_key": "niche_distances",
    }

    nichevi.nicheSCVI.preprocessing_anndata(
        adata,
        k_nn=setup.K_NN,
        sample_key=setup.SAMPLE,
        **setup_kwargs,
    )

    nichevi.nicheSCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key=setup.BATCH,
        **setup_kwargs,
    )

    if nichevi_is_trained is False:
        # check if path_to_save_nichevae exists
        if os.path.exists(path_to_save_nichevae):
            print("Model already exists...")
            # get out of the loop
            continue

        nichevae = nichevi.nicheSCVI(
            adata,
            cell_rec_weight=setup_dict["cell_rec_weight"],
            latent_kl_weight=setup_dict["latent_kl_weight"],
            spatial_weight=setup_dict["spatial_weight"],
            niche_rec_weight=setup_dict["niche_rec_weight"],
            compo_rec_weight=setup_dict["compo_rec_weight"],
            niche_likelihood=NICHE_LIKELIHOOD,
            gene_likelihood=setup.LIKELIHOOD,
            n_layers=setup.N_LAYERS,
            n_heads=setup_dict["n_heads"],
            n_tokens_decoder=setup_dict["n_tokens_decoder"],
            n_layers_niche=setup_dict["n_layers_niche"],
            n_layers_compo=setup_dict["n_layers_compo"],
            n_hidden_niche=setup_dict["n_hidden_niche"],
            n_hidden_compo=setup_dict["n_hidden_compo"],
            n_latent=setup_dict["n_latent"],
            use_batch_norm="both" if setup.USE_BATCH_NORM else "none",
            use_layer_norm="none" if setup.USE_BATCH_NORM else "both",
            prior_mixture=setup_dict["prior_mixture"],
            semisupervised=True,
            linear_classifier=True,
            # prior_mixture_k=setup_dict["prior_mixture_k"],
        )

        nichevae.train(
            max_epochs=setup.N_EPOCHS_NICHEVI,
            train_size=0.8,
            validation_size=0.2,
            early_stopping=True,
            check_val_every_n_epoch=1,
            batch_size=setup.BATCH_SIZE_NICHEVI,
            plan_kwargs=dict(
                lr=setup.LR_NICHEVI,
                # n_epochs_kl_warmup=setup.KL_WARMUP,
                n_epochs_kl_warmup=setup_dict["kl_warmup"],
                # n_steps_kl_warmup=setup.N_STEPS_KL_WARMUP,
                # max_kl_weight=setup.MAX_KL_WEIGHT,
                max_kl_weight=setup_dict["max_kl_weight"],
                n_epochs_spatial_warmup=setup.SPATIAL_WARMUP,
                min_spatial_weight=setup.MIN_SPATIAL_WEIGHT,
                max_spatial_weight=setup.MAX_SPATIAL_WEIGHT,
                optimizer=setup.OPTIMIZER,
                weight_decay=setup.WEIGHT_DECAY,
                reduce_lr_on_plateau=setup.REDUCE_LR_ON_PLATEAU,
            ),
            early_stopping_patience=100,  # trick because sometimes KL goes up.
            # lr_scheduler_metric="reconstruction_loss_validation",
        )

        nichevae.save(
            dir_path=path_to_save_nichevae,
            save_anndata=False,
        )

    if nichevi_is_trained:
        nichevae = nichevi.nicheSCVI.load(
            dir_path=path_to_save_nichevae,
            adata=adata,
        )

    # To revert to the initial state:
    adata.obsm[setup.COORDS] = initial_obsm
    adata.obs[setup.NICHE] = initial_region

    latent_key = setting + "_X_nicheVI"

    adata.obsm[latent_key] = nichevae.get_latent_representation(batch_size=1024)

    # history_setup[latent_key] = scvivae.history
    # if compute_mde:
    #     NICHEVI_MDE_KEY = latent_key + "_MDE"
    #     adata.obsm[NICHEVI_MDE_KEY] = scvi.model.utils.mde(adata.obsm[latent_key])

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
        adata_filtered.obs[setup.CELL_TYPE] == setup_dict["type_to_shuffle"]
    ].copy()

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
        label_key=region_shuffled_obs,
        embedding_obsm_keys=embedding_obsm_keys,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcorr,
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, show=False)

    type_to_shuffle = setup_dict["type_to_shuffle"].split()[0]

    plt.savefig(
        f"{setup.FIGURES_FOLDER_NICHEVI}scib_nichevi_{type_to_shuffle}.png",
        bbox_inches="tight",
        dpi=80,
    )

    nichevi_clisi = _lisi_per_cell_type(
        adata_subset,
        latent_key,
        region_shuffled_obs,
    )

    np.save(f"{setup.FIGURES_FOLDER_NICHEVI}cLISI_{type_to_shuffle.split()[0]}.npy", nichevi_clisi)

    adata_subset.obsm["MDE"] = scvi.model.utils.mde(adata_subset.obsm[latent_key])

    sc.pp.neighbors(adata_subset, use_rep=latent_key)
    sc.tl.leiden(
        adata_subset,
        resolution=0.5,
        key_added=f"leiden_nichevi{type_to_shuffle}",
    )

    sc.pl.embedding(
        adata_subset,
        basis="MDE",
        color=["cell_type_niche", f"leiden_nichevi{type_to_shuffle}"],
        frameon=False,
        ncols=1,
        palette="tab20",
        show=False,
    )

    plt.savefig(
        f"{setup.FIGURES_FOLDER_NICHEVI}leiden_{type_to_shuffle}.png",
        bbox_inches="tight",
        dpi=100,
    )


from plot_history import plot_history

plot_history(figure_folder=setup.FIGURES_FOLDER_NICHEVI)


# Save the dictionary to a PKL file
with open(path_to_save + "/models_history.pkl", "wb") as pickle_file:
    pd.to_pickle(history_setup, pickle_file)

# adata.write_h5ad(os.path.join(path_to_save, data_file_name + "_scanVI.h5ad"))
