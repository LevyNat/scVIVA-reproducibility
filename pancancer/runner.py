import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix
import scanpy as sc
import numpy as np

from rich import print
import os
import matplotlib.pyplot as plt
import scvi
import nichevi
import time
import colorcet as cc

from params import setup, niche_setup

from scib_metrics.benchmark import Benchmarker, BatchCorrection, BioConservation

scvi.settings.seed = 34
# Set SVG font type to 'none' to keep text as text in SVG files
plt.rcParams["svg.fonttype"] = "none"

# ----------load data----------#

data_dir, data_file = setup.DATA_FOLDER, setup.DATA_FILE

data_file_name = os.path.splitext(data_file)[0]
print(data_file_name)

path_to_save = os.path.join("../checkpoints", data_file_name)
os.makedirs(path_to_save, exist_ok=True)
os.makedirs(setup.FIGURES_FOLDER, exist_ok=True)

adata = ad.read_h5ad(os.path.join(data_dir, data_file))

print(adata)
saving_suffix = "_nicheVI_c50.h5ad"

print(
    "Will save the adata as: ",
    os.path.join(path_to_save, data_file_name + saving_suffix),
)


# ----------scVI----------#


scvi_is_trained = False
nichevi_is_trained = False
save_scvi = True
compute_mde = False
compute_umap = True
compute_scib = False
scib_mde_umap_sample = 10

print(
    f"{setup.EXPRESSION_MODEL} trained: ",
    scvi_is_trained,
    "nicheVI trained: ",
    nichevi_is_trained,
    "compute MDE: ",
    f"{scib_mde_umap_sample}% of cells" if compute_mde else False,
    "compute scib: ",
    f"{scib_mde_umap_sample}% of cells" if compute_scib else False,
    "compute UMAP: ",
    f"{scib_mde_umap_sample}% of cells" if compute_umap else False,
)

history_setup = {}

# if either compute_mde or scib_sample is True, we need to sample cells from the dataset:

if compute_mde or compute_scib or compute_umap:
    # Calculate the number of cells to sample
    n_cells = int((scib_mde_umap_sample / 100) * adata.n_obs)

    # Randomly select indices of the cells
    random_indices = np.random.choice(adata.n_obs, size=n_cells, replace=False)

    # Create a new AnnData object with the sampled cells
    adata_sampled = adata[random_indices, :].copy()

    print(f"Sampled {n_cells} cells out of {adata.n_obs} total cells.")


if scvi_is_trained is False:
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

        latent_key = "X_scanvi"

    elif setup.EXPRESSION_MODEL == "scvi":
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

        latent_key = "X_scVI"

    else:
        raise ValueError(f"{setup.EXPRESSION_MODEL} is an invalid expression model")
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

    adata.obsm[latent_key] = scvivae.get_latent_representation(
        batch_size=setup.BATCH_SIZE_SCVI
    )

    if save_scvi:
        scvivae.save(
            dir_path=path_to_save
            + f"/{setup.EXPRESSION_MODEL}vae_E"
            + str(setup.N_EPOCHS_SCVI)
            + "_"
            + str(setup.LIKELIHOOD)
            # + "L"
            + ".pt",
            save_anndata=False,
        )

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
        latent_key = "X_scanvi"
        adata.obsm[latent_key] = scvivae.get_latent_representation(
            batch_size=setup.BATCH_SIZE_SCVI
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
        latent_key = "X_scVI"
        adata.obsm[latent_key] = scvivae.get_latent_representation(
            batch_size=setup.BATCH_SIZE_SCVI
        )
    if setup.EXPRESSION_MODEL == "resolvi":
        latent_key = "X_resolvi"


if compute_mde:
    SCVI_MDE_KEY = f"X_{setup.EXPRESSION_MODEL}_MDE"
    adata_sampled.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(
        adata_sampled.obsm[latent_key]
    )

    sc.pl.embedding(
        adata_sampled,
        basis=SCVI_MDE_KEY,
        color=[setup.CELL_TYPE, setup.BATCH],
        frameon=False,
        ncols=1,
        palette=cc.glasbey_light,
        show=False,
    )

    plt.savefig(
        f"{setup.FIGURES_FOLDER}{SCVI_MDE_KEY}.png", bbox_inches="tight", dpi=1000
    )

if compute_umap:
    sc.pp.neighbors(adata, use_rep=latent_key)
    sc.tl.umap(adata, min_dist=0.3)
    sc.pl.umap(
        adata,
        color=[setup.CELL_TYPE, setup.BATCH],
        frameon=False,
        ncols=1,
        palette=cc.glasbey_light,
        show=False,
    )

    plt.savefig(
        f"{setup.FIGURES_FOLDER}{latent_key}_umap.png", bbox_inches="tight", dpi=1000
    )
    plt.savefig(
        f"{setup.FIGURES_FOLDER}{latent_key}_umap.svg", bbox_inches="tight", dpi=1000
    )


history_setup[latent_key] = scvivae.history

print(scvivae.history.keys())


print(f"{setup.EXPRESSION_MODEL} done...")


# ----------NicheVI----------#


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

    # if setup_dict["niche_expression"] == "pca":
    #     NICHE_LIKELIHOOD = "gaussian"
    #     adata.obsm["qz1_m"] = adata.obsm["X_pca"]

    # if setup_dict["niche_expression"] == "scvi":
    #     NICHE_LIKELIHOOD = "gaussian"
    #     adata.obsm["qz1_m"] = adata.obsm[latent_key]

    # if setup_dict["niche_expression"] == "resolvi":
    #     NICHE_LIKELIHOOD = "gaussian"
    #     # adata_resolvi = ad.read_h5ad(
    #     #     os.path.join(path_to_save, data_file_name + "_nicheVI.h5ad")
    #     # )
    #     adata.obsm["qz1_m"] = adata.obsm["X_resolvi"].copy()

    # if setup_dict["niche_expression"] == "gaussian_counts":
    #     NICHE_LIKELIHOOD = "gaussian"
    #     adata.obsm["qz1_m"] = adata.X.toarray()

    setup_kwargs = {
        "sample_key": setup.SAMPLE,
        "labels_key": setup.CELL_TYPE,
        "cell_coordinates_key": setup.COORDS,
        "expression_embedding_key": latent_key,
        "expression_embedding_niche_key": "qz1_m_niche_ct",
        "niche_composition_key": "neighborhood_composition",
        "niche_indexes_key": "niche_indexes",
        "niche_distances_key": "niche_distances",
    }

    if "qz1_m_niche_ct_FUN" not in adata.obsm:
        # Start time
        start_time = time.time()

        nichevi.nicheSCVI.preprocessing_anndata(
            adata,
            k_nn=setup.K_NN,
            **setup_kwargs,
        )

        # End time
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Convert to minutes and seconds
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60

        # Print the time taken in minutes and seconds
        print(f"Preprocessing time: {minutes} minutes and {seconds:.2f} seconds")

        adata.write_h5ad(
            os.path.join(data_dir, data_file),
        )
    else:
        print("Using existing neighborhood composition and expression embedding...")

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
            niche_likelihood="gaussian",
            gene_likelihood=setup.LIKELIHOOD,
            n_layers=setup.N_LAYERS,
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
            prior_mixture_k=setup_dict["prior_mixture_k"],
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

    history_setup[setting] = nichevae.history
    print(nichevae.history.keys())
    adata.obsm[setting + "_X_nicheVI"] = nichevae.get_latent_representation(
        batch_size=30000
    )

    if compute_mde:
        NICHEVI_MDE_KEY = setting + "_MDE"
        # Create a boolean mask that is True for sampled indices
        # mask = adata.obs.index.isin(adata_sampled.obs.index)
        # adata_sampled.obsm[NICHEVI_MDE_KEY] = scvi.model.utils.mde(
        #     adata.obsm[setting + "_X_nicheVI"][mask]
        # )
        adata_sampled.obsm[setting + "_X_nicheVI"] = nichevae.get_latent_representation(
            adata_sampled, batch_size=3000
        )
        adata_sampled.obsm[NICHEVI_MDE_KEY] = scvi.model.utils.mde(
            adata_sampled.obsm[setting + "_X_nicheVI"]
        )

        sc.pl.embedding(
            adata_sampled,
            basis=NICHEVI_MDE_KEY,
            color=[setup.CELL_TYPE, setup.BATCH],
            frameon=False,
            ncols=1,
            palette=cc.glasbey_light,
            show=False,
        )

        plt.savefig(
            f"{setup.FIGURES_FOLDER}{NICHEVI_MDE_KEY}.png",
            bbox_inches="tight",
            dpi=1000,
        )
    if compute_umap:
        sc.pp.neighbors(adata, use_rep=setting + "_X_nicheVI")
        sc.tl.umap(adata, min_dist=0.3)
        sc.pl.umap(
            adata,
            color=[setup.CELL_TYPE, setup.BATCH],
            frameon=False,
            ncols=1,
            palette=cc.glasbey_light,
            show=False,
        )

        plt.savefig(
            f"{setup.FIGURES_FOLDER}{setting}_X_nicheVI_umap.png",
            bbox_inches="tight",
            dpi=1000,
        )
        plt.savefig(
            f"{setup.FIGURES_FOLDER}{setting}_X_nicheVI_umap.svg",
            bbox_inches="tight",
            dpi=1000,
        )

# --- Save history + latent space for benchmarking ---#

adata.layers["counts"] = csr_matrix(adata.layers["counts"])

print(adata)
print("Saving adatas...")

adata.write_h5ad(
    os.path.join(path_to_save, data_file_name + saving_suffix),
)

# adata_sampled.write_h5ad(
#     os.path.join(path_to_save, data_file_name + "_nicheVI_sampled.h5ad"),
# )

# Save the dictionary to a PKL file
with open(path_to_save + "/models_history.pkl", "wb") as pickle_file:
    pd.to_pickle(history_setup, pickle_file)


from plot_history import plot_history

plot_history()

# ----------Benchmarks----------#

if compute_scib:
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
        nmi_ari_cluster_labels_kmeans=False,
        silhouette_label=True,
        clisi_knn=True,
    )

    # from harmony import harmonize

    # adata.obsm["X_banksy_02_harmony"] = harmonize(
    #     adata.obsm["banksy_pc_20_lambda_0.2"], adata.obs, batch_key=setup.BATCH
    # )

    # adata.obsm["X_banksy_04_harmony"] = harmonize(
    #     adata.obsm["banksy_pc_20_lambda_0.4"], adata.obs, batch_key=setup.BATCH
    # )

    sc.pp.pca(adata_sampled, n_comps=20)

    embedding_obsm_keys = (
        [latent_key]
        + [setting + "_X_nicheVI" for setting in niche_setup.keys()]
        + ["X_pca"]
    )

    # bm = Benchmarker(
    #     adata,
    #     batch_key=setup.BATCH,
    #     label_key=setup.NICHE,
    #     embedding_obsm_keys=embedding_obsm_keys,
    #     bio_conservation_metrics=biocons,
    #     batch_correction_metrics=batchcorr,
    #     n_jobs=-1,
    # )
    # bm.benchmark()
    # bm.plot_results_table(min_max_scale=False, show=False)

    # plt.savefig(f"{setup.FIGURES_FOLDER}scib_niche.png", bbox_inches="tight", dpi=80)

    bm = Benchmarker(
        adata_sampled,
        batch_key=setup.BATCH,
        label_key=setup.CELL_TYPE,
        embedding_obsm_keys=embedding_obsm_keys,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcorr,
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, show=False)

    plt.savefig(
        f"{setup.FIGURES_FOLDER}scib_cell_type.png", bbox_inches="tight", dpi=80
    )

    # adata.obs["cell_type_niche"] = (
    #     adata.obs[setup.CELL_TYPE].astype(str)
    #     + "_"
    #     + adata.obs[setup.NICHE].astype(str)
    # )

    # value_counts = adata.obs["cell_type_niche"].value_counts()
    # cell_types_to_keep = value_counts[value_counts >= 1000].index
    # adata_subset = adata[adata.obs["cell_type_niche"].isin(cell_types_to_keep)]

    # bm = Benchmarker(
    #     adata_subset,
    #     batch_key=setup.BATCH,
    #     label_key="cell_type_niche",
    #     embedding_obsm_keys=embedding_obsm_keys,
    #     bio_conservation_metrics=biocons,
    #     batch_correction_metrics=batchcorr,
    #     n_jobs=-1,
    # )

    # bm.benchmark()
    # bm.plot_results_table(min_max_scale=False, show=False)

    # plt.savefig(
    #     f"{setup.FIGURES_FOLDER}scib_cell_type_niche.png",
    #     bbox_inches="tight",
    #     dpi=80,
    # )
