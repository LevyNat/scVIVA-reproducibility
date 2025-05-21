import anndata as ad
import pandas as pd
from scipy.sparse import csr_matrix
import scanpy as sc

from rich import print
import os
import matplotlib.pyplot as plt
import colorcet as cc

import scvi
import nichevi

from params import setup, niche_setup

from scib_metrics.benchmark import Benchmarker, BatchCorrection, BioConservation

scvi.settings.seed = 34


# ----------load data----------#

data_dir, data_file = setup.DATA_FOLDER, setup.DATA_FILE

data_file_name = os.path.splitext(data_file)[0]
print(data_file_name)

path_to_save = os.path.join("../checkpoints", data_file_name)
os.makedirs(path_to_save, exist_ok=True)
os.makedirs(setup.FIGURES_FOLDER, exist_ok=True)

adata = ad.read_h5ad(os.path.join(data_dir, data_file))

print("data loaded...")

print(
    "Will save the adata as: ",
    os.path.join(path_to_save, data_file_name + "_nicheVI.h5ad"),
)


# ----------scVI----------#


scvi_is_trained = True
nichevi_is_trained = True
save_scvi = True
compute_mde = False  # deprecated
compute_umap = True
compute_scib = False

PALETTE = "tab20"


print(
    f"{setup.EXPRESSION_MODEL} trained: ",
    scvi_is_trained,
    "nicheVI trained: ",
    nichevi_is_trained,
    "compute MDE: ",
    compute_mde,
    "compute UMAP: ",
    compute_umap,
)

history_setup = {}

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

if compute_mde:
    SCVI_MDE_KEY = "X_scVI_MDE" if setup.EXPRESSION_MODEL == "scvi" else "X_scanvi_MDE"
    adata.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(adata.obsm[latent_key])

    sc.pl.embedding(
        adata,
        basis=SCVI_MDE_KEY,
        color=[setup.CELL_TYPE, setup.BATCH],
        frameon=False,
        ncols=1,
        palette=PALETTE,
        show=False,
    )

    plt.savefig(
        f"{setup.FIGURES_FOLDER}{SCVI_MDE_KEY}.png", bbox_inches="tight", dpi=100
    )

if compute_umap:
    sc.pp.neighbors(adata, use_rep=latent_key)
    sc.tl.umap(adata, min_dist=0.3)

    sc.pl.umap(
        adata,
        color=[setup.CELL_TYPE],
        frameon=False,
        ncols=1,
        palette=PALETTE,
        show=False,
    )

    plt.savefig(
        f"{setup.FIGURES_FOLDER}{latent_key}_umap.png", bbox_inches="tight", dpi=500
    )


history_setup[latent_key] = scvivae.history


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

    if setup_dict["niche_expression"] == "pca":
        NICHE_LIKELIHOOD = "gaussian"
        adata.obsm["qz1_m"] = adata.obsm["X_pca"]

    if setup_dict["niche_expression"] == "scvi":
        NICHE_LIKELIHOOD = "gaussian"
        adata.obsm["qz1_m"] = adata.obsm[latent_key]

    if setup_dict["niche_expression"] == "resolvi":
        NICHE_LIKELIHOOD = "gaussian"
        adata.obsm["qz1_m"] = adata.obsm["X_resolvi"].copy()

    if setup_dict["niche_expression"] == "gaussian_counts":
        NICHE_LIKELIHOOD = "gaussian"
        adata.obsm["qz1_m"] = adata.X.toarray()

    setup_kwargs = {
        "sample_key": setup.SAMPLE,
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
            # n_heads=setup_dict["n_heads"],
            # n_tokens_decoder=setup_dict["n_tokens_decoder"],
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

    # if nichevae.module.n_heads:
    #     print("Computing attention...")
    #     # adata.obsm["attention_weights"] = nichevae.get_niche_attention(batch_size=1024)
    #     # adata.uns["cell_type_attention_weights"] = nichevae.get_cell_type_attention()

    history_setup[setting] = nichevae.history
    adata.obsm[setting + "_X_nicheVI"] = nichevae.get_latent_representation(
        batch_size=1024
    )

    if compute_mde:
        NICHEVI_MDE_KEY = setting + "_MDE"
        adata.obsm[NICHEVI_MDE_KEY] = scvi.model.utils.mde(
            adata.obsm[setting + "_X_nicheVI"]
        )

        sc.pl.embedding(
            adata,
            basis=NICHEVI_MDE_KEY,
            color=[setup.CELL_TYPE, setup.BATCH],
            frameon=False,
            ncols=1,
            palette=PALETTE,
            show=False,
        )

        plt.savefig(
            f"{setup.FIGURES_FOLDER}{NICHEVI_MDE_KEY}.png", bbox_inches="tight", dpi=100
        )

    if compute_umap:
        sc.pp.neighbors(adata, use_rep=setting + "_X_nicheVI")
        sc.tl.umap(adata, min_dist=0.3)

        sc.pl.umap(
            adata,
            color=[setup.CELL_TYPE],
            frameon=False,
            ncols=1,
            palette=PALETTE,
            show=False,
        )

        plt.savefig(
            f"{setup.FIGURES_FOLDER}{setting}_umap.png", bbox_inches="tight", dpi=500
        )

# --- Save history + latent space for benchmarking ---#

adata.layers["counts"] = csr_matrix(adata.layers["counts"])

adata.write_h5ad(
    os.path.join(path_to_save, data_file_name + "_nicheVI.h5ad"),
)

# Save the dictionary to a PKL file
with open(path_to_save + "/models_history.pkl", "wb") as pickle_file:
    pd.to_pickle(history_setup, pickle_file)

from plot_history import plot_history

plot_history()

# ----------Benchmarks----------#

if compute_scib:
    import warnings

    warnings.filterwarnings("ignore")

    # X = adata.obsm["neighborhood_composition"]

    # kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(X)
    # adata.obs["kmeans_alpha_5"] = kmeans.labels_

    # cluster_to_niche = {
    #     0: "DCIS 1 tumor niche",
    #     1: "Peri-tumor area",
    #     2: "Intact tissue, stromal area",
    #     3: "Invasive tumor niche",
    #     4: "DCIS 2 tumor niche",
    # }

    # adata.obs[setup.NICHE] = adata.obs["kmeans_alpha_5"].map(cluster_to_niche)

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

    embedding_obsm_keys = (
        [
            latent_key,
        ]
        + [setting + "_X_nicheVI" for setting in niche_setup.keys()]
        + [
            # "simvi_both_08",
            "banksy_pc_20_lambda_0.2",
            # "simvi_interact_08",
            # "simvi_intrinsic_08",
        ]
    )

    bm = Benchmarker(
        adata,
        batch_key=setup.BATCH,
        label_key=setup.NICHE,
        embedding_obsm_keys=embedding_obsm_keys,
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcorr,
        n_jobs=-1,
    )
    bm.benchmark()
    bm.plot_results_table(min_max_scale=False, show=False)

    plt.savefig(f"{setup.FIGURES_FOLDER}scib_niche.png", bbox_inches="tight", dpi=80)

    bm = Benchmarker(
        adata,
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
