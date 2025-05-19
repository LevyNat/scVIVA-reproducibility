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
from icecream import ic

from params import setup, niche_setup
from pancancer_utils import compute_hotspot

scvi.settings.seed = 34
# Set SVG font type to 'none' to keep text as text in SVG files
plt.rcParams["svg.fonttype"] = "none"

# ----------parameters----------#

compute_hospot = True
compute_umap = True
compute_leiden = False
LEIDEN_RESOLUTION = 0.5

# ----------load data----------#

data_dir, data_file = setup.DATA_FOLDER, setup.DATA_FILE

data_file_name = os.path.splitext(data_file)[0]
print(data_file_name)

path_to_save = os.path.join("../checkpoints", data_file_name)
os.makedirs(path_to_save, exist_ok=True)
os.makedirs(setup.FIGURES_FOLDER, exist_ok=True)

suffix_adata = "_nicheVI_c50.h5ad"

adata = ad.read_h5ad(os.path.join(path_to_save, data_file_name + suffix_adata))

adata = adata[adata.obs[setup.CELL_TYPE].isin(["T Cell"])].copy()
ic(adata.shape)
# sc.pp.filter_genes(adata, min_cells=10)
sc.pp.filter_cells(adata, min_counts=50)
# adata = adata[adata.obs["predicted_celltype_prob"] >= 0.85].copy()
ic(adata.shape)

print(adata)

# Define the samples_organs dictionary
samples_organs = {
    "uterine": [
        "HumanUterineCancerPatient2-RACostain",
        "HumanUterineCancerPatient2-ROCostain",
        "HumanUterineCancerPatient1",
    ],
    "lung": ["HumanLungCancerPatient1", "HumanLungCancerPatient2"],
    "prostate": ["HumanProstateCancerPatient1", "HumanProstateCancerPatient2"],
    "colon": ["HumanColonCancerPatient1", "HumanColonCancerPatient2"],
    "ovarian": [
        "HumanOvarianCancerPatient1",
        "HumanOvarianCancerPatient2Slice1",
        "HumanOvarianCancerPatient2Slice2",
        "HumanOvarianCancerPatient2Slice3",
    ],
    "melanoma": ["HumanMelanomaPatient1", "HumanMelanomaPatient2"],
    "breast": ["HumanBreastCancerPatient1"],
    "liver": ["HumanLiverCancerPatient1", "HumanLiverCancerPatient2"],
}

# Create a sample-to-organ mapping from the samples_organs dictionary
sample_to_organ = {
    sample: organ for organ, samples in samples_organs.items() for sample in samples
}

# Map the `SAMPLE` column in `adata.obs` to the organ using sample_to_organ
adata.obs["organ"] = adata.obs[setup.SAMPLE].map(sample_to_organ)

# ----------run hotspot----------#

latent_obsm_key = "s10_scanvi_lr0.0005_poisson_X_nicheVI"
# latent_obsm_key = "X_scanvi"
# latent_obsm_key = "spatial"


if compute_hospot:
    adata = compute_hotspot(adata, latent_obsm_key)

if compute_umap:
    sc.pp.neighbors(adata, use_rep=latent_obsm_key)
    sc.tl.umap(adata, min_dist=0.3)
    # sc.pl.umap(
    #     adata,
    #     color=[setup.SAMPLE],
    #     frameon=False,
    #     vmin=-1,
    #     vmax="p96",
    #     wspace=0.4,
    #     palette=cc.glasbey_light,
    # )
    # plt.savefig(
    #     os.path.join(setup.FIGURES_FOLDER, data_file_name + "_nicheVI_T_Cell_umap.svg"),
    #     format="svg",
    # )
    # plt.savefig(
    #     os.path.join(setup.FIGURES_FOLDER, data_file_name + "_nicheVI_T_Cell_umap.png"),
    #     format="png",
    # )

if compute_leiden:
    sc.tl.leiden(
        adata,
        resolution=LEIDEN_RESOLUTION,
        key_added=f"leiden_T_Cell_{LEIDEN_RESOLUTION}",
    )
    sc.tl.leiden(adata, resolution=0.3, key_added="leiden_T_Cell_0.3")
    sc.pl.umap(
        adata,
        color=[f"leiden_T_Cell_{LEIDEN_RESOLUTION}", "leiden_T_Cell_0.3", "organ"],
        frameon=False,
        wspace=0.4,
        palette=cc.glasbey_light,
    )
    plt.savefig(
        os.path.join(setup.FIGURES_FOLDER, data_file_name + "_nicheVI_T_Cell_umap.svg"),
        format="svg",
    )
    plt.savefig(
        os.path.join(setup.FIGURES_FOLDER, data_file_name + "_nicheVI_T_Cell_umap.png"),
        format="png",
    )

adata.write_h5ad(
    os.path.join(path_to_save, data_file_name + "_nicheVI_T_Cell_c50_p85.h5ad")
)
