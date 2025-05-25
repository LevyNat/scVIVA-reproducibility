from dataclasses import dataclass
from typing import Optional

import seaborn as sns


@dataclass(frozen=True)
class PARAMS:
    "set all params for our analysis"

    DATA_FOLDER: str = "/home/nathanlevy/sda/Data/"
    # DATA_FOLDER: str = "/home/labs/nyosef/Collaboration/SpatialDatasets/merfish/whole_brain_atlas_zhuang/"

    FIGURES_FOLDER: str = "figures_journal/"

    FIGURES_FOLDER_SHUFFLE = FIGURES_FOLDER + "cLISI/"

    DATA_FILE: str = "adata_M1_M2_core_6_sections.h5ad"

    # for the data
    COORDS: str = "spatial"
    BATCH: str = "brain_section_label"
    SAMPLE: str = "brain_section_label"
    CELL_TYPE: str = "cell_type"
    NICHE: str = "major_brain_region"

    # for scVI
    EXPRESSION_MODEL = "scanvi"
    N_LAYERS: int = 1  
    N_LATENT: int = 10
    LIKELIHOOD: str = "poisson"
    ######################
    N_EPOCHS_SCVI: int = 1000
    N_EPOCHS_RESOLVI: int = 500
    LR_SCVI: float = 1e-4
    BATCH_SIZE_SCVI: int = 1024
    OPTIMIZER: str = "Adam"
    ######################
    WEIGHT_DECAY: float = 1e-6
    KL_WARMUP: int | None = 400
    N_STEPS_KL_WARMUP: int | None = None
    MAX_KL_WEIGHT: float = 1

    # for nicheVI
    K_NN: int = 20
    N_LAYERS_NICHE: int = 1  
    N_LAYERS_COMPO: int = 1
    N_HIDDEN: int = 128
    N_HIDDEN_COMPO: int = 128  
    N_HIDDEN_NICHE: int = 128
    N_LATENT_NICHEVI: int = 10
    N_HEADS: int | None = None
    N_TOKENS_DECODER: int | None = None
    ######################
    N_EPOCHS_NICHEVI: int = 1000
    LR_NICHEVI: float = 5e-4
    BATCH_SIZE_NICHEVI: int = 1024
    REDUCE_LR_ON_PLATEAU: bool = True
    SPATIAL_WARMUP: int | None = None
    MIN_SPATIAL_WEIGHT: float = 1.0 if SPATIAL_WARMUP is None else 0
    MAX_SPATIAL_WEIGHT: float = 1.0
    USE_LAYER_NORM: bool = True
    USE_BATCH_NORM: bool = False

    # for nichetype scib
    TRESHOLD: int = 400


setup = PARAMS()

# ----- NicheVI settings -----#
niche_setup = {
    # f"rvi_s20_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
    #     "cell_rec_weight": 1,
    #     "latent_kl_weight": 1.0,
    #     "spatial_weight": 20,
    #     "niche_rec_weight": 20,
    #     "compo_rec_weight": 20,
    #     "n_latent": setup.N_LATENT_NICHEVI,
    #     "n_layers_niche": setup.N_LAYERS_NICHE,
    #     "n_layers_compo": setup.N_LAYERS_COMPO,
    #     "n_hidden_niche": setup.N_HIDDEN_NICHE,
    #     "n_hidden_compo": setup.N_HIDDEN_COMPO,
    #     "niche_expression": "resolvi",
    #     "kl_warmup": 400,
    #     "max_kl_weight": 1,
    #     "n_heads": None,
    #     "n_tokens_decoder": setup.N_TOKENS_DECODER,
    #     "prior_mixture": False,
    #     "prior_mixture_k": 1,
    # },
    # f"rvi_s10_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
    #     "cell_rec_weight": 1,
    #     "latent_kl_weight": 1.0,
    #     "spatial_weight": 10,
    #     "niche_rec_weight": 10,
    #     "compo_rec_weight": 10,
    #     "n_latent": setup.N_LATENT_NICHEVI,
    #     "n_layers_niche": setup.N_LAYERS_NICHE,
    #     "n_layers_compo": setup.N_LAYERS_COMPO,
    #     "n_hidden_niche": setup.N_HIDDEN_NICHE,
    #     "n_hidden_compo": setup.N_HIDDEN_COMPO,
    #     "niche_expression": "resolvi",
    #     "kl_warmup": 400,
    #     "max_kl_weight": 1,
    #     "n_heads": None,
    #     "n_tokens_decoder": setup.N_TOKENS_DECODER,
    #     "prior_mixture": False,
    #     "prior_mixture_k": 1,
    # },
    f"s10_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
        "cell_rec_weight": 1,
        "latent_kl_weight": 1.0,
        "spatial_weight": 10,
        "niche_rec_weight": 10,
        "compo_rec_weight": 10,
        "n_latent": setup.N_LATENT_NICHEVI,
        "n_layers_niche": setup.N_LAYERS_NICHE,
        "n_layers_compo": setup.N_LAYERS_COMPO,
        "n_hidden_niche": setup.N_HIDDEN_NICHE,
        "n_hidden_compo": setup.N_HIDDEN_COMPO,
        "niche_expression": "scvi",
        "kl_warmup": 400,
        "max_kl_weight": 1,
        "prior_mixture": False,
        "prior_mixture_k": 1,
    },
    # f"mog_s10_w400_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
    #     "cell_rec_weight": 1,
    #     "latent_kl_weight": 1.0,
    #     "spatial_weight": 10,
    #     "niche_rec_weight": 10,
    #     "compo_rec_weight": 10,
    #     "n_latent": setup.N_LATENT_NICHEVI,
    #     "n_layers_niche": setup.N_LAYERS_NICHE,
    #     "n_layers_compo": setup.N_LAYERS_COMPO,
    #     "n_hidden_niche": setup.N_HIDDEN_NICHE,
    #     "n_hidden_compo": setup.N_HIDDEN_COMPO,
    #     "niche_expression": "scvi",
    #     "kl_warmup": 400,
    #     "max_kl_weight": 1,
    #     "prior_mixture": True,
    #     # "prior_mixture_k": 1,
    # },
    # f"mog_s20_w400_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
    #     "cell_rec_weight": 1,
    #     "latent_kl_weight": 1.0,
    #     "spatial_weight": 20,
    #     "niche_rec_weight": 20,
    #     "compo_rec_weight": 20,
    #     "n_latent": setup.N_LATENT_NICHEVI,
    #     "n_layers_niche": setup.N_LAYERS_NICHE,
    #     "n_layers_compo": setup.N_LAYERS_COMPO,
    #     "n_hidden_niche": setup.N_HIDDEN_NICHE,
    #     "n_hidden_compo": setup.N_HIDDEN_COMPO,
    #     "niche_expression": "scvi",
    #     "kl_warmup": 400,
    #     "max_kl_weight": 1,
    #     "prior_mixture": True,
    #     # "prior_mixture_k": 1,
    # },
    # f"rvi_c50_mog_s20_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
    #     "cell_rec_weight": 1,
    #     "latent_kl_weight": 1.0,
    #     "spatial_weight": 20,
    #     "niche_rec_weight": 20,
    #     "compo_rec_weight": 20,
    #     "n_latent": setup.N_LATENT_NICHEVI,
    #     "n_layers_niche": setup.N_LAYERS_NICHE,
    #     "n_layers_compo": setup.N_LAYERS_COMPO,
    #     "n_hidden_niche": setup.N_HIDDEN_NICHE,
    #     "n_hidden_compo": setup.N_HIDDEN_COMPO,
    #     "niche_expression": "resolvi",
    #     "kl_warmup": 400,
    #     "max_kl_weight": 1,
    #     "n_heads": None,
    #     "n_tokens_decoder": setup.N_TOKENS_DECODER,
    #     "prior_mixture": True,
    #     "prior_mixture_k": 1,
    # },
    # f"scanvi_c50_mog_s20_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
    #     "cell_rec_weight": 1,
    #     "latent_kl_weight": 1.0,
    #     "spatial_weight": 20,
    #     "niche_rec_weight": 20,
    #     "compo_rec_weight": 20,
    #     "n_latent": setup.N_LATENT_NICHEVI,
    #     "n_layers_niche": setup.N_LAYERS_NICHE,
    #     "n_layers_compo": setup.N_LAYERS_COMPO,
    #     "n_hidden_niche": setup.N_HIDDEN_NICHE,
    #     "n_hidden_compo": setup.N_HIDDEN_COMPO,
    #     "niche_expression": "scvi",
    #     "kl_warmup": 400,
    #     "max_kl_weight": 1,
    #     "n_heads": None,
    #     "n_tokens_decoder": setup.N_TOKENS_DECODER,
    #     "prior_mixture": True,
    #     "prior_mixture_k": 1,
    # },
    # f"rvi_c20_mog_semi_ln_s20_h64_w400_{setup.EXPRESSION_MODEL}_lr{str(setup.LR_NICHEVI)}_{setup.LIKELIHOOD}": {
    #     "cell_rec_weight": 1,
    #     "latent_kl_weight": 1.0,
    #     "spatial_weight": 20,
    #     "niche_rec_weight": 20,
    #     "compo_rec_weight": 20,
    #     "n_latent": setup.N_LATENT_NICHEVI,
    #     "n_layers_niche": setup.N_LAYERS_NICHE,
    #     "n_layers_compo": setup.N_LAYERS_COMPO,
    #     "n_hidden_niche": 64,
    #     "n_hidden_compo": setup.N_HIDDEN_COMPO,
    #     "niche_expression": "resolvi",
    #     "kl_warmup": 400,
    #     "max_kl_weight": 1,
    #     "n_heads": None,
    #     "n_tokens_decoder": setup.N_TOKENS_DECODER,
    #     "prior_mixture": True,
    #     "prior_mixture_k": 1,
    # },
}


# ----- Plotting settings -----#


# Niche Categories
niches_categories = [
    "n/a",
    "Fiber_tracts",
    "Hypothalamus",
    "Olfactory",
    "Striatum",
    "Ventricular_systems",
    "Cortical_subplate",
    "Pallidum",
    "Thalamus",
    "Isocortex",
    "Hippocampus",
    "Midbrain",
    "Cerebellum",
    "Medulla",
    "Pons",
]

# Generate a tab20 colormap for the categories
tab20_colors = sns.color_palette("tab20", n_colors=len(niches_categories))
niche_color_dict = dict(zip(niches_categories, tab20_colors))


# Cell types
cell_types = [
    "oligodendrocyte",
    "glutamatergic neuron",
    "oligodendrocyte precursor cell",
    "microglial cell",
    "astrocyte",
    "GABAergic neuron",
    "endothelial cell",
    "vascular leptomeningeal cell",
    "pericyte",
    "lymphocyte",
    "smooth muscle cell",
    "dopaminergic neuron",
    "tanycyte",
    "ependymal cell",
    "macrophage",
    "neuroblast (sensu Vertebrata)",
    "dendritic cell",
    "choroid plexus epithelial cell",
    "histaminergic neuron",
    "monocyte",
    "hypendymal cell",
    "olfactory ensheathing cell",
    "glycinergic neuron",
    "Bergmann glial cell",
    "cholinergic neuron",
]

# Generate a tab20 colormap for the cell types
viridis_colors = sns.color_palette("tab20", n_colors=len(cell_types))
cell_type_color_dict = dict(zip(cell_types, viridis_colors))
