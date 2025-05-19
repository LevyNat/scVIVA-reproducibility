def compute_hotspot(adata, latent_obsm_key):
    import hotspot

    adata.layers["counts_csc"] = adata.layers["counts"].tocsc()
    hs = hotspot.Hotspot(
        adata,
        layer_key="counts_csc",
        model="danb",
        latent_obsm_key=latent_obsm_key,
        # umi_counts_obs_key="total_counts"
    )

    hs.create_knn_graph(
        weighted_graph=False,
        n_neighbors=30,
    )

    hs_results = hs.compute_autocorrelations(jobs=-1)
    hs_results.head(15)

    adata.uns[f"results_{latent_obsm_key}"] = hs_results.loc[hs_results.FDR < 0.05]

    # Select the genes with significant lineage autocorrelation
    hs_genes = (
        hs_results.loc[hs_results.FDR < 0.05]
        .sort_values("Z", ascending=False)
        # .head(500)
        .index
    )

    # Compute pair-wise local correlations between these genes
    lcz = hs.compute_local_correlations(hs_genes, jobs=-1)

    modules = hs.create_modules(
        min_gene_threshold=15, core_only=True, fdr_threshold=0.05
    )

    module_scores = hs.calculate_module_scores()

    # Convert all column names to strings
    module_scores.columns = module_scores.columns.map(str)

    hs.results.join(hs.modules)
    adata.obsm[f"modules_scores{latent_obsm_key}"] = module_scores
    adata.uns[f"results_{latent_obsm_key}"] = hs_results.join(hs.modules).loc[
        hs_results.FDR < 0.05
    ]

    return adata
