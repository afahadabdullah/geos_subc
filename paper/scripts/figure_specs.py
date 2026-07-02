FIGURE_SPECS = [
    {
        "filename": "fig1_system_overview.pdf",
        "title": "Figure 1 Placeholder: System Overview",
        "subtitle": "Hybrid dynamical-AI subseasonal workflow",
        "grid": (2, 3),
        "panels": [
            "(a) Observed state channels\nSST, SSS, soil moisture, IVT,\nZ500 zonal deviation, U250, MJO wave",
            "(b) GEOS dynamical guidance\nWeekly PR and T2M leads",
            "(c) Seasonal and lead encodings\nsin(month), cos(month), lead scalar",
            "(d) Multi-target flow-matching U-Net\nShared backbone, lead-specific heads",
            "(e) Ensemble generation\nStructured noise + random blend + variance tempering",
            "(f) Probabilistic evaluation\nCRPS, RMSE, spatial diagnostics",
        ],
        "caption": (
            "System overview for the proposed hybrid dynamical-AI subseasonal forecast framework. "
            "The final figure should show the observed-state channels, GEOS forecast guidance, "
            "seasonal and lead encodings, the multi-target flow-matching model, and the "
            "probabilistic sampling path that produces weekly precipitation and 2 m temperature ensembles."
        ),
    },
    {
        "filename": "fig2_method_detail.pdf",
        "title": "Figure 2 Placeholder: Method Detail",
        "subtitle": "Flow matching, structured noise, and variance-tempered initialization",
        "grid": (2, 3),
        "panels": [
            "(a) Linear interpolation path\nx_t = t x_1 + (1 - t) x_0",
            "(b) Velocity target\nv* = x_1 - x_0",
            "(c) Lead-aware routing\nWeek 1-4 output heads",
            "(d) EOF-based teleconnection prior\nMJO / NAO / ENSO conditioned noise",
            "(e) Random-noise blending\nrho_PR and rho_T2M",
            "(f) Variance-tempered spread control\nbeta_PR and beta_T2M",
        ],
        "caption": (
            "Method figure for the multi-target flow-matching system. The final version should "
            "illustrate the straight-line flow-matching path, the teleconnection-aware EOF "
            "initialization, the random-noise blending coefficients for precipitation and temperature, "
            "the variance-tempered initial state, and the lead-specific output heads."
        ),
    },
    {
        "filename": "fig3_skill_by_lead.pdf",
        "title": "Figure 3 Placeholder: Lead-Dependent Skill",
        "subtitle": "Four-panel summary for CRPS and RMSE across weeks 1-4",
        "grid": (2, 2),
        "panels": [
            "(a) Precipitation CRPS by lead\nGEOS vs Hybrid FlowMatch-S2S",
            "(b) Precipitation RMSE by lead\nGEOS vs Hybrid FlowMatch-S2S",
            "(c) T2M CRPS by lead\nGEOS vs Hybrid FlowMatch-S2S",
            "(d) T2M RMSE by lead\nGEOS vs Hybrid FlowMatch-S2S",
        ],
        "caption": (
            "Lead-dependent skill summary. The final version should plot weekly precipitation "
            "and temperature CRPS and RMSE for GEOS and the hybrid model across weeks 1-4, "
            "with confidence intervals or standard-error bars if multiple cases are aggregated."
        ),
    },
    {
        "filename": "fig4_spatial_maps.pdf",
        "title": "Figure 4 Placeholder: Spatial Diagnostics",
        "subtitle": "Representative cases and aggregate map-based skill",
        "grid": (4, 4),
        "panels": [
            "(a) PR target W1", "(b) PR GEOS mean W1", "(c) PR hybrid mean W1", "(d) PR improvement map W1",
            "(e) PR target W2", "(f) PR GEOS mean W2", "(g) PR hybrid mean W2", "(h) PR improvement map W2",
            "(i) T2M target W1", "(j) T2M GEOS mean W1", "(k) T2M hybrid mean W1", "(l) T2M improvement map W1",
            "(m) T2M target W2", "(n) T2M GEOS mean W2", "(o) T2M hybrid mean W2", "(p) T2M improvement map W2",
        ],
        "caption": (
            "Spatial diagnostics and representative forecast cases. The final figure should compare "
            "targets, GEOS ensemble mean, hybrid forecast mean, and error diagnostics for precipitation "
            "and temperature, ideally with one representative initialization and one aggregate skill map."
        ),
    },
    {
        "filename": "fig5_probabilistic_diagnostics.pdf",
        "title": "Figure 5 Placeholder: Probabilistic Diagnostics",
        "subtitle": "Calibration, ablations, and teleconnection-aware behavior",
        "grid": (2, 3),
        "panels": [
            "(a) CRPS by sampling strategy\nRandom vs structured vs structured+variance",
            "(b) Spread-skill or calibration plot\nEnsemble spread vs observed error",
            "(c) Teleconnection-stratified CRPS\nWeak vs active MJO and other regimes",
            "(d) Checkpoint sweep summary\nValidation CRPS across candidate models",
            "(e) Rank histogram or PIT-style diagnostic\nProbabilistic calibration",
            "(f) Ablation table graphic\nCompact summary of gains and trade-offs",
        ],
        "caption": (
            "Probabilistic diagnostics and ablations. The intended final content is a set of compact "
            "panels showing calibration behavior, CRPS impact of structured noise, teleconnection-"
            "stratified skill, and optional checkpoint or ablation summaries."
        ),
    },
]
