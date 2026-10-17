"""Shared constants: datasets, splits, model settings and hyperparameter grids
of the paper (Section V and Table 2).

Each dataset is split into training (60%), validation (10%), calibration
(15%) and test (15%) sets. The validation split selects the hyperparameters
of the baselines (IBUG k, LoBoost region size, normalized-CP exponent, RBF
bandwidth), fixes the LPM quartile thresholds and supplies the source points
of the extrapolation sets.
"""

SEEDS = [42, 7, 13, 99, 2024]

# The nine core datasets (file names of the
# inria-soda/tabular-benchmark "reg_num" collection).
MAIN9 = [
    "Ailerons", "elevators", "cpu_act", "superconduct", "Bike_Sharing_Demand",
    "houses", "MiamiHousing2016", "pol", "sulfur",
]
# Remaining numerical regression tasks of the same benchmark.
EXTRA10 = [
    "Brazilian_houses", "abalone", "delays_zurich_transport", "diamonds",
    "house_16H", "house_sales", "medical_charges", "nyc-taxi-green-dec-2016",
    "wine_quality", "yprop_4_1",
]
ALL19 = MAIN9 + EXTRA10


DISPLAY = {
    "Ailerons": "Ailerons", "elevators": "Elevators", "cpu_act": "CPU_act",
    "superconduct": "Superconduct", "Bike_Sharing_Demand": "Bike_Sharing",
    "houses": "Houses", "MiamiHousing2016": "MiamiHousing", "pol": "Pol",
    "sulfur": "Sulfur", "Brazilian_houses": "Brazilian_houses",
    "abalone": "Abalone", "delays_zurich_transport": "Delays_Zurich",
    "diamonds": "Diamonds", "house_16H": "House_16H",
    "house_sales": "House_sales", "medical_charges": "Medical_charges",
    "nyc-taxi-green-dec-2016": "NYC_taxi", "wine_quality": "Wine_quality",
    "yprop_4_1": "Yprop",
}

SPLIT = dict(train=0.60, val=0.10, cal=0.15, test=0.15,
             cap_train=6000, cap_val=1000, cap_cal=1000, cap_test=800)
SPLIT_QUICK = dict(train=0.60, val=0.10, cal=0.15, test=0.15,
                   cap_train=1500, cap_val=300, cap_cal=300, cap_test=300)

# Main GBDT configuration (Table 2).
GBDT_MAIN = dict(n_estimators=100, max_depth=4, min_samples_leaf=20,
                 learning_rate=0.05, subsample=0.8)
LGBM_NUM_LEAVES = 16

ALPHA = 0.10
N_EXT = 400          # size of each extrapolation set
SHIFT_MAG = 3.0      # main shift, in units of the training PC1 standard deviation
ISO_TREES = 100
KNN_K = 10
LOF_K = 20
MAHAL_REG = 1e-4

# Hyperparameter grids for baselines tuned on the validation split.
IBUG_K_GRID = [3, 5, 7, 9, 11, 15, 31, 61, 91, 121, 151, 201, 301, 401, 501,
               601, 701]
LOBOOST_M_GRID = [20, 50, 100]
NCP_GAMMA_GRID = [0.25, 0.5, 1.0, 2.0]
RBF_BW_GRID = [0.25, 0.5, 1.0, 2.0]      # multiples of the median distance

# Leaf-agreement kernels compared for WLACP.
# (kind, gamma): "la" is the leaf agreement LA; "la_norm" divides each tree's indicator by
# the training population of the shared leaf (quantile-regression-forest
# style per-tree normalization); gamma > 1 sharpens the kernel.
WLACP_KERNELS = [("la", 1.0), ("la_norm", 1.0), ("la", 2.0), ("la", 4.0)]
