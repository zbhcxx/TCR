import os
import json
import math
import scipy.sparse as sp
import scanpy as sc
from time import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from DeepTracing import DEEPTRACING
import DeepTracing
print("DeepTracing imported from:", DeepTracing.__file__)
print("Has train_model:", hasattr(DEEPTRACING, "train_model"))

from preprocess import normalize  # 复用你的 normalize：normalize_per_cell/log1p/scale


# tcrdist3
try:
    from tcrdist.repertoire import TCRrep
except Exception as e:
    TCRrep = None


def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x)
    if s.lower() in {"nan", "none"}:
        return ""
    return s.strip()


def _first_token_semicolon(s: str) -> str:
    """
    Handle strings like 'TRBV10-1; TRBV10-2; TRBV10-3' by taking the first non-empty token.
    """
    s = _safe_str(s)
    if not s:
        return ""
    parts = [p.strip() for p in s.split(";")]
    for p in parts:
        if p and p.lower() not in {"nan", "none"}:
            return p
    return ""


def _normalize_allele(gene: str) -> str:
    """
    tcrdist3 prefers IMGT-like names with allele, e.g., TRBV10-1*01.
    If allele is missing, append '*01'.
    """
    gene = _first_token_semicolon(gene)
    if not gene:
        return ""
    return gene if "*" in gene else (gene + "*01")


def build_tcr_string(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """
    Build a per-cell string for bookkeeping / saving to adata.obs.
    """
    out = []
    for _, row in df[cols].iterrows():
        parts = []
        for c in cols:
            v = _safe_str(row.get(c, ""))
            if v:
                parts.append(v)
        out.append("|".join(parts))
    return np.asarray(out, dtype=object)


def compute_tcrdist_distance_matrix_with_map(
    unique_df: pd.DataFrame,
    organism: str = "human",
    w_alpha: float = 1.0,
    w_beta: float = 1.0,
):
    """
    unique_df columns:
      cdr3_a_aa, v_a_gene, j_a_gene,
      cdr3_b_aa, v_b_gene, j_b_gene
    return:
      D: (M_kept, M_kept) float32 distance  (combined)
      kept_df: kept clones in SAME ORDER as D
      old_to_new: old unique -> new clone index (or -1)
    """
    if TCRrep is None:
        raise ImportError("tcrdist3 not available. pip install tcrdist3")

    cell_df = unique_df.copy()
    cell_df["count"] = 1

    tr = TCRrep(
        cell_df=cell_df,
        organism=organism,
        chains=["alpha", "beta"],
    )
    tr.compute_distances()

    clone_df = tr.clone_df.copy()
    kept_df = clone_df[
        ["cdr3_a_aa", "v_a_gene", "j_a_gene", "cdr3_b_aa", "v_b_gene", "j_b_gene"]
    ].reset_index(drop=True)

    # 分别取 α/β 距离，再做加权和
    D_alpha = tr.pw_alpha.astype(np.float32)
    D_beta  = tr.pw_beta.astype(np.float32)
    D = (w_alpha * D_alpha + w_beta * D_beta).astype(np.float32)
    np.fill_diagonal(D, 0.0)

    old_keys = list(zip(
        unique_df["cdr3_a_aa"], unique_df["v_a_gene"], unique_df["j_a_gene"],
        unique_df["cdr3_b_aa"], unique_df["v_b_gene"], unique_df["j_b_gene"],
    ))
    new_keys = list(zip(
        kept_df["cdr3_a_aa"], kept_df["v_a_gene"], kept_df["j_a_gene"],
        kept_df["cdr3_b_aa"], kept_df["v_b_gene"], kept_df["j_b_gene"],
    ))

    key_to_new = {k: i for i, k in enumerate(new_keys)}
    old_to_new = np.full(len(old_keys), -1, dtype=np.int64)

    dropped = 0
    for i, k in enumerate(old_keys):
        j = key_to_new.get(k, -1)
        old_to_new[i] = j
        if j < 0:
            dropped += 1

    print("      tcrdist3 kept clones:", len(kept_df), " / input uniques:", len(unique_df), " (dropped:", dropped, ")")
    if dropped > 0:
        ex = [old_keys[i] for i in np.where(old_to_new < 0)[0][:5]]
        print("      Examples dropped (cdr3a,v_a,j_a,cdr3b,v_b,j_b):", ex)

    return D, kept_df, old_to_new


def select_inducing_points_random(
    unique_indices: np.ndarray,
    n_points: int = 50,
    random_state: int = 0,
) -> np.ndarray:
    """
    在 unique index 空间随机选择 inducing points。
    
    Parameters:
    -----------
    unique_indices : np.ndarray
        All unique indices (M,)
    n_points : int
        Number of inducing points to select
    random_state : int
        Random seed
    
    Returns:
    --------
    np.ndarray of inducing unique indices
    """
    rng = np.random.default_rng(random_state)
    
    # 如果 unique_indices 数量小于 n_points，则全部返回
    M = len(unique_indices)
    if M <= n_points:
        return unique_indices.copy()
    
    # 随机选择 n_points 个索引
    chosen = rng.choice(unique_indices, size=n_points, replace=False)
    chosen = np.sort(chosen)  # 排序以保持一致性
    
    return chosen


def main():
    import argparse

    parser = argparse.ArgumentParser("DeepTracing with TRB_1 TCRdist kernel (human, beta chain)")
    parser.add_argument("--input", default="output_cd4_only.h5ad")
    parser.add_argument("--outdir", default="results_tcrdist_trb15")
    parser.add_argument("--celltype_key", default="cluster")
    parser.add_argument(
        "--tcr_cols",
        default="CDR3(Beta1),V_gene(Beta1),J_gene(Beta1)",
        help="Columns used to build per-cell tcr_string (for saving). Distance always uses CDR3(Beta1)/v/j.",
    )
    parser.add_argument("--random_state", type=int, default=0)
    parser.add_argument("--n_inducing_points", type=int, default=150,
                       help="Number of inducing points to randomly select from unique TCRs")
    # training hyperparams (keep your original defaults)
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--maxiter", default=100, type=int)
    parser.add_argument("--train_size", default=0.95, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-6, type=float)
    parser.add_argument("--noise", default=0.0, type=float)
    parser.add_argument("--dropoutE", default=0.0, type=float)
    parser.add_argument("--dropoutD", default=0.0, type=float)
    parser.add_argument("--encoder_layers", nargs="+", default=[128, 64], type=int)
    parser.add_argument("--decoder_layers", nargs="+", default=[128], type=int)

    parser.add_argument("--GP_dim", default=5, type=int)
    parser.add_argument("--Normal_dim", default=5, type=int)

    parser.add_argument("--dynamicVAE", default=True, type=bool)
    parser.add_argument("--init_beta_gaussian", default=5, type=float)
    parser.add_argument("--min_beta_gaussian", default=1, type=float)
    parser.add_argument("--max_beta_gaussian", default=10, type=float)
    parser.add_argument("--init_beta_gp", default=10, type=float)
    parser.add_argument("--min_beta_gp", default=4, type=float)
    parser.add_argument("--max_beta_gp", default=25, type=float)
    parser.add_argument("--KL_loss", default=0.05, type=float)
    parser.add_argument("--lambda_tc", default=10, type=float)
    parser.add_argument("--num_samples", default=1, type=int)

    parser.add_argument("--fix_inducing_points", default=True, type=bool)
    parser.add_argument("--fixed_gp_params", default=False, type=bool)
    parser.add_argument("--kernel_scale", default=150.0, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--w_alpha", type=float, default=0.5)
    parser.add_argument("--w_beta", type=float, default=0.5)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # save config
    with open(outdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    torch.manual_seed(args.random_state)
    np.random.seed(args.random_state)

    # 1) read data
    print(f"[1/6] Read AnnData: {args.input}")
    adata = sc.read(args.input)

    # 2) validate required TRB_1 fields for distance
    required_cols = ["CDR3(Beta1)", "V_gene(Beta1)", "J_gene(Beta1)",
                     "CDR3(Alpha1)", "V_gene(Alpha1)", "J_gene(Alpha1)"]
    missing_required = [c for c in required_cols if c not in adata.obs.columns]
    if missing_required:
        raise ValueError(f"Missing required TRB_1 columns in adata.obs: {missing_required}")

    if args.celltype_key not in adata.obs.columns:
        raise ValueError(f"celltype_key '{args.celltype_key}' not found in adata.obs")

    # build cleaned TCR table
    tcr_df = adata.obs[required_cols].copy()

    tcr_df["CDR3(Beta1)"] = tcr_df["CDR3(Beta1)"].map(_safe_str)
    tcr_df["V_gene(Beta1)"] = tcr_df["V_gene(Beta1)"].map(_normalize_allele)
    tcr_df["J_gene(Beta1)"] = tcr_df["J_gene(Beta1)"].map(_normalize_allele)

    tcr_df["CDR3(Alpha1)"] = tcr_df["CDR3(Alpha1)"].map(_safe_str)
    tcr_df["V_gene(Alpha1)"] = tcr_df["V_gene(Alpha1)"].map(_normalize_allele)
    tcr_df["J_gene(Alpha1)"] = tcr_df["J_gene(Alpha1)"].map(_normalize_allele)

    valid = (
        (tcr_df["CDR3(Beta1)"].str.len() > 0)
        & (tcr_df["V_gene(Beta1)"].str.len() > 0)
        & (tcr_df["J_gene(Beta1)"].str.len() > 0)
        & (tcr_df["CDR3(Alpha1)"].str.len() > 0)
        & (tcr_df["V_gene(Alpha1)"].str.len() > 0)
        & (tcr_df["J_gene(Alpha1)"].str.len() > 0)
    ).to_numpy()

    print(f"      Total cells: {adata.n_obs}")
    print(f"      Cells with TRB_1(cdr3+v+j): {int(valid.sum())}")
    print("Before filter:", adata.obs[args.celltype_key].value_counts(dropna=False))

    # subset to valid TCR cells
    adata = adata[valid].copy()
    tcr_df = tcr_df.loc[valid].reset_index(drop=True)
    print("After filter:", adata.obs[args.celltype_key].value_counts(dropna=False))

    # build tcr_string for saving (user-configurable columns)
    tcr_cols = [c.strip() for c in args.tcr_cols.split(",") if c.strip()]
    missing_tcr_cols = [c for c in tcr_cols if c not in adata.obs.columns]
    if missing_tcr_cols:
        raise ValueError(f"Missing columns in adata.obs for tcr_string: {missing_tcr_cols}")
    tcr_str = build_tcr_string(adata.obs, tcr_cols)

    # 3) unique mapping in (cdr3, v, j) space
    print("[2/6] Build unique TRB_1 (cdr3+v+j) index mapping ...")
    tuples = list(zip(tcr_df["CDR3(Alpha1)"], tcr_df["V_gene(Alpha1)"], tcr_df["J_gene(Alpha1)"],
                      tcr_df["CDR3(Beta1)"], tcr_df["V_gene(Beta1)"], tcr_df["J_gene(Beta1)"],))
    codes, uniques = pd.factorize(pd.Series(tuples), sort=False)

    indices_all = codes.astype(np.int64)
    unique_df = pd.DataFrame(list(uniques), columns=[
        "cdr3_a_aa", "v_a_gene", "j_a_gene",
        "cdr3_b_aa", "v_b_gene", "j_b_gene",
    ])

    print(f"      Cells used: {adata.n_obs}")
    print(f"      Unique TRB_1 receptors: {len(unique_df)}")

    # 4) distance matrix (unique-level) - TCRdist
    dist_dir = outdir / "distance"
    dist_dir.mkdir(parents=True, exist_ok=True)
    dist_file = dist_dir / "unique_tcrdist_trb1.npy"

    # 不建议在未对齐映射前直接复用缓存；先计算并对齐后再缓存
    # 传入 dist_file 路径，函数内部会自动检查是否存在
    unique_distance, kept_df, old_to_new = compute_tcrdist_distance_matrix_with_map(
        unique_df, organism="human", w_alpha=args.w_alpha, w_beta=args.w_beta
    )

    # 对齐：把 cell->old_unique 的 indices_all 映射到 cell->new_clone
    indices_new = old_to_new[indices_all]  # shape (n_cells,)
    cell_keep = indices_new >= 0

    if not np.all(cell_keep):
        print(f"      Filtering cells due to dropped TCRs by tcrdist3: kept {int(cell_keep.sum())}/{len(cell_keep)}")
        adata = adata[cell_keep].copy()
        indices_new = indices_new[cell_keep]
        tcr_str = tcr_str[cell_keep]  # 若你要保存 tcr_string

    indices_all = indices_new.astype(np.int64)
    unique_df = kept_df  # 让 unique_df 与距离矩阵顺序一致

    # 最关键的断言：防止再出现越界
    M = unique_distance.shape[0]
    max_idx = int(indices_all.max())
    assert max_idx < M, f"indices_all.max()={max_idx} but distance size={M}. mapping failed?"

    # 缓存（可选）
    np.save(dist_file, unique_distance)
    print(f"      Saved distance matrix: {dist_file} shape={unique_distance.shape}")


    # 5) expression preprocessing (keep your original logic)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"]
    else:
        print("[WARN] No adata.layers['counts'] found; using adata.X for HVG. seurat_v3 may fail if X is not counts.")

    n_hvg = 3000
    print(f"[opt] Select HVGs: n_top_genes={n_hvg}")

    sc.pp.filter_genes(adata, min_cells=3)

    if sp.issparse(adata.X):
        if not np.isfinite(adata.X.data).all():
            raise ValueError("Found non-finite values (NaN/Inf) in adata.X sparse data.")
        adata.X = adata.X.astype(np.float64)
    else:
        if not np.isfinite(adata.X).all():
            raise ValueError("Found non-finite values (NaN/Inf) in adata.X.")
        adata.X = adata.X.astype(np.float64)

    try:
        try:
            sc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                n_top_genes=n_hvg,
                subset=True,
                span=0.6,
            )
        except TypeError:
            sc.pp.highly_variable_genes(
                adata,
                flavor="seurat_v3",
                n_top_genes=n_hvg,
                subset=True,
            )
        print(f"[OK] seurat_v3 HVG done. Remaining genes: {adata.n_vars}")
    except ValueError as e:
        msg = str(e)
        if "reciprocal condition number" in msg or "loess" in msg.lower():
            print(f"[WARN] seurat_v3 loess failed: {e}")
            print("[WARN] Fallback to flavor='seurat' (more robust).")
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.highly_variable_genes(
                adata,
                flavor="seurat",
                n_top_genes=n_hvg,
                subset=True,
            )
            print(f"[OK] seurat HVG done. Remaining genes: {adata.n_vars}")
        else:
            raise

    print(f"      After HVG: n_vars={adata.n_vars}")

    # 6) normalize expression
    print("[4/6] Normalize expression (size_factors/log1p/scale) ...")
    adata = normalize(adata, size_factors=True, normalize_input=True, logtrans_input=True)


    # 7) select inducing points (unique-index space)
    print(f"[5/6] Select inducing points by {args.celltype_key} ...")
    unique_indices = np.arange(M)
    
    inducing_unique = select_inducing_points_random(
        unique_indices=unique_indices,
        n_points=args.n_inducing_points,
        random_state=args.random_state,
    )
    print(f"      Selected inducing points (unique): {len(inducing_unique)}")

    # 【强制去重逻辑】
    print("      [Processing] Removing inducing points with duplicate distances...")
    # 提取这些点之间的距离子矩阵
    sub_D = unique_distance[np.ix_(inducing_unique, inducing_unique)]
    
    # 标记要保留的点
    keep_mask = np.ones(len(inducing_unique), dtype=bool)
    
    # 遍历检查：如果距离为0且不是同一个点，则删掉后面的
    for i in range(len(inducing_unique)):
        if not keep_mask[i]: continue
        # 找到与当前点距离为0的所有其他点
        # 这是一个简单的贪心策略
        dists = sub_D[i]
        # 找到所有距离为0的索引
        dups = np.where(dists < 1e-5)[0] # 使用小阈值而不是严格0
        for d in dups:
            if d > i: # 如果是后面的点，删掉
                keep_mask[d] = False
    
    inducing_unique = inducing_unique[keep_mask]
    print(f"      Cleaned inducing points (deduplicated): {len(inducing_unique)}")

    # batch size auto
    if args.batch_size == "auto":
        if adata.X.shape[0] <= 1024:
            batch_size = 128
        elif adata.X.shape[0] <= 2048:
            batch_size = 256
        else:
            batch_size = 512
    else:
        batch_size = int(args.batch_size)
    print(f"      batch_size={batch_size}")

    # outputs
    model_file = outdir / "model.pt"
    ue_file = outdir / "latent_UE.csv"
    le_file = outdir / "latent_LE_gp.csv"
    ie_file = outdir / "latent_IE_gaussian.csv"
    used_adata_file = outdir / "adata_used_with_latents.h5ad"

    # build + train model
    print("[6/6] Train DeepTracing (GP prior on TRB_1 TCRdist + Gaussian prior on intrinsic) ...")
    model = DEEPTRACING(
        input_dim=adata.n_vars,
        GP_dim=args.GP_dim,
        Normal_dim=args.Normal_dim,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        noise=args.noise,
        encoder_dropout=args.dropoutE,
        decoder_dropout=args.dropoutD,
        distance=unique_distance,
        initial_inducing_points=inducing_unique,
        fixed_inducing_points=args.fix_inducing_points,
        fixed_gp_params=args.fixed_gp_params,
        kernel_scale=args.kernel_scale,
        N_train=adata.n_obs,
        KL_loss=args.KL_loss,
        dynamicVAE=args.dynamicVAE,
        init_beta_gaussian=args.init_beta_gaussian,
        min_beta_gaussian=args.min_beta_gaussian,
        max_beta_gaussian=args.max_beta_gaussian,
        init_beta_gp=args.init_beta_gp,
        min_beta_gp=args.min_beta_gp,
        max_beta_gp=args.max_beta_gp,
        dtype=torch.float64,
        device=args.device,
        lambda_tc=args.lambda_tc,
    )

    if not model_file.exists():
        t0 = time()
        model.train_model(
            indices=indices_all,
            ncounts=adata.X,
            raw_counts=adata.raw.X,
            size_factors=adata.obs["size_factors"].to_numpy(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=batch_size,
            num_samples=args.num_samples,
            train_size=args.train_size,
            maxiter=args.maxiter,
            patience=args.patience,
            save_model=True,
            model_weights=str(model_file),
        )
        print(f"      Training time: {int(time() - t0)} sec")
    else:
        print(f"      Load existing model: {model_file}")
        model.load_model(str(model_file))

    # export latents
    # 【核心修改】：不要使用 adata.X.shape[0]，改用 batch_size 或 512
    export_batch_size = 512 
    print(f"      Exporting latents (batch_size={export_batch_size})...")
    
    ue, le, ie = model.batching_latent_samples(
        X=indices_all, 
        Y=adata.X, 
        batch_size=export_batch_size
    )
    
    np.savetxt(ue_file, ue, delimiter=",")
    np.savetxt(le_file, le, delimiter=",")
    np.savetxt(ie_file, ie, delimiter=",")

    # attach to adata and save
    adata.obsm["mindTCR_UE"] = ue
    adata.obsm["mindTCR_TE"] = le
    adata.obsm["mindTCR_IE"] = ie
    adata.obs["tcr_string"] = tcr_str.astype(str)
    adata.obs["tcr_unique_index"] = indices_all

    adata.write_h5ad(used_adata_file)
    print(f"Saved: {ue_file}")
    print(f"Saved: {le_file}")
    print(f"Saved: {ie_file}")
    print(f"Saved: {used_adata_file}")
    print(f"Saved distance: {dist_file}")


if __name__ == "__main__":
    main()
