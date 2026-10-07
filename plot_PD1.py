import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def run_leiden_custom(adata, use_rep, key_added, res=0.5):
    """在指定空间运行 Leiden 聚类"""
    print(f"[Process] Clustering on {use_rep} (res={res})...")
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=15, key_added=f"neigh_{key_added}")
    sc.tl.leiden(adata, resolution=res, neighbors_key=f"neigh_{key_added}", key_added=key_added)
    return adata

def main():
    # ================= 参数设置 (Parameters) =================
    params = {
        "input_h5ad": "results_tcrdist_trb20PD-1150/adata_used_with_latents.h5ad",
        "output_dir": Path("results_tcrdist_trb20PD-1150/complex_analysis"),
        "le_rep": "DeepTracing_UE",
        "k_remove": 20,           # 移除克隆数 k (参数化)
        "top_n_plot": 20,        # 绘图展示的克隆数
        "leiden_res": 0.2,       # Leiden 分辨率
        "point_size": 5,
        # 隐空间列表
        "reps": [ "DeepTracing_UE", "DeepTracing_LE", "DeepTracing_IE"],
        # 分类标签
        "cat_colors": ["cluster", "sample", "patient","leiden_1", "top_clonotypes"],

    }
    params["output_dir"].mkdir(parents=True, exist_ok=True)

    # 1. 加载数据 [cite: 850]
    print(f"Loading data from {params['input_h5ad']}...")
    adata = sc.read(params["input_h5ad"])

    # 2. 第一次 Leiden 聚类 (LE 空间)
    adata = run_leiden_custom(adata, params["le_rep"], "leiden_1", res=params["leiden_res"])

    # 3. 统计并处理 Clonotype
    # 假设 TCR 信息存储在 'tcr_unique_index' 或类似的列中
    tcr_col = "tcr_unique_index" 
    counts = adata.obs[tcr_col].value_counts()
    top_k_clones = counts.head(params["k_remove"]).index.tolist()
    top_n_plot_clones = counts.head(params["top_n_plot"]).index.tolist()

    print(f"Top {params['k_remove']} clones to remove: {top_k_clones}")

    # 标注 Top 20 用于绘图
    adata.obs["top_clonotypes"] = adata.obs[tcr_col].astype(str)
    adata.obs.loc[~adata.obs["top_clonotypes"].isin([str(x) for x in top_n_plot_clones]), "top_clonotypes"] = "Others"

    # 4. 移除 Top k 后进行第二次 Leiden 聚类
    mask_remain = ~adata.obs[tcr_col].isin(top_k_clones)
    adata_sub = adata[mask_remain].copy()
    
    print(f"Cells remaining after removal: {adata_sub.n_obs}")
    adata_sub = run_leiden_custom(adata_sub, params["le_rep"], "leiden_2_tmp", res=params["leiden_res"])
    
    # 将第二次聚类结果映射回原 adata
    adata.obs["leiden_2"] = pd.Series(pd.NA, index=adata.obs_names, dtype="object")
    adata.obs.loc[mask_remain, "leiden_2"] = adata_sub.obs["leiden_2_tmp"].astype(str)
    adata.obs["leiden_2"] = adata.obs["leiden_2"].astype("category")

    # 5. 绘图循环
    for rep in params["reps"]:
        print(f"Processing Embedding: {rep}")
        
        # 对于非内置 UMAP，需要重新计算坐标用于可视化
        if rep != "X_umap":
            sc.pp.neighbors(adata, use_rep=rep, n_neighbors=15)
            sc.tl.umap(adata)
            basis = "umap"
        else:
            basis = "umap" # 直接使用 X_umap

        # A. 绘制分类标签 (Categorical)
        for ck in params["cat_colors"]:
            if ck not in adata.obs.columns: continue
            
            sc.pl.embedding(
                adata, basis=basis, color=ck, size=params["point_size"],
                frameon=False, show=False, title=f"{rep} - {ck}"
            )
            plt.savefig(params["output_dir"] / f"umap_{rep}_{ck}.pdf", bbox_inches="tight")
            plt.close()


    # 6. 保存带有新标签的 adata
    adata.write_h5ad(params["output_dir"] / "adata_final_analysis.h5ad")
    print(f"Analysis complete. Results saved in {params['output_dir']}")

if __name__ == "__main__":
    main()