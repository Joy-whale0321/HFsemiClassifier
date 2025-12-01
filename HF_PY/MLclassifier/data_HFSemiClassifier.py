import numpy as np
import uproot
import awkward as ak
import torch
from torch.utils.data import Dataset


class HFSemiClassifier(Dataset):
    """
    Heavy-Flavor Semi-leptonic electron classifier dataset.

    一条样本 = 一条电子：
      - 输入: 电子特征 + 关联的 hadron 点云 (当前是 away-side hadrons)
      - 输出: label = 0(D), 1(B), 2(other)

    参数
    ----
    root_file : str
        由 ppHF_eXDecay 生成的 ROOT 文件路径，例如 "ppHF_eXDecay.root"
    tree_name : str
        TTree 名字，默认 "tree"
    use_log_pt : bool
        是否对 pt 取 log。推荐 True。
    pt_min, pt_max : float or None
        只选择满足 pt_min <= pt_e < pt_max 的电子。
        若为 None 则不在这一侧做 cut。
        例如:
          pt_min=3.0, pt_max=4.0  -> 3–4 GeV 这个 bin
          pt_min=8.0, pt_max=None -> >= 8 GeV
    eta_abs_max : float or None
        |eta_e| 的最大值。若不为 None，则只保留 |eta_e| <= eta_abs_max 的电子，
        且只保留满足 |eta_h| = |eta_e + dEta| <= eta_abs_max 的 hadrons。
        例如：eta_abs_max = 1.0 -> 只用 |eta| <= 1 的区域。
    use_had_eta : bool
        是否在 hadron 特征里使用 Δη 信息。
        若为 False，则第二列 Δη 统一置 0，相当于屏蔽 hadron η。
    dphi_windows : list[tuple[float, float]] or None
        对 hadron 做 Δφ 选择。只保留 Δφ 落在这些区间内的 hadrons。
        Δφ 为生成 ntuple 时定义的 had_phi 分支（已经是 Δφ，通常在 [-pi, pi]）。

        例如:
          dphi_windows = [(-np.pi/2, np.pi/2)]
          dphi_windows = [(-np.pi/3, np.pi/3), (2.0, 3.0)]
        若为 None，则不对 Δφ 做 cut。
    """

    def __init__(
        self,
        root_file: str,
        tree_name: str = "tree",
        use_log_pt: bool = True,
        pt_min: float | None = None,
        pt_max: float | None = None,
        eta_abs_max: float | None = 1.0,
        use_had_eta: bool = True,
        dphi_windows: list[tuple[float, float]] | None = None,  # <<< 新增
    ):
        super().__init__()

        self.root_file = root_file
        self.tree_name = tree_name
        self.use_log_pt = use_log_pt
        self.pt_min = pt_min
        self.pt_max = pt_max
        self.eta_abs_max = eta_abs_max
        self.use_had_eta = use_had_eta
        self.dphi_windows = dphi_windows  # <<< 新增

        # 打开 ROOT 文件并读取需要的分支
        file = uproot.open(self.root_file)
        tree = file[self.tree_name]

        branches = [
            "nEle",
            "ele_charge",
            "ele_pt",
            "ele_eta",
            "ele_phi",
            "ele_hf_TAG",
            "nHad_away",
            "had_fromEle",
            "had_charge",
            "had_pt",
            "had_eta",   # already Δη
            "had_phi",   # already Δφ
        ]

        arrays = tree.arrays(branches, library="ak")

        # 保存为成员变量（都是 awkward.Array，形状 (nEvents, variable-length)）
        self.nEle = arrays["nEle"]
        self.ele_charge = arrays["ele_charge"]
        self.ele_pt = arrays["ele_pt"]
        self.ele_eta = arrays["ele_eta"]
        self.ele_phi = arrays["ele_phi"]
        self.ele_hf_TAG = arrays["ele_hf_TAG"]

        self.nHad_away = arrays["nHad_away"]
        self.had_fromEle = arrays["had_fromEle"]
        self.had_charge = arrays["had_charge"]
        self.had_pt = arrays["had_pt"]
        self.had_deta = arrays["had_eta"]   # already Δη = eta_h - eta_e
        self.had_dphi = arrays["had_phi"]   # already Δφ = phi_h - phi_e

        # 建立 "全局电子索引表"：sample_idx -> (event_idx, ele_local_idx)
        self._build_index()

    def _build_index(self):
        """
        遍历所有 event，为每条电子建立一个全局 index。
        同时在这里做：
          - 只保留 D/B (ele_hf_TAG=1/2)
          - pt bin 选择 (pt_min, pt_max)
          - η cut: 若 eta_abs_max 不为 None，则只保留 |eta_e| <= eta_abs_max
        """
        self.electron_index = []  # list of (evt_idx, ele_idx)

        n_events = len(self.nEle)
        for evt in range(n_events):
            n_ele_evt = int(self.nEle[evt])
            for i_ele in range(n_ele_evt):
                # ----- 0) η cut on electron -----
                eta_e = float(self.ele_eta[evt][i_ele])
                if (self.eta_abs_max is not None) and (abs(eta_e) > self.eta_abs_max):
                    continue

                # ----- 1) 先看是不是 D/B -----
                raw_tag = int(self.ele_hf_TAG[evt][i_ele])
                if raw_tag not in (1, 2):
                    # raw_tag == 0: other，直接丢弃
                    continue

                # ----- 2) 再看 pt bin cut -----
                pt_e = float(self.ele_pt[evt][i_ele])
                if self.pt_min is not None and pt_e < self.pt_min:
                    continue
                if self.pt_max is not None and pt_e >= self.pt_max:
                    continue

                # 通过所有 cut，就保留
                self.electron_index.append((evt, i_ele))

        self._length = len(self.electron_index)
        print(
            f"[HFSemiClassifier] built index with {self._length} electrons "
            f"(pt_min={self.pt_min}, pt_max={self.pt_max}, "
            f"eta_abs_max={self.eta_abs_max}, "
            f"use_had_eta={self.use_had_eta}, "
            f"dphi_windows={self.dphi_windows})"
        )

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        """
        返回一个样本：

        返回 dict:
          {
            "ele_feat": (3,) float32 tensor,
            "had_feat": (N_had, 5) float32 tensor,
            "label":    () long tensor (0/1/2)
          }
        """
        evt_idx, ele_idx = self.electron_index[idx]

        # ---------- 电子特征 ----------
        pt_e = float(self.ele_pt[evt_idx][ele_idx])
        eta_e = float(self.ele_eta[evt_idx][ele_idx])
        charge_e = float(self.ele_charge[evt_idx][ele_idx])  # -1 or +1

        if self.use_log_pt:
            ele_pt_feat = np.log(pt_e + 1e-6)
        else:
            ele_pt_feat = pt_e

        ele_feat = np.array(
            [ele_pt_feat, eta_e, charge_e],
            dtype=np.float32
        )

        # ---------- hadron 点云特征 ----------
        # 先选出来自这条 electron 的 hadrons
        had_fromEle_evt = self.had_fromEle[evt_idx]
        base_mask = (had_fromEle_evt == ele_idx)

        # 把对应 hadron 的变量取出来（先不做 cut）
        had_pt_all = np.array(self.had_pt[evt_idx][base_mask], dtype=np.float32)
        had_deta_all = np.array(self.had_deta[evt_idx][base_mask], dtype=np.float32)
        had_dphi_all = np.array(self.had_dphi[evt_idx][base_mask], dtype=np.float32)
        had_charge_all = np.array(self.had_charge[evt_idx][base_mask], dtype=np.float32)

        # ========== 1) |eta_h| cut ==========
        if self.eta_abs_max is not None and had_deta_all.size > 0:
            had_eta_global = had_deta_all + eta_e     # eta_h = eta_e + dEta
            mask_eta = np.abs(had_eta_global) <= self.eta_abs_max
        else:
            mask_eta = np.ones_like(had_pt_all, dtype=bool)

        # ========== 2) Δφ cut ==========
        if self.dphi_windows is not None and had_dphi_all.size > 0:
            # 初始全部 False
            mask_dphi = np.zeros_like(had_dphi_all, dtype=bool)
            for (lo, hi) in self.dphi_windows:
                # 支持一个或多个区间
                mask_dphi |= (had_dphi_all >= lo) & (had_dphi_all < hi)
        else:
            # 不设 dphi_windows 时，相当于不过 Δφ cut
            mask_dphi = np.ones_like(had_dphi_all, dtype=bool)

        # ========== 3) 综合 mask ==========
        mask = mask_eta & mask_dphi

        had_pt = had_pt_all[mask]
        had_deta = had_deta_all[mask]
        had_dphi = had_dphi_all[mask]
        had_charge = had_charge_all[mask]

        if had_pt.size > 0:
            if self.use_log_pt:
                had_pt_feat = np.log(had_pt + 1e-6)
            else:
                had_pt_feat = had_pt

            sin_dphi = np.sin(had_dphi)
            cos_dphi = np.cos(had_dphi)

            # >>> 如果不用 hadron η，就把这一列置 0
            if self.use_had_eta:
                had_deta_feat = had_deta
            else:
                had_deta_feat = np.zeros_like(had_deta, dtype=np.float32)

            # 组合为 (N_had, 5)
            had_feat = np.stack(
                [had_pt_feat, had_deta_feat, sin_dphi, cos_dphi, had_charge],
                axis=-1
            ).astype(np.float32)
        else:
            # 没有任何 associated hadron 的情况（或 cut 全砍掉了）
            had_feat = np.zeros((0, 5), dtype=np.float32)

        # ---------- label ----------
        raw_tag = int(self.ele_hf_TAG[evt_idx][ele_idx])
        if raw_tag == 1:
            label = 0  # class 0: D
        elif raw_tag == 2:
            label = 1  # class 1: B
        else:
            label = 2  # class 2: other

        # 转成 torch.Tensor
        ele_feat = torch.from_numpy(ele_feat)              # (3,)
        had_feat = torch.from_numpy(had_feat)              # (N_had, 5)
        label = torch.tensor(label, dtype=torch.long)      # ()

        return {
            "ele_feat": ele_feat,
            "had_feat": had_feat,
            "label": label,
        }


def hf_semi_collate(batch):
    """
    自定义 collate_fn，用于 DataLoader，处理 variable-length hadron 点云。

    输入: batch 是一个 list，每个元素是 HFSemiClassifier.__getitem__ 返回的 dict
    输出: 一个打包好的 dict:
      {
        "ele_feat": (B, 3),
        "had_feat": (B, N_max, 5),
        "had_mask": (B, N_max), bool, True=有效点, False=padding
        "label":    (B,)
      }
    """
    batch_size = len(batch)

    # 电子特征：直接 stack
    ele_feats = torch.stack([item["ele_feat"] for item in batch], dim=0)  # (B, 3)

    # label：直接 stack
    labels = torch.stack([item["label"] for item in batch], dim=0)       # (B,)

    # hadron 点云：需要 padding
    n_hads = [item["had_feat"].shape[0] for item in batch]
    max_hads = max(n_hads) if n_hads else 0

    if max_hads == 0:
        # 极端情况：所有样本都没有 hadron
        had_feats = torch.zeros(batch_size, 0, 5, dtype=torch.float32)
        had_mask = torch.zeros(batch_size, 0, dtype=torch.bool)
    else:
        had_feats = torch.zeros(batch_size, max_hads, 5, dtype=torch.float32)
        had_mask = torch.zeros(batch_size, max_hads, dtype=torch.bool)

        for i, item in enumerate(batch):
            h = item["had_feat"]
            n = h.shape[0]
            if n > 0:
                had_feats[i, :n, :] = h
                had_mask[i, :n] = True

    return {
        "ele_feat": ele_feats,   # (B, 3)
        "had_feat": had_feats,   # (B, N_max, 5)
        "had_mask": had_mask,    # (B, N_max)
        "label": labels,         # (B,)
    }
