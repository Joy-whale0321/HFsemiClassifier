# data.py
# Dataset for HF semi-leptonic electron classification using point-cloud of hadrons
#
# 使用前提：有一个由 ppHF_eXDecay.cc 生成的 ROOT 文件：
#   - 文件内有 TTree "tree"
#   - 分支包含：
#       nEle, ele_charge, ele_E, ele_pt, ele_eta, ele_phi, ele_hf_TAG, ele_is_semileptonic
#       nHad_away, had_fromEle, had_charge, had_pt, had_eta, had_phi
#
# 每个样本 = 一条电子：
#   - ele_feat: [log(pt_e), eta_e, charge_e]
#   - had_feat: N_had × 5 矩阵，列为 [log(pt_h), dEta, sin(dPhi), cos(dPhi), charge_h]
#   - label: 0 (D), 1 (B), 2 (other)
#
# 注意：当前 ROOT 只保存了 "away-side" hadrons，本 Dataset 就用这些作为点云。
# 如果你以后在 ntuple 里加入 full-event hadrons 分支，逻辑可以类似扩展。

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
    """

    def __init__(self, root_file: str, tree_name: str = "tree", use_log_pt: bool = True):
        super().__init__()

        self.root_file = root_file
        self.tree_name = tree_name
        self.use_log_pt = use_log_pt

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
        """
        self.electron_index = []  # list of (evt_idx, ele_idx)

        n_events = len(self.nEle)
        for evt in range(n_events):
            n_ele_evt = int(self.nEle[evt])
            for i_ele in range(n_ele_evt):
                # 如果你想额外加 cut（比如 pt>某值），可以在这里判断
                # self.electron_index.append((evt, i_ele))
               
                # 只保留 D(1) 和 B(2)，跳过 other(0)
                # 在这里读 raw_tag
                raw_tag = int(self.ele_hf_TAG[evt][i_ele])
                if raw_tag == 1 or raw_tag == 2:
                    self.electron_index.append((evt, i_ele))
                else:
                    # raw_tag == 0: other
                    continue

        self._length = len(self.electron_index)

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
        # 当前 ntuple 中 only has away-side hadrons via had_fromEle
        had_fromEle_evt = self.had_fromEle[evt_idx]
        had_mask = (had_fromEle_evt == ele_idx)

        # 选出属于这条电子的 hadrons
        had_pt = np.array(self.had_pt[evt_idx][had_mask], dtype=np.float32)
        had_deta = np.array(self.had_deta[evt_idx][had_mask], dtype=np.float32)
        had_dphi = np.array(self.had_dphi[evt_idx][had_mask], dtype=np.float32)
        had_charge = np.array(self.had_charge[evt_idx][had_mask], dtype=np.float32)

        if had_pt.size > 0:
            if self.use_log_pt:
                had_pt_feat = np.log(had_pt + 1e-6)
            else:
                had_pt_feat = had_pt

            sin_dphi = np.sin(had_dphi)
            cos_dphi = np.cos(had_dphi)

            # 组合为 (N_had, 5)
            had_feat = np.stack(
                [had_pt_feat, had_deta, sin_dphi, cos_dphi, had_charge],
                axis=-1
            ).astype(np.float32)
        else:
            # 没有任何 associated hadron 的情况（极少，但为了代码健壮）
            had_feat = np.zeros((0, 5), dtype=np.float32)

        # ---------- label ----------
        # ele_hf_TAG: 1(D), 2(B), 0(other) in your generation code
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
