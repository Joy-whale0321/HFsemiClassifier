// ppHF_eXDecay.cc
// 生成带 heavy-flavor 半轻衰变电子 + away-side hadrons 的 ntuple（TTree）
//
// 编译：
// g++ -std=c++17 ppHF_eXDecay.cc \
//     `pythia8-config --cxxflags --libs` \
//     `fastjet-config --cxxflags --libs` \
//     `root-config --cflags --libs` \
//     -o ppHF_eXDecay
//
// 运行：
// ./ppHF_eXDecay
//
// 输出：ppHF_eXDecay.root，里面有 TTree "tree"（一 event 一 entry）

#include "Pythia8/Pythia.h"
#include "TFile.h"
#include "TTree.h"

#include <iostream>
#include <cmath>
#include <vector>

using namespace Pythia8;

// ===================== 工具函数 =====================

// open charm meson：PDG 400~499
bool isCharmMeson(int id)
{
    int pdg = std::abs(id);
    return (pdg >= 400 && pdg < 500);
}

// open bottom meson：PDG 500~599
bool isBottomMeson(int id)
{
    int pdg = std::abs(id);
    return (pdg >= 500 && pdg < 600);
}

// 只关心“直接来自 D/B → e ν X”的电子：
//   1. 看这条电子的母粒子 mom 是不是 D/B meson
//   2. 在 mom 的直接子代里：同时包含这条 e 和 ν_e/ν̄_e (|id|=12)
//   满足则：flavor = 1(D), 2(B)，并返回 true；否则 false
bool isDirectDBSemiLeptonicToElectron(const Event& ev,
                                      int eleIdx,
                                      int& flavor,
                                      int& hadronPdg,
                                      float& hadronPt)
{
    flavor    = 0;
    hadronPdg = 0;
    hadronPt  = 0.0f;

    if (eleIdx < 0 || eleIdx >= ev.size()) return false;

    const Particle& e = ev[eleIdx];
    int m1 = e.mother1();
    int m2 = e.mother2();

    if (m1 <= 0 && m2 <= 0) {
        return false; // 没有母粒子
    }

    // 简单选一个母粒子（通常只有一个）
    int mom = (m2 > 0 && m2 != m1) ? m2 : m1;
    if (mom <= 0 || mom >= ev.size()) return false;

    const Particle& h = ev[mom];
    int idMom = h.id();

    if (isCharmMeson(idMom)) {
        flavor = 1;
    } else if (isBottomMeson(idMom)) {
        flavor = 2;
    } else {
        return false; // 母粒子不是 D/B
    }

    hadronPdg = idMom;
    hadronPt  = h.pT();

    // 检查 mom 的直接子代里：包含这条 e 且存在 ν_e/ν̄_e
    int d1 = h.daughter1();
    int d2 = h.daughter2();
    if (d1 <= 0 || d2 <= 0 || d2 < d1) {
        return false;
    }

    bool hasThisEle = false;
    bool hasNeu     = false;

    for (int d = d1; d <= d2; ++d) {
        if (d < 0 || d >= ev.size()) continue;
        const Particle& ch = ev[d];
        int id   = ch.id();
        int apid = std::abs(id);

        if (d == eleIdx && apid == 11) {
            hasThisEle = true;
        }
        if (apid == 12) {
            hasNeu = true;
        }
    }

    if (hasThisEle && hasNeu) {
        return true;
    } else {
        flavor    = 0;
        hadronPdg = 0;
        hadronPt  = 0.0f;
        return false;
    }
}

// 把 角度 归一化到 [-π, π]
double deltaPhi(double phi1, double phi2)
{
    double dphi = phi1 - phi2;
    while (dphi >  M_PI) dphi -= 2.0 * M_PI;
    while (dphi < -M_PI) dphi += 2.0 * M_PI;
    return dphi;
}

// ===================== main =====================
int main(int argc, char* argv[])
{
    int nEvent = 100000;
    std::string card = "ppHF.cmnd";
    std::string outName = "ppHF_eXDecay_test.root";

    if (argc > 1) nEvent = std::atoi(argv[1]);
    if (argc > 2) card   = argv[2];
    if (argc > 3) outName = argv[3]; 
    if (argc > 4) seed    = std::atoi(argv[4]);

    std::string seedStr = "Random:seed = " + std::to_string(seed);

    Pythia pythia;
    pythia.readFile(card);
    pythia.readString("Random:setSeed = on");
    pythia.readString(seedStr);
    pythia.init();

    // --- ROOT 输出 ---
    outDir = "/sphenix/user/jzhang1/HFsemiClassifier/HF_PY/Generate/DataSet/";
    outNameFile = outDir + outName;
    TFile* fout = new TFile(outNameFile.c_str(), "RECREATE");
    TTree* t    = new TTree("tree", "HF semi-leptonic electrons + away-side hadrons (event-wise)");

    // ========== TTree 变量（event-wise + vectors） ==========

    // event-level
    int nEle;        // 本 event 中满足条件的电子数（semi-leptonic DB->e nu X）
    int nHad_away;   // 本 event 中所有 away-side hadron 的数目

    // 电子信息（长度 = nEle）
    std::vector<int>   ele_charge;
    std::vector<float> ele_E;
    std::vector<float> ele_pt;
    std::vector<float> ele_eta;
    std::vector<float> ele_phi;

    std::vector<int>   ele_hf_TAG;         // 0=none, 1=D, 2=B
    std::vector<bool>  ele_is_semileptonic;// 此代码中只填 semi-leptonic，但留这个 flag

    // 每个电子对应的 away-side multiplicity & sum pT
    std::vector<int>   ele_nCh_away;
    std::vector<float> ele_sumPt_away;

    // away-side hadron 信息
    // had_fromEle: 这个 hadron 是相对于第几个电子（0~nEle-1）的 away-side
    std::vector<int>   had_fromEle;
    std::vector<int>   had_charge;
    std::vector<float> had_pt;
    std::vector<float> had_eta;
    std::vector<float> had_phi;

    // ========== 建立分支 ==========
    t->Branch("nEle",      &nEle,      "nEle/I");
    t->Branch("nHad_away", &nHad_away, "nHad_away/I");

    t->Branch("ele_charge",         &ele_charge);
    t->Branch("ele_E",              &ele_E);
    t->Branch("ele_pt",             &ele_pt);
    t->Branch("ele_eta",            &ele_eta);
    t->Branch("ele_phi",            &ele_phi);
    t->Branch("ele_hf_TAG",         &ele_hf_TAG);
    t->Branch("ele_is_semileptonic",&ele_is_semileptonic);

    t->Branch("ele_nCh_away",       &ele_nCh_away);
    t->Branch("ele_sumPt_away",     &ele_sumPt_away);

    t->Branch("had_fromEle",        &had_fromEle);
    t->Branch("had_charge",         &had_charge);
    t->Branch("had_pt",             &had_pt);
    t->Branch("had_eta",            &had_eta);
    t->Branch("had_phi",            &had_phi);

    // acceptance & cut
    const double dphiWindow = M_PI; // 这里沿用你原来的设定；如要 |Δφ-π|<π/3 可以改成 M_PI/3
    const double etaMaxHad  = 1.0;
    const double etaMaxEle  = 1.0;
    const double ptMinEle   = 3.0;

    // ========== 事件循环 ==========
    for (int iEvent = 0; iEvent < nEvent; ++iEvent)
    {
        if (!pythia.next()) continue;
        const Event& ev = pythia.event;

        // 每个 event 开始先清空所有 vector
        ele_charge.clear();
        ele_E.clear();
        ele_pt.clear();
        ele_eta.clear();
        ele_phi.clear();
        ele_hf_TAG.clear();
        ele_is_semileptonic.clear();
        ele_nCh_away.clear();
        ele_sumPt_away.clear();

        had_fromEle.clear();
        had_charge.clear();
        had_pt.clear();
        had_eta.clear();
        had_phi.clear();

        // ---------- 遍历最终态电子 ----------
        for (int i = 0; i < ev.size(); ++i)
        {
            const Particle& p = ev[i];
            if (!p.isFinal()) continue;

            int id = p.id();
            if (id != 11 && id != -11) continue; // 只要 e-/e+

            double Energy = p.e();
            double pt  = p.pT();
            double eta = p.eta();
            double phi = p.phi();

            if (pt < ptMinEle)        continue;
            if (std::abs(eta) > etaMaxEle) continue;

            // 判断是否为 直接 D/B -> e ν X 的半轻子电子
            int   flavor   = 0;
            int   hPdg     = 0;
            float hPt      = 0.0f;
            bool  semi     = isDirectDBSemiLeptonicToElectron(ev, i, flavor, hPdg, hPt);

            // if (!semi) {
            //     // 如果你想保留所有电子，可以把这句注释掉，
            //     // 然后 push_back 时仍然记录 flavor 和 semi 标记
            //     continue;
            // }

            // 记录这个电子的信息
            int eleIndex = ele_pt.size(); // 新电子的 index（0 ~ nEle-1）

            ele_charge.push_back( (id > 0 ? +1 : -1) );
            ele_E     .push_back( Energy );
            ele_pt    .push_back( pt );
            ele_eta   .push_back( eta );
            ele_phi   .push_back( phi );

            ele_hf_TAG        .push_back( flavor ); // 1(D), 2(B)
            ele_is_semileptonic.push_back( true );

            // 统计这个电子对应的 away-side hadron
            int   multAllH  = 0;
            float sumPtAllH = 0.0f;

            for (int j = 0; j < ev.size(); ++j)
            {
                if (j == i) continue; // 不数这条电子本身
                const Particle& h = ev[j];
                if (!h.isFinal())  continue;
                if (!h.isCharged()) continue;
                if (h.isLepton())   continue; // 只数 hadron

                double etaH = h.eta();
                if (std::abs(etaH) > etaMaxHad) continue;

                double phiH = h.phi();
                double dphi = deltaPhi(phiH, phi);
                double dphiToPi = std::abs(std::abs(dphi) - M_PI);

                if (dphiToPi < dphiWindow)
                {
                    multAllH++;
                    sumPtAllH += h.pT();

                    // 记录这个 hadron 的信息（相对 eleIndex 是 away-side）
                    had_fromEle.push_back( eleIndex );
                    had_charge .push_back( h.charge() );
                    had_pt     .push_back( h.pT() );
                    had_eta    .push_back( h.eta() - eta ); // 相对电子的 Δη
                    had_phi    .push_back( dphi ); // 相对电子的 Δφ
                }
            }

            ele_nCh_away  .push_back( multAllH );
            ele_sumPt_away.push_back( sumPtAllH );
        }

        nEle      = ele_pt.size();
        nHad_away = had_pt.size();

        // 只写入至少有一个满足条件电子的 event
        if (nEle > 0) {
            t->Fill();
        }
    }

    pythia.stat();

    fout->cd();
    t->Write();
    fout->Close();

    std::cout << "Finished. Wrote file ppHF_eXDecay.root with tree 'tree' (event-wise vectors)." << std::endl;
    return 0;
}
