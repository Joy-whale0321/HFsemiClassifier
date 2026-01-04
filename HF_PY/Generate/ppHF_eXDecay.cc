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
#include <deque>
#include <algorithm>

using namespace Pythia8;

// ===================== tool function =====================

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

// charm/bottom baryon 也算进来：charm baryon: 4000~4999, bottom baryon: 5000~5999
bool isCharmHadron(int id)
{
    int pdg = std::abs(id);
    return isCharmMeson(pdg) || (pdg >= 4000 && pdg < 5000);
}

bool isBottomHadron(int id)
{
    int pdg = std::abs(id);
    return isBottomMeson(pdg) || (pdg >= 5000 && pdg < 6000);
}

// ---------------------------------------------------------
// strictVertex
// 严格顶点验证：候选 hadron 的 direct daughters 中必须同时包含 e(index) 和 νe(±12)
// 输入：Pythia8::Event ev, hadron 在 ev 中的 index, electron 在 ev 中的 index
// 输出：bool
// ---------------------------------------------------------
bool strictVertex(const Event& ev, int hadIdx, int eleIndex)
{
    if (hadIdx <= 0 || hadIdx >= ev.size()) return false;
    if (eleIndex <= 0 || eleIndex >= ev.size()) return false;

    const Particle& H = ev[hadIdx];

    int d1 = H.daughter1();
    int d2 = H.daughter2();
    if (d1 <= 0 || d2 <= 0) return false;

    bool hasThisE = false;
    bool hasNuE   = false;

    // Pythia8 daughters 通常是连续区间 [d1, d2]
    for (int k = d1; k <= d2; ++k)
    {
        if (k == eleIndex) hasThisE = true;
        int ab = std::abs(ev[k].id());
        if (ab == 12) hasNuE = true; // νe or anti-νe
        if (hasThisE && hasNuE) return true;
    }
    return false;
}

// ---------------------------------------------------------
// hasBottomAncestor
// 用 BFS（限深度）判断某个粒子（这里用于 D meson）是否存在 bottom meson 祖先
// 输入：Pythia8::Event ev, startIdx(从该粒子开始向上找), maxDepth(最大祖先代数)
// 输出：bool，是否存在 B meson 祖先
// 注意：这一步不需要 strict 顶点验证，只要祖先里出现 B meson 即可
// ---------------------------------------------------------
bool hasBottomAncestor(const Event& ev, int startIdx, int maxDepth = 10)
{
    if (startIdx <= 0 || startIdx >= ev.size()) return false;

    std::vector<char> visited(ev.size(), 0);
    std::vector<int> current;
    current.push_back(startIdx);
    visited[startIdx] = 1;

    for (int depth = 1; depth <= maxDepth; ++depth)
    {
        std::vector<int> next;
        next.reserve(current.size() * 2);

        // 生成下一代祖先集合
        for (int idx : current)
        {
            const Particle& p = ev[idx];
            int m1 = p.mother1();
            int m2 = p.mother2();

            auto pushMother = [&](int m)
            {
                if (m <= 0 || m >= ev.size()) return;
                if (visited[m]) return;
                visited[m] = 1;
                next.push_back(m);
            };

            pushMother(m1);
            if (m2 != m1) pushMother(m2);
        }

        if (next.empty()) break;

        // 这一代里只要出现 B meson 祖先就返回 true
        for (int m : next)
        {
            if (isBottomMeson(ev[m].id())) return true;
        }

        current.swap(next);
    }

    return false;
}

// ---------------------------------------------------------
// tagHFSemiLeptonicElectron
// 目标：在“限深度向上追溯 N 代”内，找到一个 HF *meson* (D/B)，并且满足严格顶点：
//      该 meson 的 direct daughters 里同时包含：
//        - 这条电子（用 index 精确匹配）
//        - 一个 (anti)νe（PDG ±12）
// 返回值：bool，是否为半轻衰变电子（strict）
//
// 输出：flavorTag（注意：这里我们把标签升级成 3 类）
//   0 = none
//   1 = prompt D -> e ν X
//   2 = non-prompt D (B -> D -> e ν X)
//   3 = B -> e ν X
//
// 额外输出：parent hadron PDG, parent hadron pT（“直接衰变出这条 e 的那个 hadron”）
// ---------------------------------------------------------
bool tagHFSemiLeptonicElectron(const Event& ev,
                               int eleIndex,
                               int& flavorTag,
                               int& hPdg,
                               float& hPt,
                               int N  = 5,   // 最大向上追溯代数（找直接产生 e 的 D/B）
                               int NB = 10)  // 判定 non-prompt D 时向上找 B 的深度
{
    // 初始化输出
    flavorTag = 0;
    hPdg      = 0;
    hPt       = 0.0f;

    // electron 验证
    if (eleIndex <= 0 || eleIndex >= ev.size()) return false;
    const Particle& ele = ev[eleIndex];
    if (!ele.isFinal()) return false;
    if (std::abs(ele.id()) != 11) return false;

    // 按“代数”逐层向上搜索（BFS层序 - Breadth First Search 广度优先），限制最多 N 代
    std::vector<char> visited(ev.size(), 0); // inti-0, 记录在搜索中已经看过哪些粒子
    std::vector<int> current; //current 用来存 当前这一层（current layer） 的所有粒子 index
    current.push_back(eleIndex); // 0-layer：{ e } 1-layer：{ e 的 mother } 2-layer：{ mothers's mother }...
    visited[eleIndex] = 1; // 标记这条电子 已经被访问过 depth=0

    for (int depth = 1; depth <= N; ++depth)
    {
        std::vector<int> next;  // next 用来存 下一层（next layer） 的所有粒子 index
        next.reserve(current.size() * 2); // 每个粒子最多 mother1/mother2

        // 生成下一代祖先集合
        for (int idx : current)
        {
            const Particle& p = ev[idx]; // 当前粒子 0 depth layer is e, then its mother, etc.
            int m1 = p.mother1();
            int m2 = p.mother2();

            auto pushMother = [&](int m)
            {
                if (m <= 0 || m >= ev.size()) return; // out range of the event
                if (visited[m]) return; // check whether be used
                visited[m] = 1;
                next.push_back(m);
            };

            pushMother(m1);
            if (m2 != m1) pushMother(m2); // if there is mother2 and not-equit with mother1 also need to be consider
        }

        if (next.empty()) break;

        // 在这一代里找“第一个通过严格顶点验证”的 HF meson
        for (int m : next)
        {
            int mid = ev[m].id(); // mother particle PDG id

            // 先不保留 baryon，只认 meson
            if (isBottomMeson(mid))
            {
                if (strictVertex(ev, m, eleIndex))
                {
                    // B -> e ν X
                    flavorTag = 3;
                    hPdg      = mid;
                    hPt       = ev[m].pT();
                    return true;
                }
            }
            else if (isCharmMeson(mid))
            {
                if (strictVertex(ev, m, eleIndex))
                {
                    // D -> e ν X
                    // 再判定：这个 D 是否来自 B（non-prompt D: B -> D -> e）
                    bool fromB = hasBottomAncestor(ev, m, NB);
                    flavorTag  = fromB ? 2 : 1;

                    hPdg = mid;
                    hPt  = ev[m].pT();
                    return true;
                }
            }
        }

        // 继续向上
        current.swap(next); // 交换 current 和 next 两个 vector 的内部指针 - 互换内容
    }

    // 没找到符合 strict 顶点的 HF meson
    flavorTag = 0;
    hPdg      = 0;
    hPt       = 0.0f;
    return false;
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
    int nEvent = 1000000;
    std::string card = "ppHF.cmnd";
    std::string outName = "ppHF_eXDecay_test.root";
    int seed = 12345;

    if (argc > 1) nEvent = std::atoi(argv[1]);   // event number
    if (argc > 2) card   = argv[2];              // pythia config file
    if (argc > 3) outName = argv[3];             // output root file name
    if (argc > 4) seed    = std::atoi(argv[4]);  // random seed

    // Pythia initialization
    std::string seedStr = "Random:seed = " + std::to_string(seed);

    Pythia pythia;
    pythia.readFile(card);
    pythia.readString("Random:setSeed = on");
    pythia.readString(seedStr);
    pythia.init();

    // --- ROOT output setting ---
    // std::string outDir = "/sphenix/user/jzhang1/HFsemiClassifier/HF_PY/Generate/DataSet/";
    std::string outDir = "./";
    std::string outNameFile = outDir + outName;
    TFile* fout = new TFile(outNameFile.c_str(), "RECREATE");
    TTree* t    = new TTree("tree", "HF semi-leptonic electrons + away-side hadrons (event-wise)");

    // ========== TTree 变量（event-wise + vectors） ==========

    // event-level
    int nEle;        // 本 event 中满足条件的电子数(D/B semi-leptonic)
    int nHad_away;   // 本 event 中所有 away-side hadron 的数目

    // 1 vector per electron info
    // 电子信息（长度 = nEle）
    std::vector<int>   ele_charge;
    std::vector<float> ele_E;
    std::vector<float> ele_pt;
    std::vector<float> ele_eta;
    std::vector<float> ele_phi;

    std::vector<int>   ele_hf_TAG;          // 0=none, 1=prompt D, 2=non-prompt D (B->D->e), 3=B
    std::vector<bool>  ele_is_semileptonic; // 此代码中只填 semi-leptonic，但留这个 flag

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
    const double dphiWindow = M_PI; // Δφ between hadron and electron window
    const double etaMaxHad  = 1.0;  // hadron acceptance |η| < 1.0
    const double etaMaxEle  = 1.0;  // electron acceptance |η| < 1.0
    const double ptMinEle   = 3.0;  // electron minimum pT > 3 GeV/c

    // ========== event loop ==========
    for (int iEvent = 0; iEvent < nEvent; ++iEvent)
    {
        if (!pythia.next()) continue; // pythia.next() 按照init生成event，如果失败会返回false跳过
        const Event& ev = pythia.event; // Pythia8::Event，一个粒子列表

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

        // ---------- 遍历粒子 找出半轻衰变的电子 ----------
        for (int i = 0; i < ev.size(); ++i)
        {
            const Particle& p = ev[i];
            if (!p.isFinal()) continue; // 只要最终态粒子

            int id = p.id();
            if (id != 11 && id != -11) continue; // 只要 e-/e+

            // electron info
            double charge = p.charge();
            double Energy = p.e();
            double pt  = p.pT();
            double eta = p.eta();
            double phi = p.phi();

            if (pt < ptMinEle)        continue;
            if (std::abs(eta) > etaMaxEle) continue;

            // 判断是否为 D/B -> e ν X 半轻衰变产生的电子
            int   flavor   = 0;
            int   hPdg     = 0;
            float hPt      = 0.0f;
            bool  semi     = tagHFSemiLeptonicElectron(ev, i, flavor, hPdg, hPt);

            // if (!semi) {
            //     // 如果你想保留所有电子，可以把这句注释掉，
            //     // 然后 push_back 时仍然记录 flavor 和 semi 标记
            //     continue;
            // }

            // 记录这个电子的信息
            int eleIndex = ele_pt.size(); // 电子的 index，size随着e的一次push back + 1

            ele_charge.push_back( charge );
            ele_E     .push_back( Energy );
            ele_pt    .push_back( pt );
            ele_eta   .push_back( eta );
            ele_phi   .push_back( phi );

            ele_hf_TAG          .push_back( flavor ); // [UPDATE] 0/1/2/3: none/promptD/nonpromptD/B
            ele_is_semileptonic .push_back( true );   // 此代码中只填 semi-leptonic，但留这个 flag

            // 统计这个电子相关的hadron信息
            int   multAllH  = 0;
            float sumPtAllH = 0.0f;

            for (int j = 0; j < ev.size(); ++j)
            {
                if (j == i) continue; // 不数这条电子本身
                const Particle& h = ev[j];
                if (!h.isFinal())  continue;  // 只要最终态粒子
                if (!h.isCharged()) continue; // 只要带电粒子
                if (h.isLepton())   continue; // 不要lepton

                // eta cut for hadrons
                double etaH = h.eta();
                if (std::abs(etaH) > etaMaxHad) continue;

                double phiH = h.phi();
                double dphi = deltaPhi(phiH, phi); // hadron φ - electron φ
                double dphiToPi = std::abs(std::abs(dphi) - M_PI);

                if (dphiToPi < dphiWindow)
                {
                    multAllH++; // multi
                    sumPtAllH += h.pT(); // all hadron pt to eval hard scattering pT

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
