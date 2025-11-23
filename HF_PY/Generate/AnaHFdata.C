#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TCanvas.h>

#include <vector>
#include <iostream>

void AnaHFdata(const char* infile  = "ppHF_eXDecay.root",
               const char* treename = "tree")
{
    // 打开文件
    TFile* f = TFile::Open(infile, "READ");
    if (!f || f->IsZombie())
    {
        std::cerr << "Error: cannot open file " << infile << std::endl;
        return;
    }

    // 获取 TTree
    TTree* PYtree = dynamic_cast<TTree*>(f->Get(treename));
    if (!PYtree)
    {
        std::cerr << "Error: cannot find TTree " << treename << " in file " << infile << std::endl;
        f->Close();
        return;
    }

    // 定义指针接 branch（都是 vector<float>）
    std::vector<float>* had_phi = nullptr;
    std::vector<float>* had_eta = nullptr;

    // 关联 branch
    PYtree->SetBranchAddress("had_phi", &had_phi);
    PYtree->SetBranchAddress("had_eta", &had_eta);

    // 定义一个 TH2D（根据你的物理量范围改 binning）
    int    nbinsX = 40;
    double xMin   = 0.0;
    double xMax   = 4.0;

    int    nbinsY = 30;
    double yMin   = -3.0;
    double yMax   = 3.0;

    TH2D* h2 = new TH2D("h2", "TH2 from vector<float> branches;X;Y",
                        nbinsX, xMin, xMax,
                        nbinsY, yMin, yMax);

    Long64_t nEntries = PYtree->GetEntries();
    std::cout << "Total entries: " << nEntries << std::endl;

    // loop over events
    for (Long64_t ievt = 0; ievt < nEntries; ++ievt)
    {
        PYtree->GetEntry(ievt);

        if (!had_phi || !had_eta) continue;

        // 防止长度不一样，取较小的长度
        size_t n = std::min(had_phi->size(), had_eta->size());

        for (size_t i = 0; i < n; ++i)
        {
            double x = std::abs(had_phi->at(i));
            double y = had_eta->at(i);
            h2->Fill(x, y);
        }
    }

    // 也可以把结果写到新文件
    TFile* fout = new TFile("PYHF_ana.root", "RECREATE");
    h2->Write();
    fout->Close();

    // 关闭输入文件
    f->Close();

    std::cout << "Done. Histogram 'h2' is in memory." << std::endl;
}
