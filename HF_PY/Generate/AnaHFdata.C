#include <TFile.h>
#include <TTree.h>
#include <TH2D.h>
#include <TCanvas.h>

#include <vector>
#include <iostream>

void AnaHFdata(
    const char* infile  = "/mnt/e/sphenix/HFsemiClassifier/HF_PY/Generate/DataSet/ppHF_eXDecay_p5B_1.root",
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
    std::vector<float>* ele_hf_TAG = nullptr;
    std::vector<float>* ele_nCh_away = nullptr;
    std::vector<float>* ele_pt = nullptr;

    std::vector<float>* had_pt = nullptr;
    std::vector<float>* had_phi = nullptr;
    std::vector<float>* had_eta = nullptr;

    // 关联 branch
    PYtree->SetBranchAddress("ele_hf_TAG", &ele_hf_TAG);
    PYtree->SetBranchAddress("ele_nCh_away", &ele_nCh_away);
    PYtree->SetBranchAddress("ele_pt", &ele_pt);

    PYtree->SetBranchAddress("had_pt", &had_pt);
    PYtree->SetBranchAddress("had_phi", &had_phi);
    PYtree->SetBranchAddress("had_eta", &had_eta);


    // 定义一个 TH（根据你的物理量范围改 binning）
    TH1D* h1_D_multi = new TH1D("h1_D_multi", "", 100, 0, 100);
    TH1D* h1_B_multi = new TH1D("h1_B_multi", "", 100, 0, 100);

    TH1D* h1_D_pt = new TH1D("h1_D_pt", "", 100, 0, 4);
    TH1D* h1_B_pt = new TH1D("h1_B_pt", "", 100, 0, 4);

    TH1D* h1_De_pt = new TH1D("h1_De_pt", "", 100, 0, 10);
    TH1D* h1_Be_pt = new TH1D("h1_Be_pt", "", 100, 0, 10);

    int    nbinsX = 80;
    double xMin   = -4.0;
    double xMax   = 4.0;

    int    nbinsY = 60;
    double yMin   = 0.0;
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
            double x = had_phi->at(i);
            double y = std::abs(had_eta->at(i));
            h2->Fill(x, y);
        }

        int size_e = ele_hf_TAG->size();
        {
            if (size_e > 1) cout<<"Event "<<ievt<<" has "<<size_e<<" electrons."<<endl;  
            // continue;
        }

        for(int i=0; i<1; ++i)
        {
            int tag = ele_hf_TAG->at(i);
            int multi = ele_nCh_away->at(i);
            double ele_pt_val = ele_pt->at(i);

            if (tag==1 && multi>0)
            {
                h1_D_multi->Fill(multi);
                h1_De_pt->Fill(ele_pt_val);

                int had_n = had_pt->size();
                for(size_t j = 0; j < had_n; ++j)
                {
                    double had_pt_val = had_pt->at(j);
                    h1_D_pt->Fill(had_pt_val);
                }
            }
            else if(tag==2 && multi>0)
            {
                h1_B_multi->Fill(multi);
                h1_Be_pt->Fill(ele_pt_val);

                int had_n = had_pt->size();
                for(size_t j = 0; j < had_n; ++j)
                {
                    double had_pt_val = had_pt->at(j);
                    h1_B_pt->Fill(had_pt_val);
                }
            }
        }

    }

    // Fit to get weights
    // 对 D-electron pt 拟合
    TF1* fDe = new TF1("fDe", "expo", 3, 6);  // expo: exp(p0 + p1*x)
    h1_De_pt->Fit(fDe, "R");                   // R: 只在 [0,10] 内拟合

    // 对 B-electron pt 拟合
    TF1* fBe = new TF1("fBe", "expo", 3, 6);
    h1_Be_pt->Fit(fBe, "R");

    // 拿到参数
    double ADe = fDe->GetParameter(0); // ln 部分的常数项
    double BDe = fDe->GetParameter(1); // ln 部分的斜率
    double ABe = fBe->GetParameter(0);
    double BBe = fBe->GetParameter(1);

    std::cout << "D: ln p ~ " << ADe << " + " << BDe << " * pt" << std::endl;
    std::cout << "B: ln p ~ " << ABe << " + " << BBe << " * pt" << std::endl;

    // poly func
    // double fitMin = 3.0;  // 你自己看分布，从这个pt开始拟合
    // double fitMax = 8.0;  // 比如6 GeV 以内统计多、比较好拟

    // // 三阶多项式：a0 + a1*x + a2*x^2 + a3*x^3
    // TF1* fDe = new TF1("fDe", "pol5", fitMin, fitMax);
    // TF1* fBe = new TF1("fBe", "pol5", fitMin, fitMax);

    // h1_De_pt->Fit(fDe, "R");  // 只在[fitMin,fitMax]拟合
    // h1_Be_pt->Fit(fBe, "R");

    // double aD0 = fDe->GetParameter(0);
    // double aD1 = fDe->GetParameter(1);
    // double aD2 = fDe->GetParameter(2);
    // double aD3 = fDe->GetParameter(3);

    // double aB0 = fBe->GetParameter(0);
    // double aB1 = fBe->GetParameter(1);
    // double aB2 = fBe->GetParameter(2);
    // double aB3 = fBe->GetParameter(3);

    // std::cout << "D count(pt) ~ " 
    //           << aD0 << " + " << aD1 << " pt + " << aD2 << " pt^2 + " << aD3 << " pt^3\n";
    // std::cout << "B count(pt) ~ " 
    //           << aB0 << " + " << aB1 << " pt + " << aB2 << " pt^2 + " << aB3 << " pt^3\n";

    // 也可以把结果写到新文件
    TFile* fout = new TFile("PYHF_ana.root", "RECREATE");
    h2->Write();
    h1_D_multi->Write();
    h1_B_multi->Write();
    h1_D_pt->Write();
    h1_B_pt->Write();
    h1_De_pt->Write();
    h1_Be_pt->Write();
    fDe->Write();
    fBe->Write();
    fout->Close();

    // 关闭输入文件
    f->Close();

    std::cout << "Done. Histogram 'h2' is in memory." << std::endl;
}
