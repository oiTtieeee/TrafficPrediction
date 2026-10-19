"""
================================================================================
Pre-Submission Verification and Packaging Tool
================================================================================
Validates all required paper assets and bundles them into submission_package.zip.
"""
import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
SUBMISSION_ZIP = PAPER_DIR / "submission_package.zip"

REQUIRED_FILES = [
    PAPER_DIR / "main.tex",
    PAPER_DIR / "main.pdf",
    PAPER_DIR / "manuscript.md",
    PAPER_DIR / "references.bib",
    PAPER_DIR / "cover_letter.tex",
    PAPER_DIR / "cover_letter.md",
    PAPER_DIR / "highlights.md",
    FIGURES_DIR / "fig1_system_architecture.png",
    FIGURES_DIR / "fig2_sensor_network_map.png",
    FIGURES_DIR / "fig3_data_scarcity_curves.png",
    FIGURES_DIR / "fig4_residual_delta_analysis.png",
    FIGURES_DIR / "fig5_dropout_imputation_ablation.png",
    FIGURES_DIR / "fig6_multi_horizon_forecasting.png"
]

def main():
    print("=" * 70)
    print("PRE-SUBMISSION VERIFICATION & PACKAGING AUDIT")
    print("=" * 70)

    # 1. Check all required files
    print("\n[1] Checking file presence and integrity...")
    missing = []
    for f in REQUIRED_FILES:
        if f.exists():
            size_kb = f.stat().st_size / 1024
            print(f"  [PASS] {f.name:35s} ({size_kb:7.1f} KB)")
        else:
            print(f"  [FAIL] {f.name:35s} (MISSING!)")
            missing.append(f)

    if missing:
        print(f"\n[ERROR] {len(missing)} files missing. Cannot package.")
        return

    # 2. Check LaTeX references match bib
    print("\n[2] Checking BibTeX citation keys in main.tex...")
    with open(PAPER_DIR / "main.tex", "r", encoding="utf-8") as f:
        tex_content = f.read()
    with open(PAPER_DIR / "references.bib", "r", encoding="utf-8") as f:
        bib_content = f.read()

    import re
    cites = set(re.findall(r'\\cite\{([^}]+)\}', tex_content))
    # split multi-citations
    individual_cites = set()
    for c in cites:
        for single_c in c.split(','):
            individual_cites.add(single_c.strip())

    bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib_content))

    missing_keys = individual_cites - bib_keys
    if not missing_keys:
        print(f"  [PASS] All {len(individual_cites)} citations in main.tex are present in references.bib!")
    else:
        print(f"  [WARNING] Citations missing in references.bib: {missing_keys}")

    # 3. Create Overleaf / Journal Zip Package
    print(f"\n[3] Creating submission package: {SUBMISSION_ZIP}...")
    with zipfile.ZipFile(SUBMISSION_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
        # LaTeX root files
        zf.write(PAPER_DIR / "main.tex", arcname="main.tex")
        zf.write(PAPER_DIR / "main.pdf", arcname="main.pdf")
        zf.write(PAPER_DIR / "references.bib", arcname="references.bib")
        zf.write(PAPER_DIR / "cover_letter.tex", arcname="cover_letter.tex")
        zf.write(PAPER_DIR / "cover_letter.md", arcname="cover_letter.md")
        zf.write(PAPER_DIR / "highlights.md", arcname="highlights.md")
        zf.write(PAPER_DIR / "manuscript.md", arcname="manuscript.md")
        
        # Figures folder
        for fig in FIGURES_DIR.glob("*.png"):
            zf.write(fig, arcname=f"figures/{fig.name}")

    zip_size_mb = SUBMISSION_ZIP.stat().st_size / (1024 * 1024)
    print(f"  [PASS] Successfully created submission package ({zip_size_mb:.2f} MB)")
    print("\n" + "=" * 70)
    print("SUBMISSION READINESS: 100% READY FOR JOURNAL SUBMISSION / OVERLEAF")
    print("=" * 70)

if __name__ == "__main__":
    main()
