import subprocess
import fitz

with open('paper/test_b2.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the minor overfull hbox in Fig 1 caption
content = content.replace(
    'processed by a 2-layer localized graph spectral convolution network, and reconstructed into physical speed forecasts.',
    'processed by a 2-layer localized spectral GCN, and reconstructed into physical speed forecasts.'
)

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated paper/main.tex. Compiling with pdflatex and bibtex...")
res1 = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'], cwd='paper', capture_output=True, text=True)
res_bib = subprocess.run(['bibtex', 'main'], cwd='paper', capture_output=True, text=True)
res2 = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'], cwd='paper', capture_output=True, text=True)
res3 = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'], cwd='paper', capture_output=True, text=True)

doc = fitz.open('paper/main.pdf')
print(f"=== Compilation Result ===")
print(f"Total Pages: {len(doc)}")
for pno in range(len(doc)):
    p = doc[pno]
    txt = p.get_text('blocks')
    c1 = [b for b in txt if b[0] < 300]
    c2 = [b for b in txt if b[0] >= 300]
    b1 = max([b[3] for b in c1], default=0)
    b2 = max([b[3] for b in c2], default=0)
    print(f"  Page {pno+1}: Col1 bottom = {b1:.1f}, Col2 bottom = {b2:.1f}")

# Render all pages for visual confirmation
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    pix.save(f'paper/page_{i+1}_final.png')

print("Rendered all final pages to PNG.")
