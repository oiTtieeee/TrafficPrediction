import subprocess
import fitz

with open('paper/test_fill_p6.tex', 'r', encoding='utf-8') as f:
    t = f.read()

# Increase itemsep slightly from 1.8pt to 2.2pt
t = t.replace(r'\setlength{\itemsep}{1.8pt plus 0.4pt minus 0.2pt}', r'\setlength{\itemsep}{2.2pt plus 0.4pt minus 0.2pt}')

with open('paper/main.tex', 'w', encoding='utf-8') as f:
    f.write(t)

subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'], cwd='paper', capture_output=True)
subprocess.run(['bibtex', 'main'], cwd='paper', capture_output=True)
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'], cwd='paper', capture_output=True)
subprocess.run(['pdflatex', '-interaction=nonstopmode', 'main.tex'], cwd='paper', capture_output=True)

doc = fitz.open('paper/main.pdf')
print(f"Total Pages: {len(doc)}")
for pno in range(len(doc)):
    p = doc[pno]
    txt = p.get_text('blocks')
    c1 = [b for b in txt if b[0] < 300]
    c2 = [b for b in txt if b[0] >= 300]
    b1 = max([b[3] for b in c1], default=0)
    b2 = max([b[3] for b in c2], default=0)
    print(f"  Page {pno+1}: Col1 bottom = {b1:.1f}, Col2 bottom = {b2:.1f}")

for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=150)
    pix.save(f'paper/page_{i+1}.png')
print("Rendered all pages to PNG.")
