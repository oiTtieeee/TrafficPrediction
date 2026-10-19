import fitz

doc = fitz.open('paper/test_main.pdf')
for pno in range(len(doc)):
    p = doc[pno]
    print(f"\n================ PAGE {pno+1} ================")
    blocks = p.get_text('blocks')
    for b in blocks:
        s = b[4].strip().replace('\n', ' ')[:75]
        s_ascii = s.encode('ascii', errors='replace').decode('ascii')
        print(f"  y0={b[1]:.1f}, y1={b[3]:.1f}, x0={b[0]:.1f}, x1={b[2]:.1f}: {s_ascii}")
