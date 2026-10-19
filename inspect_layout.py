import fitz

doc = fitz.open('paper/main.pdf')
for pno in [3, 4, 5]:
    p = doc[pno]
    print(f"\n================ PAGE {pno+1} ================")
    for b in p.get_text('blocks'):
        s = b[4].replace('\n', ' ')[:100]
        # remove non-ascii for console printing
        s_ascii = s.encode('ascii', errors='replace').decode('ascii')
        print(f"({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}): {s_ascii}")
