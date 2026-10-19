import fitz

doc = fitz.open('paper/test_main.pdf')
p4 = doc[3]
print("=== PAGE 4 BLOCKS ===")
for b in p4.get_text('blocks'):
    s = b[4].strip().replace('\n', ' ')[:80]
    s_ascii = s.encode('ascii', errors='replace').decode('ascii')
    print(f"y0={b[1]:.1f}, y1={b[3]:.1f}, x0={b[0]:.1f}, x1={b[2]:.1f}: {s_ascii}")
