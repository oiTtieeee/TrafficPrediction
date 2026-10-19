import fitz

doc = fitz.open('paper/main.pdf')
p6 = doc[5]
print("--- PAGE 6 FULL TEXT WITH RECTS ---")
for b in p6.get_text('blocks'):
    s = b[4].strip().replace('\n', ' ')
    s_ascii = s[:90].encode('ascii', errors='replace').decode('ascii')
    print(f"y0={b[1]:.1f}, y1={b[3]:.1f}, x0={b[0]:.1f}, x1={b[2]:.1f}: {s_ascii}")
