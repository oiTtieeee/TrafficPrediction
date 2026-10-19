import fitz

doc = fitz.open('paper/main.pdf')
for pno in range(len(doc)):
    p = doc[pno]
    print(f"\n================ PAGE {pno+1} ================")
    blocks = p.get_text('blocks')
    for b in blocks:
        lines = [l.strip() for l in b[4].split('\n') if l.strip()]
        if len(lines) > 1:
            first_line = lines[0][:60]
            last_line = lines[-1][:60]
            print(f"[{b[1]:.1f}-{b[3]:.1f}] ({len(lines)} lines): '{first_line}' ... '{last_line}'")
