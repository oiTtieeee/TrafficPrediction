import fitz

doc = fitz.open('paper/main.pdf')
for pno in range(len(doc)):
    p = doc[pno]
    print(f"\n================ PAGE {pno+1} (height={p.rect.height}, width={p.rect.width}) ================")
    col1 = []
    col2 = []
    for b in p.get_text('blocks'):
        # b is (x0, y0, x1, y1, text, block_no, block_type)
        if b[0] < 300:
            col1.append(b)
        else:
            col2.append(b)
    
    print(f"  Col 1: {len(col1)} blocks. Top={min([b[1] for b in col1], default=0):.1f}, Bottom={max([b[3] for b in col1], default=0):.1f}")
    print(f"  Col 2: {len(col2)} blocks. Top={min([b[1] for b in col2], default=0):.1f}, Bottom={max([b[3] for b in col2], default=0):.1f}")
