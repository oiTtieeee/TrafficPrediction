import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_fig1():
    # Set up figure with clean modern dimensions
    fig, ax = plt.subplots(figsize=(15.5, 4.2), dpi=300)
    ax.axis('off')

    # Coordinated modern academic color palette
    c_bg_group = "#F8FAFC"       # Slate 50
    c_border_group = "#CBD5E1"   # Slate 300
    c_header = "#0F172A"         # Slate 900
    
    c_box_bg = "#FFFFFF"         # Pure white box
    c_box_border = "#334155"     # Slate 700
    c_box_title = "#0F172A"      # Slate 900
    c_box_sub = "#475569"        # Slate 600
    
    # Highlight colors
    c_accent_blue = "#0284C7"    # Sky 600
    c_blue_bg = "#F0F9FF"        # Sky 50
    c_blue_border = "#0284C7"
    
    c_accent_green = "#059669"   # Emerald 600
    c_green_bg = "#ECFDF5"       # Emerald 50
    c_green_border = "#059669"
    
    c_arrow = "#1E293B"          # Slate 800
    c_arrow_skip = "#2563EB"     # Royal blue for skip connection
    
    # Coordinates setup
    # 4 distinct columns (stages)
    # Stage 1: x in [0.015, 0.235]
    # Stage 2: x in [0.255, 0.475]
    # Stage 3: x in [0.495, 0.725]
    # Stage 4: x in [0.745, 0.985]
    
    stages = [
        (0.015, 0.225, "1. Ingestion & Road Geometry", "#F1F5F9"),
        (0.255, 0.225, "2. Preprocessing & Graph Laplacian", "#F1F5F9"),
        (0.495, 0.235, "3. Localized Spatiotemporal GCN", "#F1F5F9"),
        (0.745, 0.240, "4. Momentum Reconstruction & Output", "#EFF6FF")
    ]
    
    group_y = 0.04
    group_h = 0.88
    
    # Draw Stage Containers
    for sx, sw, stitle, sbg in stages:
        rect = patches.FancyBboxPatch(
            (sx, group_y), sw, group_h,
            boxstyle="round,pad=0.008,rounding_size=0.018",
            facecolor=sbg, edgecolor=c_border_group, lw=1.2, linestyle="--", zorder=1
        )
        ax.add_patch(rect)
        ax.text(sx + sw/2, group_y + group_h - 0.045, stitle,
                ha='center', va='center', weight='bold', fontsize=11.5, color=c_header, zorder=2)

    # Box drawing helper
    def draw_node(x, y, w, h, title, lines=None, bg=c_box_bg, border=c_box_border, lw=1.4, title_color=c_box_title):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.008,rounding_size=0.014",
            facecolor=bg, edgecolor=border, lw=lw, zorder=3
        )
        ax.add_patch(box)
        
        # Title
        if lines:
            ax.text(x + w/2, y + h - 0.048, title, ha='center', va='center',
                    weight='bold', fontsize=11.0, color=title_color, zorder=4)
            # Lines
            n_lines = len(lines)
            start_y = y + h - 0.095
            line_spacing = (h - 0.11) / max(n_lines, 1)
            for idx, line in enumerate(lines):
                ly = start_y - idx * line_spacing
                ax.text(x + w/2, ly, line, ha='center', va='center',
                        fontsize=9.2, color=c_box_sub, weight='medium', zorder=4)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                    weight='bold', fontsize=11.0, color=title_color, zorder=4)

    # Stage 1 Nodes
    draw_node(0.030, 0.50, 0.195, 0.32, "Freeway Speed Telemetry",
              [r"Loop Detectors ($N=207$ / $325$)",
               r"Lookback Horizon: $T_{\mathrm{in}} = 12$ steps",
               r"Raw Sequence $X_{\mathrm{raw}} \in \mathbb{R}^{B \times N \times 12}$"])
    
    draw_node(0.030, 0.08, 0.195, 0.32, "Physical Highway Topology",
              [r"Road Distance Matrix $\mathrm{dist}(u, v)$",
               r"Shortest Path Travel Distance (km)",
               r"True Sensor Geo-Coordinates"])

    # Stage 2 Nodes
    draw_node(0.270, 0.50, 0.195, 0.32, "Leak-Free Normalizer",
              [r"Train-split only: $\mu, \sigma$",
               r"Temporal Imputation (Forward-Fill)",
               r"Standardized Input $X \in \mathbb{R}^{B \times N \times 12}$"])
    
    draw_node(0.270, 0.08, 0.195, 0.32, "Thresholded Road Graph",
              [r"Gaussian Kernel: $W_{ij} = \exp(-d^2/\sigma^2)$",
               r"Self-Loops: $\tilde{A} = W + I_N$",
               r"Normalized Laplacian: $\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$"])

    # Stage 3 Nodes
    draw_node(0.510, 0.50, 0.205, 0.32, "GCN Layer 1 + ReLU",
              [r"Temporal Projection: $W^{(0)} \in \mathbb{R}^{12 \times 64}$",
               r"Spatial Conv: $H^{(1)} = \mathrm{ReLU}(\tilde{A}_{\mathrm{norm}} X W^{(0)})$",
               r"Latent Graph State: $H^{(1)} \in \mathbb{R}^{B \times N \times 64}$"])
    
    draw_node(0.510, 0.08, 0.205, 0.32, "GCN Layer 2 (Residual)",
              [r"Projection: $W^{(1)} \in \mathbb{R}^{64 \times 1}$",
               r"$\Delta\hat{y} = \tilde{A}_{\mathrm{norm}} H^{(1)} W^{(1)}$",
               r"Residual Momentum: $\Delta\hat{y} \in \mathbb{R}^{B \times N \times 1}$"])

    # Stage 4 Nodes
    # 1. Summation Junction
    sum_cx, sum_cy = 0.865, 0.73
    r_sum = 0.032
    # Circle patch
    circ = patches.Circle((sum_cx, sum_cy), r_sum, facecolor="#F8FAFC", edgecolor=c_box_border, lw=1.8, zorder=5)
    ax.add_patch(circ)
    ax.text(sum_cx, sum_cy, "+", fontsize=18, weight='bold', ha='center', va='center', color=c_header, zorder=6)
    
    # De-normalization
    draw_node(0.760, 0.38, 0.210, 0.21, "Inverse Z-Score Mapping",
              [r"Denormalize: $\hat{y}_{\mathrm{phys}} = \hat{y}_{\mathrm{norm}} \cdot \sigma + \mu$",
               r"Restores Physical Metric Units (mph)"])
    
    # Final Output
    draw_node(0.760, 0.08, 0.210, 0.22, "Physical Traffic Forecast",
              [r"Predicted Network Speed $\hat{y}_{t+h} \in \mathbb{R}^N$",
               r"Evaluated at $h \in \{1, 3, 6, 12\}$ horizons",
               r"Robust Under Data Scarcity & Faults"],
              bg=c_blue_bg, border=c_blue_border, lw=2.0, title_color=c_accent_blue)

    # Connecting Arrows
    def draw_arrow(x1, y1, x2, y2, color=c_arrow, lw=1.5, zorder=5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=lw,
                                    headwidth=6.5, headlength=7.0, shrink=0.01), zorder=zorder)

    # 1 -> 2
    draw_arrow(0.225, 0.66, 0.270, 0.66)
    draw_arrow(0.225, 0.24, 0.270, 0.24)

    # 2 -> 3
    # X -> GCN 1
    draw_arrow(0.465, 0.66, 0.510, 0.66)
    ax.text(0.4875, 0.685, r"$X$", ha='center', va='bottom', fontsize=10.0, weight='bold', color=c_header)

    # Laplacian -> GCN 1 and GCN 2
    draw_arrow(0.465, 0.24, 0.510, 0.24)
    # Branch up to GCN 1
    ax.plot([0.488, 0.488, 0.510], [0.24, 0.55, 0.55], color=c_arrow, lw=1.5, zorder=4)
    ax.annotate('', xy=(0.510, 0.55), xytext=(0.495, 0.55),
                arrowprops=dict(facecolor=c_arrow, edgecolor=c_arrow, width=1.5,
                                headwidth=6.5, headlength=7.0), zorder=5)
    ax.text(0.480, 0.395, r"$\tilde{A}_{\mathrm{norm}}$", ha='right', va='center', fontsize=10.5, weight='bold', color=c_header)

    # GCN 1 -> GCN 2
    draw_arrow(0.6125, 0.50, 0.6125, 0.40)
    ax.text(0.622, 0.45, r"$H^{(1)}$", ha='left', va='center', fontsize=10.0, weight='bold', color=c_header)

    # GCN 2 -> Summation Junction (Residual Delta)
    ax.plot([0.715, 0.738, 0.738, sum_cx - r_sum], [0.24, 0.24, sum_cy, sum_cy], color=c_arrow, lw=1.5, zorder=4)
    ax.annotate('', xy=(sum_cx - r_sum, sum_cy), xytext=(sum_cx - r_sum - 0.015, sum_cy),
                arrowprops=dict(facecolor=c_arrow, edgecolor=c_arrow, width=1.5,
                                headwidth=6.5, headlength=7.0), zorder=5)
    ax.text(0.746, 0.48, r"Residual Velocity $\Delta\hat{y}$", ha='left', va='center',
            fontsize=9.5, weight='bold', color=c_header, rotation=-90)

    # Skip Connection: Current Speed x_t (Lookback Origin) -> Summation Junction
    # Routed cleanly over the top
    ax.plot([0.3675, 0.3675, sum_cx, sum_cx], [0.82, 0.88, 0.88, sum_cy + r_sum],
            color=c_arrow_skip, lw=1.8, linestyle="-", zorder=4)
    ax.annotate('', xy=(sum_cx, sum_cy + r_sum), xytext=(sum_cx, sum_cy + r_sum + 0.015),
                arrowprops=dict(facecolor=c_arrow_skip, edgecolor=c_arrow_skip, width=1.8,
                                headwidth=7.0, headlength=7.5), zorder=5)
    
    # Badge / pill for Current Speed x_t
    pill = patches.FancyBboxPatch(
        (0.550, 0.855), 0.190, 0.050,
        boxstyle="round,pad=0.005,rounding_size=0.010",
        facecolor="#EFF6FF", edgecolor=c_arrow_skip, lw=1.2, zorder=6
    )
    ax.add_patch(pill)
    ax.text(0.645, 0.880, r"Current Speed $x_t$ (Lookback Origin)", ha='center', va='center',
            fontsize=9.8, weight='bold', color=c_arrow_skip, zorder=7)

    # Summation Junction -> De-normalization
    draw_arrow(sum_cx, sum_cy - r_sum, sum_cx, 0.59)
    ax.text(sum_cx + 0.012, 0.655, r"$\hat{y}_{\mathrm{norm}} = x_t + \Delta\hat{y}$",
            ha='left', va='center', fontsize=9.5, weight='bold', color=c_header)

    # De-normalization -> Output
    draw_arrow(sum_cx, 0.38, sum_cx, 0.30)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.98)
    
    out_path = Path("paper/figures/fig1_system_architecture.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Successfully generated modern fig1_system_architecture.png")

if __name__ == "__main__":
    create_fig1()
