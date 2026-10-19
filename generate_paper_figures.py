import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def create_fig1():
    # Figure dimensions: 16.0 x 4.4 (clean widescreen IEEE ratio)
    fig_w, fig_h = 16.0, 4.4
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis('off')

    # Color palette
    c_stage_bg = "#F8FAFC"
    c_stage_border = "#CBD5E1"
    
    c_box_bg = "#FFFFFF"
    c_box_border = "#475569"
    c_box_title = "#0F172A"
    c_box_text = "#334155"
    
    c_header_bg = "#F1F5F9"
    c_header_border = "#CBD5E1"
    c_header_title = "#0F172A"
    c_header_sub = "#64748B"
    
    # Accent colors
    c_blue_line = "#2563EB"
    c_blue_bg = "#EFF6FF"
    c_blue_border = "#3B82F6"
    c_blue_text = "#1D4ED8"
    
    c_purple_bg = "#FAF5FF"
    c_purple_border = "#7C3AED"
    c_purple_text = "#6D28D9"

    c_arrow = "#334155"

    # Stage columns across [0.015, 0.985]
    stages = [
        (0.015, 0.228, "STAGE 1: DATA INGESTION", "Sensors & Physical Network Topology"),
        (0.262, 0.228, "STAGE 2: PREPROCESSING", "Leak-Free Normalization & Laplacian"),
        (0.510, 0.228, "STAGE 3: SPATIOTEMPORAL GCN", "Residual Graph Convolutions"),
        (0.757, 0.228, "STAGE 4: RECONSTRUCTION", "Identity Skip & Physical Speed")
    ]
    
    container_y = 0.025
    container_h = 0.950

    # Draw Stage Containers with header banners
    for sx, sw, stitle, sdesc in stages:
        rect = patches.FancyBboxPatch(
            (sx, container_y), sw, container_h,
            boxstyle="round,pad=0.005,rounding_size=0.012",
            facecolor=c_stage_bg, edgecolor=c_stage_border, lw=1.2, zorder=1
        )
        ax.add_patch(rect)
        
        # Header banner inside container
        h_h = 0.082
        h_y = container_y + container_h - h_h - 0.012
        h_w = sw - 0.020
        h_x = sx + 0.010
        h_rect = patches.FancyBboxPatch(
            (h_x, h_y), h_w, h_h,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            facecolor=c_header_bg, edgecolor=c_header_border, lw=1.0, zorder=2
        )
        ax.add_patch(h_rect)
        
        ax.text(sx + sw/2, h_y + h_h - 0.026, stitle,
                ha='center', va='center', weight='bold', fontsize=9.6, color=c_header_title, zorder=3)
        ax.text(sx + sw/2, h_y + 0.024, sdesc,
                ha='center', va='center', fontsize=7.8, color=c_header_sub, weight='semibold', zorder=3)

    # Box drawing helper
    def draw_node(x, y, w, h, title, lines=None, bg=c_box_bg, border=c_box_border, lw=1.3,
                  title_color=c_box_title, text_color=c_box_text, title_fs=10.0, text_fs=8.2):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            facecolor=bg, edgecolor=border, lw=lw, zorder=4
        )
        ax.add_patch(box)
        
        if lines:
            title_y = y + h - 0.042
            ax.text(x + w/2, title_y, title, ha='center', va='center',
                    weight='bold', fontsize=title_fs, color=title_color, zorder=5)
            n_lines = len(lines)
            start_y = title_y - 0.038
            line_gap = (start_y - (y + 0.025)) / max(n_lines - 1, 1) if n_lines > 1 else 0
            for idx, line in enumerate(lines):
                ly = start_y - idx * line_gap if n_lines > 1 else start_y - 0.015
                ax.text(x + w/2, ly, line, ha='center', va='center',
                        fontsize=text_fs, color=text_color, weight='medium', zorder=5)
        else:
            ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                    weight='bold', fontsize=title_fs, color=title_color, zorder=5)

    # Unified vertical card positions
    box_h = 0.315
    top_y = 0.465   # Top row: 0.465 to 0.780
    bot_y = 0.060   # Bottom row: 0.060 to 0.375
    card_w = 0.208

    def draw_arrow(x1, y1, x2, y2, color=c_arrow, lw=1.4, zorder=6):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=lw,
                                    headwidth=5.5, headlength=6.0, shrink=0.005), zorder=zorder)

    # -------------------------------------------------------------
    # STAGE 1 NODES
    # -------------------------------------------------------------
    s1_x = 0.025
    draw_node(s1_x, top_y, card_w, box_h, "Freeway Speed Series",
              [r"Loop Detectors ($N=207$ / $325$)",
               r"Lookback: $T_{\mathrm{in}} = 12$ steps ($60$ min)",
               r"Raw Speeds $X_{\mathrm{raw}} \in \mathbb{R}^{B \times N \times 12}$"])
    
    draw_node(s1_x, bot_y, card_w, box_h, "Road Network Geometry",
              [r"Network Distance Matrix $D \in \mathbb{R}^{N \times N}$",
               r"Shortest Highway Distance $\mathrm{dist}(u, v)$",
               r"Physical Sensor Graph Topology"])

    # -------------------------------------------------------------
    # STAGE 2 NODES
    # -------------------------------------------------------------
    s2_x = 0.272
    draw_node(s2_x, top_y, card_w, box_h, "Leak-Free Normalizer",
              [r"Train-split Statistics: $\mu, \sigma$",
               r"Extract Origin: $x_t \in \mathbb{R}^{B \times N \times 1}$",
               r"Normalized Lookback $X \in \mathbb{R}^{B \times N \times 12}$"])
    
    draw_node(s2_x, bot_y, card_w, box_h, "Graph Adjacency Matrix",
              [r"Gaussian Kernel: $W_{ij} = \exp(-d^2/\sigma^2)$",
               r"Self-Loops Added: $\tilde{A} = W + I_N$",
               r"Laplacian: $\tilde{A}_{\mathrm{norm}} = \tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}$"])

    # Connections between Stage 1 and Stage 2
    draw_arrow(s1_x + card_w, top_y + box_h/2, s2_x, top_y + box_h/2)
    draw_arrow(s1_x + card_w, bot_y + box_h/2, s2_x, bot_y + box_h/2)

    # -------------------------------------------------------------
    # STAGE 3 NODES (Spatiotemporal GCN)
    # -------------------------------------------------------------
    s3_x = 0.520
    draw_node(s3_x, top_y, card_w, box_h, "GCN Layer 1 + ReLU",
              [r"Temporal Projection: $W^{(0)} \in \mathbb{R}^{12 \times 64}$",
               r"$H^{(1)} = \mathrm{ReLU}(\tilde{A}_{\mathrm{norm}} X W^{(0)})$",
               r"Latent Graph State: $H^{(1)} \in \mathbb{R}^{B \times N \times 64}$"])
    
    draw_node(s3_x, bot_y, card_w, box_h, "GCN Layer 2 (Residual)",
              [r"Velocity Projection: $W^{(1)} \in \mathbb{R}^{64 \times 1}$",
               r"$\Delta\hat{y} = \tilde{A}_{\mathrm{norm}} H^{(1)} W^{(1)}$",
               r"Residual Momentum: $\Delta\hat{y} \in \mathbb{R}^{B \times N \times 1}$"])

    # Connections into Stage 3:
    # 1. Normalizer -> GCN 1 (Input tensor X)
    draw_arrow(s2_x + card_w, top_y + box_h/2, s3_x, top_y + box_h/2)
    ax.text((s2_x + card_w + s3_x)/2, top_y + box_h/2 + 0.022, r"$X$",
            ha='center', va='bottom', fontsize=9.8, weight='bold', color=c_box_title)

    # 2. Graph Laplacian -> GCN 2 and branch to GCN 1
    draw_arrow(s2_x + card_w, bot_y + box_h/2, s3_x, bot_y + box_h/2)
    lap_branch_x = (s2_x + card_w + s3_x) / 2
    # Vertical branch up to GCN 1
    ax.plot([lap_branch_x, lap_branch_x, s3_x],
            [bot_y + box_h/2, top_y + 0.08, top_y + 0.08],
            color=c_arrow, lw=1.4, zorder=5)
    ax.annotate('', xy=(s3_x, top_y + 0.08), xytext=(s3_x - 0.012, top_y + 0.08),
                arrowprops=dict(facecolor=c_arrow, edgecolor=c_arrow, width=1.4,
                                headwidth=5.5, headlength=6.0), zorder=6)
    ax.text(lap_branch_x, (top_y + 0.08 + bot_y + box_h/2)/2, r"$\tilde{A}_{\mathrm{norm}}$",
            ha='center', va='center', fontsize=9.2, weight='bold', color=c_box_title,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor=c_stage_border, lw=0.8),
            zorder=7)

    # 3. GCN 1 -> GCN 2 (Latent state H^(1))
    draw_arrow(s3_x + card_w/2, top_y, s3_x + card_w/2, bot_y + box_h)
    ax.text(s3_x + card_w/2 + 0.010, (top_y + bot_y + box_h)/2, r"$H^{(1)}$ (64-dim)",
            ha='left', va='center', fontsize=8.6, weight='bold', color=c_box_title)

    # -------------------------------------------------------------
    # STAGE 4 NODES (Momentum Reconstruction)
    # -------------------------------------------------------------
    s4_x = 0.767
    
    # Residual Reconstructor Box (Top)
    draw_node(s4_x, top_y, card_w, box_h, r"Residual Addition $\bigoplus$",
              [r"Base Speed + Residual Momentum",
               r"$\hat{y}_{\mathrm{norm}} = x_t + \Delta\hat{y}$",
               r"Eliminates 1-Step Lag Attractor",
               r"Variance Contraction: $2(1-\rho) \ll 1$"],
              bg=c_purple_bg, border=c_purple_border, lw=1.5,
              title_color=c_purple_text, text_color="#4C1D95", title_fs=10.2)

    # Physical Speed Forecast Box (Bottom)
    draw_node(s4_x, bot_y, card_w, box_h, "Physical Speed Forecast",
              [r"Inverse Transform: $\hat{y}_{\mathrm{phys}} = \hat{y}_{\mathrm{norm}} \cdot \sigma + \mu$",
               r"Forecasted Speed $\hat{y}_{t+h} \in \mathbb{R}^N$ (mph)",
               r"Multi-Horizon: $h \in \{5, 15, 30, 60\}$ min",
               r"Zero-Lag Shockwave Tracking"],
              bg=c_blue_bg, border=c_blue_border, lw=1.8,
              title_color=c_blue_text, text_color="#1E3A8A")

    # Downward connection from Residual Addition to Physical Forecast
    draw_arrow(s4_x + card_w/2, top_y, s4_x + card_w/2, bot_y + box_h)
    ax.text(s4_x + card_w/2 + 0.010, (top_y + bot_y + box_h)/2,
            r"$\hat{y}_{\mathrm{norm}}$ (De-normalize)",
            ha='left', va='center', fontsize=8.4, weight='bold', color=c_blue_text)

    # -------------------------------------------------------------
    # HIGHWAY SKIP CONNECTION: Current Speed x_t (Lookback Origin)
    # -------------------------------------------------------------
    # Routing cleanly through open channel at y = 0.816
    skip_y = 0.816
    ax.plot([s2_x + card_w/2, s2_x + card_w/2, s4_x + card_w/2, s4_x + card_w/2],
            [top_y + box_h, skip_y, skip_y, top_y + box_h],
            color=c_blue_line, lw=1.8, zorder=5)
    # Downward arrow into Residual Addition card
    ax.annotate('', xy=(s4_x + card_w/2, top_y + box_h), xytext=(s4_x + card_w/2, top_y + box_h + 0.014),
                arrowprops=dict(facecolor=c_blue_line, edgecolor=c_blue_line, width=1.8,
                                headwidth=6.0, headlength=6.5), zorder=6)

    # Pill badge for Skip Connection centered cleanly within Stage 3
    pill_w = 0.195
    pill_h = 0.038
    pill_cx = s3_x + card_w/2  # 0.624 (centered directly over GCN Layer 1)
    pill_x = pill_cx - pill_w/2
    pill = patches.FancyBboxPatch(
        (pill_x, skip_y - pill_h/2), pill_w, pill_h,
        boxstyle="round,pad=0.003,rounding_size=0.008",
        facecolor="#FFFFFF", edgecolor=c_blue_border, lw=1.3, zorder=7
    )
    ax.add_patch(pill)
    ax.text(pill_cx, skip_y, r"Identity Skip: Origin $x_t$",
            ha='center', va='center', fontsize=8.6, weight='bold', color=c_blue_text, zorder=8)

    # -------------------------------------------------------------
    # RESIDUAL MOMENTUM CONNECTION: GCN 2 -> Residual Addition
    # -------------------------------------------------------------
    # Exits GCN 2 right side, routes up to Stage 4 Top left side
    res_x = s3_x + card_w
    res_y_out = bot_y + box_h/2
    res_mid_x = (res_x + s4_x) / 2
    res_y_in = top_y + box_h/2

    ax.plot([res_x, res_mid_x, res_mid_x, s4_x],
            [res_y_out, res_y_out, res_y_in, res_y_in],
            color=c_arrow, lw=1.5, zorder=5)
    ax.annotate('', xy=(s4_x, res_y_in), xytext=(s4_x - 0.012, res_y_in),
                arrowprops=dict(facecolor=c_arrow, edgecolor=c_arrow, width=1.5,
                                headwidth=5.5, headlength=6.0), zorder=6)
    
    # Horizontal label badge in clear space between stages
    ax.text(res_mid_x, (res_y_out + res_y_in)/2,
            r"$\Delta\hat{y}$",
            ha='center', va='center', fontsize=9.8, weight='bold', color=c_header_title,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFFFFF", edgecolor=c_stage_border, lw=0.8),
            zorder=7)

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    
    out_path_png = Path("paper/figures/fig1_system_architecture.png")
    out_path_pdf = Path("paper/figures/fig1_system_architecture.pdf")
    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    # Save PNG for preview at 300 DPI
    plt.savefig(out_path_png, dpi=300, bbox_inches='tight')
    # Save PDF for LaTeX inclusion (vector, losslessly scalable, IEEE camera-ready standard)
    plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print("Successfully generated fig1_system_architecture.png + fig1_system_architecture.pdf (vector)")

def create_fig2():
    import numpy as np
    import pandas as pd
    from src.config import RAW_DIR, CACHE_DIR

    with open(RAW_DIR / 'graph_sensor_ids.txt') as f:
        sensor_ids = f.read().strip().split(',')
    num_sensors = len(sensor_ids)

    loc_df = pd.read_csv(RAW_DIR / 'graph_sensor_locations.csv')
    loc_df['sensor_id'] = loc_df['sensor_id'].astype(str)
    loc_df = loc_df.set_index('sensor_id').reindex(sensor_ids).reset_index()

    adj = np.load(CACHE_DIR / 'adj_mx.npy')
    deg = (adj > 0).sum(axis=1)

    fig, ax = plt.subplots(figsize=(3.45, 2.05), dpi=300)

    # Draw highway links
    for i in range(num_sensors):
        for j in range(num_sensors):
            if adj[i, j] > 0 and i != j:
                ax.plot([loc_df.longitude[i], loc_df.longitude[j]],
                        [loc_df.latitude[i], loc_df.latitude[j]],
                        color='#64748B', linewidth=0.50, alpha=0.45, zorder=1)

    # Draw sensor nodes
    sc = ax.scatter(loc_df.longitude, loc_df.latitude, c=deg, cmap='viridis',
                    s=16, edgecolor='#0F172A', linewidth=0.30, zorder=3)

    # Corridor labels
    ax.text(-118.50, 34.165, 'US-101', fontsize=6.2, weight='bold', color='#1E293B',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='#F1F5F9', edgecolor='#94A3B8', lw=0.5))
    ax.text(-118.25, 34.055, 'I-10 / DTLA', fontsize=6.2, weight='bold', color='#1E293B',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='#F1F5F9', edgecolor='#94A3B8', lw=0.5))
    ax.text(-118.32, 34.195, 'I-210 Foothill', fontsize=6.2, weight='bold', color='#1E293B',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='#F1F5F9', edgecolor='#94A3B8', lw=0.5))

    ax.set_xlabel('Longitude (°W)', fontsize=7.2, labelpad=1.5)
    ax.set_ylabel('Latitude (°N)', fontsize=7.2, labelpad=1.5)
    ax.tick_params(axis='both', which='major', labelsize=6.2, pad=1)

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, fraction=0.044, pad=0.025)
    cbar.set_label('Degree $d_i$', fontsize=6.8, labelpad=1.5)
    cbar.ax.tick_params(labelsize=5.8)

    ax.grid(True, linestyle='--', alpha=0.35, color='#CBD5E1')
    plt.tight_layout(pad=0.25)
    out_png = Path('paper/figures/fig2_sensor_network_map.png')
    out_pdf = Path('paper/figures/fig2_sensor_network_map.pdf')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close()
    print("Successfully generated fig2_sensor_network_map.png + fig2_sensor_network_map.pdf (vector)")

if __name__ == "__main__":
    create_fig1()
    create_fig2()
