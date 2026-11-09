import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from celluloid import Camera
from IPython.display import HTML
from matplotlib.collections import PathCollection
from matplotlib.transforms import Affine2D


def plot_current_state_of_graph(
    G,
    lim=(50.1, 54.8),
    limx=(9.65, 15.1),
    node_size=50,
    arrowsize=20,
    fs=(10, 10),
    font_size=1,
    font_color="black",
    save=False,
    river_map=0,
    ger_map=True,
    emphasize=None,
    label=True,
    autozoom=None,
    width=1,
    show_edge_origin=False,
    hardcode_colors=None,
    ger_path="product/visualization/east_germany/east_german_map.shp",
    river_path="product/visualization/east_germany/river_east_german_map.shp",
    extra_points=None,
    river_width=0.5,
    title="Rivers East Germany",
    pos=False,
    ax=False,
    rotate_by=0,  # TODO doesnt work.
    custom_legend=None,
    log_scale=None,
):

    if extra_points is None:
        extra_points = []
    if hardcode_colors is None:
        hardcode_colors = []
    if emphasize is None:
        emphasize = []
    if not pos:
        pos = {x: np.flip(np.array(G.nodes[x]["p"][:2]).astype(float)) for x in G.nodes}

    if not ax:
        _fig, ax = plt.subplots(1, 1, figsize=fs)

    if ger_map:
        fp = ger_path
        map_df2 = gpd.read_file(fp)
        if rotate_by != 0:
            map_df2 = map_df2.rotate(rotate_by)
        map_df2.plot(color="green", ax=ax, alpha=0.3, linewidth=5, edgecolor="black")

    if river_map:
        fp = river_path
        map_df = gpd.read_file(fp)
        if rotate_by != 0:
            map_df = map_df.rotate(rotate_by)
        map_df.plot(color="blue", alpha=0.3, ax=ax, linewidth=river_width, edgecolor="blue")

    if hardcode_colors:
        colors = hardcode_colors
    else:
        colors = []
        for x in G.nodes:
            if x in emphasize:
                colors.append("black")
            else:
                colors.append(G.nodes[x]["c"])
    if show_edge_origin:
        cmap = mpl.colormaps["Set1"]
        ege_base_colors = cmap(np.linspace(0, 1, 8))
        edge_colors = []
        for x in G.edges:
            edge_colors.append(tuple(ege_base_colors[G.edges[x]["origin"]]))
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=label,
        font_size=font_size,
        font_color=font_color,
        node_size=node_size,
        arrows=True,
        node_color=colors,
        arrowsize=arrowsize,
        edge_color=edge_colors if show_edge_origin else "black",
        width=width,
        ax=ax,
    )

    if rotate_by != 0:
        r = Affine2D().rotate_deg(rotate_by)  # Create a rotation transformation
        for x in ax.images + ax.lines + ax.collections:
            trans = x.get_transform()
            x.set_transform(r + trans)
            if isinstance(x, PathCollection):
                transoff = x.get_offset_transform()
                x._transOffset = r + transoff  # ax.set_frame_on(True)
    if autozoom:
        ax.set_xlim(
            min([pos[x][0] for x in pos]) - autozoom,
            max([pos[x][0] for x in pos]) + autozoom,
        )
        ax.set_ylim(
            min([pos[x][1] for x in pos]) - autozoom,
            max([pos[x][1] for x in pos]) + autozoom,
        )
    else:
        if lim:
            ax.set_ylim(lim[0], lim[1])
            ax.set_xlim(limx[0], limx[1])
        else:
            pass

    if custom_legend:
        ax.legend(handles=custom_legend[0], loc=custom_legend[1])

    if log_scale:
        ax.set_xscale("log", base=3.75)

        ax.set_yscale("asinh")

    ax.set_title(title)
    # doesnt work...

    if len(extra_points):
        for ex in extra_points:
            ax.scatter(ex[0], ex[1], color="pink", s=100, edgecolors="black")
            ax.annotate(ex[2], (ex[0], ex[1]))

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=500)
    else:
        plt.show()


def simple_sample_display(sample_data):
    _fix, axs = plt.subplots(len(sample_data.T), 1, figsize=(3 * len(sample_data.T), 10))
    for n, x in enumerate(sample_data.columns):
        sample_data[x].plot(ax=axs[n])
        axs[n].set_ylabel("m³/s")
    plt.show()


def simple_sample_display_2(sample_data):
    _fig, axs = plt.subplots(len(sample_data.columns), 1)
    cmap = mpl.colormaps["plasma"]
    # Take colors at regular intervals spanning the colormap.
    for n, s in enumerate(sample_data.columns):
        axs[n].set_ylabel(s, fontstyle="normal", fontsize=8, rotation=45)
        rgba = cmap(1 / (n + 1))
        axs[n].plot(sample_data[s].values, linewidth=2, color=rgba)
        axs[n].get_xaxis().set_ticks([])
        axs[n].get_yaxis().set_ticks([])
    axs[n].get_yaxis().set_ticks([])
    axs[n].set_xlabel("Timesteps")
    plt.show()


def fancy_plot(sample_data, base_c, save=0, emph=19999, label_nodes=True):
    _fig, axs = plt.subplots(3, 1, figsize=(12, 5))
    for n, x in enumerate(sample_data.columns):
        axs[n].plot(sample_data[x], linewidth=2, color=base_c[n + 1], alpha=0.8)
        axs[n].set_ylabel("m³/s", fontsize=15)
        position = (
            axs[n].get_xbound()[0] + 150,
            sample_data[x].max() - (sample_data[x].max() - sample_data[x].min()) / 7,
        )
        if label_nodes:
            axs[n].scatter(position[0], position[1], s=900, color=base_c[n + 1])
            offset = 27 if len(x) == 2 else 35
            axs[n].text(
                position[0] - offset,
                position[1],
                x,
                verticalalignment="center",
                fontstyle="italic",
                fontsize=12,
            )
        axs[n].set_xlabel(None)
        axs[n].tick_params(axis="both", which="major", labelsize=12)

        limit = axs[n].get_xlim()

        axs[n].vlines(emph, sample_data[x].min(), sample_data[x].max(), color="red", linewidth=2)
        axs[n].vlines(emph + 150, sample_data[x].min(), sample_data[x].max(), color="red", linewidth=2)
        axs[n].hlines(0, emph, emph + 150, color="red", linewidth=2)
        axs[n].hlines(sample_data[x].max(), emph, emph + 150, color="red", linewidth=2)
        axs[n].set_xlim(limit[0], limit[1])

    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])

    axs[n].set_xlabel("Year", fontsize=14)
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=500)
    else:
        plt.show()


def bare_fancy_plot(sample_data, save, base_c):

    _fig, axs = plt.subplots(3, 1, figsize=(8, 5))
    labels = ["A", "B", "C"]
    for n, x in enumerate(sample_data.columns):
        axs[n].plot(sample_data[x], linewidth=3, color=base_c[n + 1], alpha=1)
        axs[n].set_ylabel(labels[n], fontsize=25)
        axs[n].set_xlabel(None)
        axs[n].tick_params(axis="both", which="major", labelsize=12)
        axs[n].set_yticklabels([])

    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])

    axs[n].set_xlabel("Year", fontsize=14)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=500)
    else:
        plt.show()


def animate_ts(
    sample_data,
    steps=50,
    length=5000,
    colors=None,
):
    if colors is None:
        colors = ["darkred", "darkblue", "darkgreen", "darkorange", "darkviolet"]
    fig, axs = plt.subplots(sample_data.shape[1], 1, figsize=(16, 9))
    data = sample_data.apply(lambda x: np.log(x))
    for x in range(data.shape[1]):
        axs[x].yaxis.set_visible(False)
        axs[x].xaxis.set_visible(False)
    plt.tight_layout()
    camera = Camera(fig)
    for x in range(steps):
        for y in range(data.shape[1]):
            axs[y].plot(
                data.iloc[x : length + x][data.columns[y]].values,
                color=colors[y],
                linewidth=2,
            )
        camera.snap()
    animation = camera.animate()
    return HTML(animation.to_html5_video())
