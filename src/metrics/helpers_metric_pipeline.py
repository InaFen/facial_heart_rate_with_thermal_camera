from matplotlib import pyplot as plt

def save_single_plot(ax, path):
    fig_single, ax_single = plt.subplots(figsize=(14, 5))

    # copy all artists from original axis
    for line in ax.get_lines():
        ax_single.plot(
            line.get_xdata(),
            line.get_ydata(),
            label=line.get_label(),
            alpha=line.get_alpha(),
            color=line.get_color(),
            marker=line.get_marker(),
            linewidth=line.get_linewidth(),
        )

    for coll in ax.collections:
        offsets = coll.get_offsets()
        if len(offsets):
            ax_single.scatter(
                offsets[:, 0],
                offsets[:, 1],
                s=coll.get_sizes(),
                alpha=coll.get_alpha(),
            )

    ax_single.set_xlabel(ax.get_xlabel())
    ax_single.set_ylabel(ax.get_ylabel())
    ax_single.grid(alpha=0.2)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax_single.legend(loc="best")

    fig_single.tight_layout()
    fig_single.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig_single)