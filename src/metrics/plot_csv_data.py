
from typing import Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_metric_boxplots_per_camera(
    csv_path: str,
    save_path: Optional[str] = None,
    camera_col: str = "camera",
    score_col: str = "f1_distance_based",
    roi_col: str = "roi",
    used_rois: Optional[list[str]] = None,
    color_by: Optional[str] = "run",
    marker_by: Optional[str] = "roi",
    label_by: Optional[str] = "participant",
    figsize_per_camera: tuple = (6, 6),
    jitter_width: float = 0.12,
    random_seed: int = 42,
):
    """
    Plot metric boxplots with individual data points for each camera.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    save_path : str, optional
        Path to save the resulting figure.
    camera_col : str
        Column identifying cameras.
    score_col : str
        Column containing metric.
    roi_col : str
        Column identifying ROIs.
    used_rois : list[str], optional
        Only include rows whose ROI is in this list.
        If None, all ROIs are included.
    color_by : str, optional
        Column used to determine point colors.
    marker_by : str, optional
        Column used to determine marker shapes.
    label_by : str, optional
        Column used to determine small letter labels next to points.
    figsize_per_camera : tuple
        Figure size per camera.
    jitter_width : float
        Horizontal jitter applied to scatter points.
    random_seed : int
        Random seed for reproducible jitter.
    """

    # Load data
    df = pd.read_csv(csv_path)

    # Filter selected ROIs
    if used_rois is not None:
        df = df[df[roi_col].isin(used_rois)].copy()

        if df.empty:
            raise ValueError(
                f"No rows found for the selected ROIs: {used_rois}"
            )

    # Check required columns
    required_columns = [camera_col, score_col]

    for col in [color_by, marker_by, label_by]:
        if col is not None:
            required_columns.append(col)

    if roi_col not in required_columns:
        required_columns.append(roi_col)

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in CSV: {missing_columns}"
        )

    # Remove rows with missing values in relevant columns
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=required_columns)

    if df.empty:
        raise ValueError("No valid data available after filtering.")

    # Get cameras
    cameras = sorted(df[camera_col].unique(), key=str)
    n_cameras = len(cameras)

    # Create figure
    fig, axes = plt.subplots(
        1,
        n_cameras,
        figsize=(figsize_per_camera[0] * n_cameras,
                 figsize_per_camera[1]),
        sharey=True,
    )

    if n_cameras == 1:
        axes = [axes]

    rng = np.random.default_rng(random_seed)

    # Define marker styles
    marker_styles = [
        "o", "s", "^", "D", "v", "P", "X",
        "<", ">", "h", "*", "p", "8"
    ]

    # Helper function to create consistent mappings
    def create_mapping(values):
        unique_values = sorted(values.dropna().unique(), key=str)
        return {
            value: i for i, value in enumerate(unique_values)
        }

    # Create mappings for colors, markers and labels
    color_mapping = (
        create_mapping(df[color_by])
        if color_by is not None else {}
    )

    marker_mapping = (
        create_mapping(df[marker_by])
        if marker_by is not None else {}
    )

    label_mapping = (
        create_mapping(df[label_by])
        if label_by is not None else {}
    )

    # Color palette
    color_values = plt.get_cmap("tab10").colors

    # Plot each camera
    for ax, camera in zip(axes, cameras):

        camera_df = df[df[camera_col] == camera].copy()

        scores = camera_df[score_col].values

        # Boxplot
        ax.boxplot(
            scores,
            positions=[1],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor="lightgrey", alpha=0.5),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(
                marker="",
                markersize=0,
            ),
        )

        # Scatter points
        for _, row in camera_df.iterrows():

            x = 1 + rng.uniform(-jitter_width, jitter_width)
            y = row[score_col]

            # Determine color
            if color_by is not None:
                color_idx = color_mapping[row[color_by]]
                point_color = color_values[
                    color_idx % len(color_values)
                ]
            else:
                point_color = "tab:blue"

            # Determine marker
            if marker_by is not None:
                marker_idx = marker_mapping[row[marker_by]]
                marker = marker_styles[
                    marker_idx % len(marker_styles)
                ]
            else:
                marker = "o"

            ax.scatter(
                x,
                y,
                color=point_color,
                marker=marker,
                s=55,
                edgecolors="black",
                linewidths=0.5,
                zorder=3,
            )

            # Add small letter label
            if label_by is not None:
                label_idx = label_mapping[row[label_by]]

                # Convert index to a, b, ..., z, a1, b1, ...
                letter = chr(97 + label_idx % 26)
                suffix = label_idx // 26

                if suffix > 0:
                    letter += str(suffix)

                ax.annotate(
                    letter,
                    (x, y),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=8,
                    color="black",
                )

        # Axis formatting
        ax.set_title(str(camera))
        ax.set_xticks([1])
        ax.set_xticklabels([score_col])
        ax.set_xlim(0.5, 1.5)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel(score_col)

    # Create legends
    legend_handles = []

    # Color legend
    if color_by is not None:
        color_handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=color_values[i % len(color_values)],
                markeredgecolor="black",
                markersize=8,
                label=str(value),
            )
            for value, i in color_mapping.items()
        ]

        legend_handles.append(
            (f"Color: {color_by}", color_handles)
        )

    # Marker legend
    if marker_by is not None:
        marker_handles = [
            Line2D(
                [0], [0],
                marker=marker_styles[i % len(marker_styles)],
                color="black",
                linestyle="None",
                markersize=8,
                label=str(value),
            )
            for value, i in marker_mapping.items()
        ]

        legend_handles.append(
            (f"Marker: {marker_by}", marker_handles)
        )

    # Label legend
    if label_by is not None:
        label_handles = []

        for value, i in label_mapping.items():
            letter = chr(97 + i % 26)
            suffix = i // 26

            if suffix > 0:
                letter += str(suffix)

            label_handles.append(
                Line2D(
                    [0], [0],
                    marker=f"${letter}$",
                    color="black",
                    linestyle="None",
                    markersize=9,
                    label=str(value),
                )
            )

        legend_handles.append(
            (f"Label: {label_by}", label_handles)
        )

    # Place legends below the plots
    if legend_handles:
        n_legends = len(legend_handles)
        legend_space = 0.16 + n_legends * 0.065

        for i, (title, handles) in enumerate(legend_handles):
            fig.legend(
                handles=handles,
                title=title,
                loc="lower center",
                bbox_to_anchor=(
                    0.5,
                    0.16 - i * 0.065
                ),
                ncol=min(len(handles), 8),
                frameon=True,
                fontsize=8,
            )

        plt.tight_layout(
            rect=[0, legend_space, 1, 1]
        )

    else:
        plt.tight_layout()

    # Save figure
    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

    return fig


if __name__ == "__main__":
    plot_metric_boxplots_per_camera(
        csv_path="saved_results/all_npz_metrics.csv",
        save_path="saved_results/new_approach/results/evaluation_csv/f1_boxplot.png",
        used_rois=["whole_face", "manual_rectangle_front_view", "forehead", "manual_rectangle_forehead", "manual_rectangle"],
        color_by="participant",
        marker_by="roi",
        label_by="run",
        score_col="f1_distance_based"
    )

    plot_metric_boxplots_per_camera(
            csv_path="saved_results/all_npz_metrics.csv",
            save_path="saved_results/new_approach/results/evaluation_csv/f1_halved_boxplot.png",
            used_rois=["whole_face", "manual_rectangle_front_view", "forehead", "manual_rectangle_forehead", "manual_rectangle"],
            color_by="participant",
            marker_by="roi",
            label_by="run",

        )
