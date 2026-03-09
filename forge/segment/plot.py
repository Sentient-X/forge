"""Timeline visualization for segmentation results."""

from __future__ import annotations

from pathlib import Path

from forge.segment.models import SegmentationReport

# Maximum episodes to show before truncating
_MAX_ROWS = 50


def plot_segmentation(report: SegmentationReport, output_path: str | Path) -> None:
    """Generate a horizontal timeline PNG showing segments per episode.

    Each episode is a row with colored rectangles for segments and
    vertical lines at changepoints.

    Args:
        report: Segmentation report with per-episode results.
        output_path: Path to save the PNG file.

    Raises:
        MissingDependencyError: If matplotlib is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend — no Qt/display needed
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        from forge.core.exceptions import MissingDependencyError

        raise MissingDependencyError(
            dependency="matplotlib",
            feature="segment --plot",
            install_hint="pip install forge-robotics[visualize]",
        )

    episodes = report.per_episode
    if not episodes:
        return

    truncated = len(episodes) > _MAX_ROWS
    if truncated:
        episodes = episodes[:_MAX_ROWS]

    n_episodes = len(episodes)
    fig_height = max(3, 0.4 * n_episodes + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    cmap = plt.get_cmap("tab10")

    for row, ep in enumerate(episodes):
        if not ep.segments:
            continue
        for seg in ep.segments:
            color = cmap(seg.start % 10)
            rect = Rectangle(
                (seg.start, row - 0.35),
                seg.duration_frames,
                0.7,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.8,
            )
            ax.add_patch(rect)

        # Changepoint lines
        for cp in ep.changepoints:
            ax.plot([cp, cp], [row - 0.4, row + 0.4], color="black", linewidth=1, alpha=0.7)

    # Labels and formatting
    ax.set_yticks(range(n_episodes))
    ax.set_yticklabels(
        [ep.episode_id for ep in episodes],
        fontsize=max(6, 10 - n_episodes // 10),
    )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Episode")

    max_frames = max((ep.num_frames for ep in episodes), default=100)
    ax.set_xlim(0, max_frames)
    ax.set_ylim(-0.5, n_episodes - 0.5)
    ax.invert_yaxis()

    title = f"Segmentation: {report.dataset_path}"
    if report.summary:
        title += f"  (mean={report.summary.get('mean_segments', '?')} segments/ep)"
    if truncated:
        title += f"  [showing {_MAX_ROWS}/{report.num_episodes}]"
    ax.set_title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
