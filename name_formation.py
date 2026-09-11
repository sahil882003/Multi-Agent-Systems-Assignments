

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import Video, display


rng = np.random.default_rng(11004)


def connected_graph(n, p, seed=41):
    """Generate an Erdos-Renyi graph repeatedly until it is connected."""
    current_seed = seed
    while True:
        graph = nx.erdos_renyi_graph(n, p, seed=current_seed)
        if nx.is_connected(graph):
            return graph


def line_points(start, end, number):
    """Return equally spaced 2-D points along a line segment."""
    t = np.linspace(0.0, 1.0, number, endpoint=False)
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return start + t[:, None] * (end - start)


def make_letter(strokes, number_of_agents=20):
    """Convert line-segment strokes into exactly number_of_agents targets."""
    points = np.vstack(
        [line_points(start, end, count) for start, end, count in strokes]
    )
    return points

# Each tuple contains: (start point, end point, number of agents on stroke).

LETTER_STROKES = {
    "S": [
        ((1, 2), (0, 2), 4),
        ((0, 2), (0, 1), 4),
        ((0, 1), (1, 1), 4),
        ((1, 1), (1, 0), 4),
        ((1, 0), (0, 0), 4),
    ],
    "A": [
        ((0, 0), (0.5, 2), 7),
        ((0.5, 2), (1, 0), 7),
        ((0.23, 0.9), (0.77, 0.9), 6),
    ],
    "H": [
        ((0, 0), (0, 2), 7),
        ((1, 0), (1, 2), 7),
        ((0, 1), (1, 1), 6),
    ],
    "I": [
        ((0, 2), (1, 2), 5),
        ((0.5, 2), (0.5, 0), 10),
        ((0, 0), (1, 0), 5),
    ],
    "L": [
        ((0, 2), (0, 0), 10),
        ((0, 0), (1, 0), 10),
    ],
}


def run_simulation():
    number_of_agents = 20
    edge_probability = 0.22
    time_step = 0.055
    anchoring_gain = 0.35
    steps_per_letter = 90

    graph = connected_graph(number_of_agents, edge_probability, seed=41)
    adjacency = nx.to_numpy_array(graph, nodelist=range(number_of_agents))
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency

    targets = {
        letter: make_letter(strokes, number_of_agents)
        for letter, strokes in LETTER_STROKES.items()
    }

    # Random initial positions of all agents in R^2.
    positions = rng.uniform(-1.5, 1.5, size=(number_of_agents, 2))
    frames = []

    for letter in "SAHIL":
        desired = targets[letter]
        desired = desired - desired.mean(axis=0)  # centre the formation

        for _ in range(steps_per_letter):
            # u = -L(x-r) + gamma(r-x)
            control = -laplacian @ (positions - desired)
            control += anchoring_gain * (desired - positions)
            positions = positions + time_step * control
            frames.append((letter, positions.copy()))

    # Plot the connected communication graph.
    plt.figure(figsize=(7, 5))
    graph_layout = nx.spring_layout(graph, seed=4)
    nx.draw(
        graph,
        graph_layout,
        with_labels=True,
        node_color="skyblue",
        edge_color="gray",
        node_size=450,
    )
    plt.title("Connected Erdos-Renyi graph: N=20, p=0.22")
    plt.tight_layout()
    plt.savefig("sahil_communication_graph.png", dpi=200, bbox_inches="tight")
    plt.show()

    # Animate the same graph while its agents form each letter.
    figure, axis = plt.subplots(figsize=(6, 6))
    scatter = axis.scatter([], [], s=65, color="royalblue", zorder=2)
    edge_lines = [
        axis.plot([], [], color="lightgray", linewidth=0.7, zorder=1)[0]
        for _ in graph.edges()
    ]
    title = axis.set_title("")
    axis.set_xlim(-1.8, 1.8)
    axis.set_ylim(-1.5, 1.7)
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.set_xlabel("x position")
    axis.set_ylabel("y position")

    def update(frame_index):
        letter, current_positions = frames[frame_index]
        scatter.set_offsets(current_positions)
        title.set_text(f"Formation control: {letter}")

        for line, (i, j) in zip(edge_lines, graph.edges()):
            line.set_data(
                current_positions[[i, j], 0],
                current_positions[[i, j], 1],
            )
        return [scatter, title, *edge_lines]

    animation = FuncAnimation(
        figure,
        update,
        frames=range(0, len(frames), 3),
        interval=50,
        blit=False,
    )

    video_name = "sahil_formation.mp4"
    animation.save(video_name, writer=FFMpegWriter(fps=20, bitrate=1800))
    plt.close(figure)

    display(Video(video_name, embed=True))


# Running this cell/file starts the complete simulation.
run_simulation()
