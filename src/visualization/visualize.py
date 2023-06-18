import wandb
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.utils.plotting import plot_2d_map_elites_repertoire


def plot_2d_map(
    repertoire: MapElitesRepertoire,
    min_bd: float,
    max_bd: float,
    **kwargs,
):
    """Plot a 2D MAP-Elites repertoire.

    Args:
        repertoire (MapElitesRepertoire): MAP-Elites repertoire
        min_bd (float): minimum value of the behavioral descriptor
        max_bd (float): maximum value of the behavioral descriptor
    """
    fig, _ = plot_2d_map_elites_repertoire(
        centroids=repertoire.centroids,
        repertoire_fitnesses=repertoire.fitnesses,
        minval=min_bd,
        maxval=max_bd,
    )
    wandb.log({"2d_map": wandb.Image(fig)})
