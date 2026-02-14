"""
Dataloaders package for the S7 model.

Provides dataset factory functions for all supported tasks, accessible
through the ``Datasets`` dictionary keyed by task name.
"""

from .collate import Data
from .event_streams import (
    create_events_shd_classification_dataset,
    create_events_ssc_classification_dataset,
    create_events_dvs_gesture_classification_dataset,
)
from .lra import (
    create_listops_tonic_classification_dataset,
    create_imdb_tonic_classification_dataset,
    create_retrieval_tonic_classification_dataset,
    create_image_tonic_classification_dataset,
    create_pathfinder_tonic_classification_dataset,
    create_pathx_tonic_classification_dataset,
)
from .timeseries import (
    create_eigenworms_tonic_classification_dataset,
    create_person_activity_tonic_classification_dataset,
    create_walker_tonic_classification_dataset,
)
from .text import (
    create_ptb_tonic_classification_dataset,
    create_wikitext2_tonic_classification_dataset,
    create_wikitext103_tonic_classification_dataset,
)

Datasets = {
    "shd-classification": create_events_shd_classification_dataset,
    "ssc-classification": create_events_ssc_classification_dataset,
    "dvs-gesture-classification": create_events_dvs_gesture_classification_dataset,
    "listops-classification": create_listops_tonic_classification_dataset,
    "text-classification": create_imdb_tonic_classification_dataset,
    "retrieval-classification": create_retrieval_tonic_classification_dataset,
    "image-classification": create_image_tonic_classification_dataset,
    "pathfinder-classification": create_pathfinder_tonic_classification_dataset,
    "pathx-classification": create_pathx_tonic_classification_dataset,
    "eigenworms-classification": create_eigenworms_tonic_classification_dataset,
    "personactivity-classification": create_person_activity_tonic_classification_dataset,
    "walker-classification": create_walker_tonic_classification_dataset,
    "ptb-classification": create_ptb_tonic_classification_dataset,
    "wikitext2-classification": create_wikitext2_tonic_classification_dataset,
    "wikitext103-classification": create_wikitext103_tonic_classification_dataset,
}
