from typing import Dict

import numpy as np

ACTUATOR_UPDATES = {
    "walker2d_uni": {
        "thigh_left_joint": np.array([0.25 * i for i in range(20)]),
        "leg_left_joint": np.array([0.25 * i for i in range(20)]),
        "foot_left_joint": np.array([0.25 * i for i in range(20)]),
    },
    "ant_uni": {
        "$ Torso_Aux 4": np.array([0.25 * i for i in range(20)]),
        "Aux 4_$ Body 13": np.array([0.25 * i for i in range(20)]),
    },
    "halfcheetah_uni": {
        "fthigh": np.array([0.25 * i for i in range(20)]),
        "fshin": np.array([0.25 * i for i in range(20)]),
        "ffoot": np.array([0.25 * i for i in range(20)]),
    },
}

GRAVITY_MULTIPLIERS = {
    "walker2d_uni": np.array([0.25 * (i + 1) for i in range(20)]),
    "ant_uni": np.array([0.25 * (i + 1) for i in range(20)]),
    "halfcheetah_uni": np.array([0.25 * (i + 1) for i in range(20)]),
}


ADAPTATION_CONSTANTS: Dict[str, dict] = {
    "gravity_multiplier": GRAVITY_MULTIPLIERS,
    "actuator_update": ACTUATOR_UPDATES,
}
