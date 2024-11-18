from typing import List, Tuple, Union

import numpy as np

ACTION_TYPE = int
REWARD_TYPE = float
SEQUENCE_TYPE = List[Union[np.ndarray, ACTION_TYPE]]
PHI_TYPE = np.ndarray
TRANSITION_TYPE = Tuple[PHI_TYPE, ACTION_TYPE, REWARD_TYPE, PHI_TYPE]
