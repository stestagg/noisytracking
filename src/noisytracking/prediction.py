
from noisytracking.parameter import Pose
from typing import Optional, Any


class Motion(Pose):
    def __init__(
        self,
        name: Optional[str] = None,
        units: Optional[Any] = None,
        expected_change: Optional[int] = None,
    ) -> None:
        super().__init__(name=name, units=units)
        self.expected_change = expected_change
