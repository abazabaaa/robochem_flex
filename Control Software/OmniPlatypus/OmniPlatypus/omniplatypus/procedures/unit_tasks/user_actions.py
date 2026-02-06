"""
File: user_actions.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: These are tasks which the platforms require a human (user) to perform. Two base types are defined: one for
    requests made by the platform and one for the responses of the user. The responses complement the physical action by
    providing the platform with the details of the implementation (e.g. which vials were refilled?).
"""

import pandas as pd


class UserActionRequest:
    """
    This is the counterpart to the UserAction class.
    Allows the platform or experiment to request certain non-automated actions to be performed by the user.
    Subclasses should be made for specific actions with specific information and handling by the platform.
    """

    description: str

    def __init__(self, description: str = ""):
        """
        Constructor.

        @param description: str = ""
            Description of what triggered the request and / or what is expected of the user.
        """
        self.description = description

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.description}"


class UserAction:
    """
    Basic implementation of the container for non-automated actions.
    These are meant to be actions performed by the user (upon request or not) for tasks that are not automated.
    Subclasses should be made for specific actions with specific information and handling by the platform.
    """

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: "


class UserSetSamplesRequest(UserActionRequest):
    """
    UserActionRequest for samples-DataFrame related issues, such as empty vials, or lack thereof.
    The user is expected to manually add/remove samples and update the samples DataFrame accordingly.
    """

    samples: pd.DataFrame
    vials: list | None

    def __init__(
        self,
        current_samples: pd.DataFrame,
        highlight_vials: list | None = None,
        description: str = "",
    ):
        """
        Constructor.

        @param current_samples: pandas.DataFrame
            The current set of vials and samples.
        @param highlight_vials: list | None = None
            Optional list of vial_ids for vials which should be highlighted to the user
            (e.g.: because there is an issue involving them specifically).
        @param description:
            Description of what triggered the request and / or what is expected of the user.
        """
        UserActionRequest.__init__(self, description)
        self.samples = current_samples
        self.vials = highlight_vials


class UserSetSamples(UserAction):
    """
    UserAction for updating the samples DataFrame of the platform.
    """

    samples: pd.DataFrame

    def __init__(self, new_samples: pd.DataFrame):
        """
        Update the platform samples with the contents of the provided dataframe.
        Note: the dataframe will override completely the current one, so it must include everything,
        not just the changes. It is recommended to modify the existing samples dataframe rather than building one
        from scratch.

        @param new_samples: pandas.DataFrame
            Updated samples dataframe according to what the user changed.
        """
        self.samples = new_samples
