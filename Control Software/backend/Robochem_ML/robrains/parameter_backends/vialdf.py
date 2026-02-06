"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import hashlib
import string
import time

import pandas as pd

from robrains.base_classes import BaseLoggedClass
from robrains.parameter_backends.stocksolutiondf import StockSolutionDF


class VialDF(BaseLoggedClass):
    """Vial DF, used to save the vials in the platform, it will store the vial ID, the name, the stock ID, the volume,
    the handler, the holder, the position and the type of the sample.
    at init the user needs to specify the number of vials, the stock_df and the handler_info
    usage:
    at init: df = VialDF(number_vials=3, stock_df=stock_df, handler_info=handler_info)
    to update: df.update_df(new_df)


    Members:
        - number_vials: int: the number of vials
        - stock_list: list: the list of stocks
        - handler_info: dict: the handler info
        - available_handlers: list: the list of available handlers
        - available_holders: list: the list of available holders
        - available_positions: list: the list of available positions
        - df: pd.DataFrame: the dataframe with the vials with the columns:
            VialID (str), VialName(str), StockID(str), Volume(float), Sampler(str),
            Holder(str), Position(str), Type(str)

    Methods:
        - update_df(new_df: pd.DataFrame): updates the df with new values
        - _validate_updating_df(df: pd.DataFrame): validates the new df
        - __bool__(): returns false if 1) VialIDs are not unique, 2) Stock solutions are not in the stock list,
        3) Samplers are not in the available handlers list, 4) Holders are not in the handler info list,
        5) Positions are not in the holder info list, 6) Types are not in the sample type list, 7) Volumes are negative
        8) The Sampler+Holder+Position combination is not unique
        -from_json:

    available_ sample types:
        Stock: a reagent that can be selected for the experiment
        Solvent: a solvent that can be selected for the experiment
        Sample: empty vial for the handler to store a reaction
        Waste: vial for waste disposal
        Cleaning: vial full of cleaning solution for the needle
        Gas: Empty vial connected to gas outlet
    """

    sample_type = [
        "Stock",
        "Solvent",
        "Sample",
        "Waste",
        "Cleaning",
        "Gas",
        "Mixing",
        "CleaningAgent",
    ]
    columns = [
        "VialID",
        "VialName",
        "StockID",
        "Volume",
        "Sampler",
        "Holder",
        "Position",
        "Type",
        "Counter",
        "Viable",
    ]

    def __init__(
        self,
        number_vials: int,
        handler_info: dict,
        stock_df: StockSolutionDF = None,
        stock_list: list = None,
    ):
        """initialises the dataframe
        :param number_vials: int: the number of vials
        :param handler_info: dict: the handler info
        :param stock_df: StockSolutionDF: the stock solution dataframe
        :param stock_list: list: the list of stocks use either the stock list or the stock_df not both, preferentially
        use the stock_df
        """
        self.number_vials = number_vials
        self.get_stock_info(stock_df=stock_df, stock_list=stock_list)
        self.get_handler_info(handler_info)
        self._generate_df()
        self.log_mssg("initialized successfully.", level="ok")

    @classmethod
    def from_json(cls, json_data: dict):
        """initialises the class from a json dictionary"""
        try:
            instance = cls(
                number_vials=json_data["number_vials"],
                handler_info=json_data["handler_info"],
                stock_list=json_data["stock_list"],
            )
            if isinstance(json_data["df"], pd.DataFrame):
                df = json_data["df"]
            elif isinstance(json_data["df"], str) and json_data["df"].startswith(
                "@DataFrame"
            ):
                path_df = json_data["df"].split(":", 1)[1]
                df = pd.read_csv(path_df)
            instance.df = df
            instance.number_vials = len(df.index)
            return instance
        except Exception as e:
            cls.log_mssg(f"Error initialising VialDF from json: {e}", level="error")

    def get_stock_info(
        self,
        stock_df: StockSolutionDF,
        stock_list: list = None,
        solvent_list: list = None,
    ):
        """returns the stock info from the StockSolutionDF object
        :param stock_df: StockSolutionDF: the stock solution dataframe way to initialise from the GUI
        :param stock_list: list: the list of stocks, way to initialise from the backend(json)
        :param solvent_list: list: the list of solvents, way to initialise from the backend (json)
        """

        if stock_df is None and stock_list is not None:
            available_stocks = stock_list
            self.solvent_list = solvent_list
        elif stock_df is not None:
            available_stocks = stock_df.df["StockID"].values.tolist()
            self.solvent_list = stock_df.solvent_list
        else:
            self.log_mssg("No stock information provided", level="error")
            raise ValueError("No stock information provided")

        # add Not a Stock in the list of available stocks
        self.stock_list = available_stocks + ["Not a Stock"]

    def get_handler_info(self, handler_info: dict):
        """initialises the handler information available to the class from the platform backend
        :param handler_info: dict: the handler information from the platform backend
        example: {'handler1':{holder1: [A1, A2, A3....], holder3:[...]},'handler2: [{holderG: [A1, A2, A3....]}]}
        generated with the PlatformBackend.get_handlers method
        """
        self.handler_info = handler_info
        self.available_handlers = set()
        self.available_holders = set()
        self.available_positions = set()

        for handler, holders in self.handler_info.items():
            # Add handler to the set of available handlers
            self.available_handlers.add(handler)

            for holder, positions in holders.items():
                # Add holder to the set of available holders
                self.available_holders.add(holder)

                # Update the set of available positions with new positions
                self.available_positions.update(positions)

        # Convert sets to sorted lists to eliminate duplicates and maintain consistency
        self.available_handlers = sorted(list(self.available_handlers))
        self.available_holders = sorted(list(self.available_holders))
        self.available_positions = sorted(list(self.available_positions))

    @staticmethod
    def generate_vial_ID():
        """each vial will have a semi unique hash as an id, this is generated from the timestamp and the session state"""
        # Convert the current time and session data to a string representation
        # Use a high-precision timestamp (time in nanoseconds)
        timestamp = str(time.time_ns())

        # Get the current state of the session as a way to make our hash (can add more session details here if needed)
        session_str = str(globals())

        # Concatenate the session information and current time
        combined_info = session_str + timestamp

        # Use SHA-256 hash function from hashlib and then take the first few bytes to fit our size limit
        hash_object = hashlib.sha256(combined_info.encode())
        # We take the first 4 bytes for a base-256 to base-36 conversion to fit into 6 characters
        short_hash = hash_object.digest()[:4]

        # Convert to base-36 (0-9, a-z)
        num = int.from_bytes(short_hash, byteorder="big")
        base36 = ""
        alphabet = string.digits + string.ascii_lowercase

        while num:
            num, i = divmod(num, 36)
            base36 = alphabet[i] + base36

        # Ensure the hash is exactly 6 characters long
        if len(base36) < 6:
            base36 = base36.zfill(6)
        else:
            base36 = base36[:6]

        # sleep a few nanoseconds before returning the hash
        time.sleep(0.000000042)

        return base36

    def _generate_df(self):
        """generates the dataframe with the correct columns and default data"""

        dummy_data = {
            "VialID": [self.generate_vial_ID() for _ in range(self.number_vials)],
            "VialName": [f"Vial_{i}" for i in range(self.number_vials)],
            "StockID": ["Not a Stock" for _ in range(self.number_vials)],
            "Volume": [0.0 for _ in range(self.number_vials)],
            "Sampler": [self.available_handlers[0] for _ in range(self.number_vials)],
            "Holder": [self.available_holders[0] for _ in range(self.number_vials)],
            "Position": ["A1" for _ in range(self.number_vials)],
            "Type": ["Solvent" for _ in range(self.number_vials)],
            "Counter": [0 for _ in range(self.number_vials)],
            "Viable": [True for _ in range(self.number_vials)],
        }

        self.df = pd.DataFrame(dummy_data)

    def _validate_updating_df(self, df: pd.DataFrame):
        """the updating df has to follow the following rules:
        1) VialIDs and SampleIDs must be unique
        2) Stock solutions must be in the stock list
        3) Samplers must be in the available handlers list
        4) Holders must be in the handler info list
        5) Positions must be in the holder info list
        6) Types must be in the sample type list
        7) Volumes must be positive (or 0)
        8) The Sampler+Holder+Position combination must be unique
        9) If vial purpose is "Solvent" stock id must be in the stock_df.solvent_list

        """

        # 1) VialIDs must be unique
        if df["VialID"].nunique() != len(df):
            self.log_mssg("VialIDs must be unique.", level="warning")
            return False, "VialIDs must be unique.", "warning"

        # 2) Stock solutions must be in the stock list if the type is not solvent
        if not df.loc[df["Type"] != "Solvent", "StockID"].isin(self.stock_list).all():
            self.log_mssg(
                "All stock solutions must be in the stock list.", level="warning"
            )
            return False, "All stock solutions must be in the stock list.", "warning"

        # 3) Samplers must be in the available handlers list
        if not df["Sampler"].isin(self.available_handlers).all():
            self.log_mssg(
                "All handlers must be in the available handlers list.", level="warning"
            )
            return (
                False,
                "All handlers must be in the available handlers list.",
                "warning",
            )

        # For holders and positions, validate dynamically per row
        for _, row in df.iterrows():
            handler = row["Sampler"]
            holder = row["Holder"]
            position = row["Position"]

            # 4 & 5) Holders and Positions validation
            if handler in self.handler_info:
                if holder not in self.handler_info[handler]:
                    self.log_mssg(
                        f"Holder {holder} not found for Sampler {handler}.",
                        level="warning",
                    )
                    return (
                        False,
                        f"Holder {holder} not found for Sampler {handler}.",
                        "warning",
                    )
                if position not in self.handler_info[handler][holder]:
                    self.log_mssg(
                        f"Position {position} not valid for Holder {holder} in Sampler {handler}.",
                        level="warning",
                    )
                    return (
                        False,
                        f"Position {position} not valid for Holder {holder} in Sampler {handler}.",
                        "warning",
                    )

            else:
                self.log_mssg(f"Sampler {handler} is not recognized.", level="warning")
                return False, f"Sampler {handler} is not recognized.", "warning"

        # 6) Types must be in the sample type list
        if not df["Type"].isin(self.sample_type).all():
            self.log_mssg("All types must be in the sample type list.", level="warning")
            return False, "All types must be in the sample type list.", "warning"

        # 7) Volumes must be positive (or 0)
        if (df["Volume"] < 0.0).any():
            self.log_mssg("Volumes must be positive or zero.", level="warning")
            return False, "Volumes must be positive or zero.", "warning"

        # 8) The Sampler+Holder+Position combination must be unique
        if df[["Sampler", "Holder", "Position"]].duplicated().any():
            self.log_mssg(
                "The Sampler+Holder+Position combination must be unique.",
                level="warning",
            )
            return (
                False,
                "The Sampler+Holder+Position combination must be unique.",
                "warning",
            )
        # 9) If vial type is "Solvent" stock id must be in the self.solvent_list
        if not df.loc[df["Type"] == "Solvent", "StockID"].isin(self.solvent_list).all():
            self.log_mssg(
                "All stock solutions for solvent vials must be in the solvent list.",
                level="warning",
            )
            return (
                False,
                "All stock solutions for solvent vials must be in the solvent list.",
                "warning",
            )

        # 10) Septum counters must be positive (or 0)
        if (df["Counter"] < 0).any():
            self.log_mssg("Counters must be positive or zero.", level="warning")
            return False, "Counters must be positive or zero.", "warning"

        # If all checks pass
        self.log_mssg("updating DataFrame is valid.", level="ok")
        return True, "updating DataFrame is valid.", "ok"

    def update_from_df(self, df):
        """updates the DF from another df which has the same format as the one in the class
        :param df: pd.DataFrame: the dataframe to update from
        """
        # Check if the new DataFrame has the right number of columns
        if df.empty:
            self.log_mssg("New DataFrame is empty, nothing to update.", level="warning")
            return {
                "message": "New DataFrame is empty, nothing to update.",
                "level": "warning",
            }

        if df.shape[1] != len(self.columns):
            self.log_mssg(
                "New DataFrame has the wrong number of columns.", level="warning"
            )
            return {
                "message": "New DataFrame has the wrong number of columns.",
                "level": "warning",
            }

        test, message, level = self._validate_updating_df(df)

        if not test:
            self.log_mssg("Validation failed, not updating.", level="warning")
            return {"message": message, "level": level}

        self.df = df

        self.log_mssg("DF updated successfully.", level="ok")

    def update_df(self, new_number_of_samples: int):
        """adds or removes rows when the number of samples changes
        :param new_number_of_samples: int: the new number of samples
        """

        current_samples = len(self.df)
        if new_number_of_samples > current_samples:
            new_rows = pd.DataFrame(
                {
                    "VialID": [
                        self.generate_vial_ID()
                        for _ in range(current_samples, new_number_of_samples)
                    ],
                    "VialName": [
                        f"Vial_{i}"
                        for i in range(current_samples, new_number_of_samples)
                    ],
                    "StockID": [
                        "Not a Stock"
                        for _ in range(current_samples, new_number_of_samples)
                    ],
                    "Volume": [
                        0.0 for _ in range(current_samples, new_number_of_samples)
                    ],
                    "Sampler": [
                        self.available_handlers[0]
                        for _ in range(current_samples, new_number_of_samples)
                    ],
                    "Holder": [
                        self.available_holders[0]
                        for _ in range(current_samples, new_number_of_samples)
                    ],
                    "Position": [
                        "A1" for _ in range(current_samples, new_number_of_samples)
                    ],
                    "Type": [
                        "Solvent" for _ in range(current_samples, new_number_of_samples)
                    ],
                    "Counter": [
                        0 for _ in range(current_samples, new_number_of_samples)
                    ],
                    "Viable": [
                        True for _ in range(current_samples, new_number_of_samples)
                    ],
                }
            )
            self.df = pd.concat([self.df, new_rows], ignore_index=True)
        elif new_number_of_samples < current_samples:
            self.df = self.df.iloc[:new_number_of_samples]

        self.number_vials = new_number_of_samples
        self.log_mssg("DF length updated successfully.", level="ok")

    def update_from_platform(self, platformDF: pd.DataFrame):
        """During a campaign usign the platform it may be necessary to update the vial dataframe with informations
        such as the remaining vial volume or other. This is information only the plaform can provide.
        """

        self.df.set_index("VialID", inplace=True)
        self.df.update(platformDF)
        self.df.reset_index(inplace=True)

    def __bool__(self):
        """returns false if the dataframe is empty or if the vial IDs are not unique"""
        # Check if the DataFrame is empty
        if self.df.empty:
            return False

        # Check if Vial IDs are unique
        if self.df["VialID"].nunique() != len(self.df):
            return False

        # If none of the conditions are met, the instance is considered True
        return True
