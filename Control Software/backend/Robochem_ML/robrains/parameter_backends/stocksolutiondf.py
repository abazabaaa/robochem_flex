"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import pandas as pd

from robrains.base_classes import BaseLoggedClass


class StockSolutionDF(BaseLoggedClass):
    """Stock Solution DF. Here the user can input the stock solutions they would like to have in their experiment.
    At init the user needs to specify the number of solutions the chemicals and solvents available,
    when updating these values can be changed depending on the session state.
    usage:
    at init: df = StockSolutionDF(number_stock_solutions=3, chemicals_list=['H2O', 'NaOH'], solvent_list=['H2O'])
       to update: df.update_df(new_number_stock_solutions=5, new_chemicals_list=['H2O', 'NaOH', 'HCl'], new_solvent_list=['H2O'])

    Members:
       - number_stock_solutions: int: the number of stock solutions
       - chemicals_list: list: the list of chemicals
       - solvent_list: list: the list of solvents
       - df: pd.DataFrame: the dataframe with the stock solutions with the columns:
           StockID (str), Solvent(str), Conc_chemical1(float), Conc_chemical2(float), ...
    Methods:
       - update_df(new_number_stock_solutions: int, new_chemicals_list: list, new_solvent_list: list): updates the df with new values
       - update_from_df(new_df: pd.DataFrame): updates the df from a new df with the same format
       - __bool__(): returns false if the df is empty, stock IDs are not right,
       or there exists a stock solution with all 0s or negative values

    """

    num_call_names = 0

    def __init__(self, chemicals_list: list, solvent_list: list):
        """initialises the dataframe
        :param chemicals_list: list: the list of chemicals i.e ['jerry', 'tom']
        :param solvent_list: list: the list of solvents i.e. ['H2O', 'DMSO']
        """

        self.chemicals_list = chemicals_list
        self.solvent_list = solvent_list
        self._generate_df()
        self.log_mssg("initialized successfully.", level="ok")

    @classmethod
    def from_json(cls, json_data: dict):
        """initialises the class from a json dictionary"""
        try:
            instance = cls(
                chemicals_list=json_data["chemicals_list"],
                solvent_list=json_data["solvent_list"],
            )

            if isinstance(json_data["df"], pd.DataFrame):
                df = json_data["df"]
            elif isinstance(json_data["df"], str) and json_data["df"].startswith(
                "@DataFrame"
            ):
                path_df = json_data["df"].split(":", 1)[1]
                df = pd.read_csv(path_df)

            instance.df = df
            return instance
        except Exception as e:
            cls.log_mssg(
                f"Error initialising StockSolutionDF from json: {e}", level="error"
            )

    def generate_name(self):
        self.num_call_names += 1
        return f"Stock_{self.num_call_names+len(self.df)}"

    def _generate_df(self):
        """sets up the dataframe the way wee need it to be set up:
        StockID, Solvent, Conc_chemical1, Conc_chemical2, ..."""
        columns = ["StockID", "Solvent"]
        for chemical in self.chemicals_list:
            columns.append(f"Conc_{chemical}")
        dummy_data = {col: 0.0 for col in columns if col != "StockID"}
        dummy_data["StockID"] = f"Stock_{0}"

        dummy_data["Solvent"] = [
            self.solvent_list[0] if len(self.solvent_list) > 0 else "Solvent?"
        ]
        self.df = pd.DataFrame(dummy_data, columns=columns, index=[0])
        self.log_mssg("Chemicals columns updated.", level="ok")

    def update_df(
        self,
        new_chemicals_list: list,
        new_solvent_list: list,
    ):
        try:
            # Similar update logic as before, with logging at key steps
            if set(new_chemicals_list) == set(self.chemicals_list) and set(
                new_solvent_list
            ) == set(self.solvent_list):
                self.log_mssg(
                    "No changes to make, new and old lists are the same.",
                    level="warning",
                )
                return

            # Update stock IDs in case the number of solutions has changed
            # self._update_stock_IDs()

            # Update the list of chemicals
            current_chemicals = [
                col.replace("Conc_", "")
                for col in self.df.columns
                if col.startswith("Conc_")
            ]
            added_chemicals = set(new_chemicals_list) - set(current_chemicals)
            removed_chemicals = set(current_chemicals) - set(new_chemicals_list)

            # Add new chemicals
            for chemical in added_chemicals:
                self.df[f"Conc_{chemical}"] = 0.0

            # Remove old chemicals
            for chemical in removed_chemicals:
                self.df.drop(columns=[f"Conc_{chemical}"], inplace=True)

            # check if all the solvents are in the new solvent list, otherwise replace the solvent with the first one in the list
            self.df["Solvent"] = self.df["Solvent"].apply(
                lambda x: (
                    x
                    if x in new_solvent_list
                    else (
                        self.solvent_list[0]
                        if len(self.solvent_list) > 0
                        else "Solvent?"
                    )
                )
            )
            # Update internal state
            self.chemicals_list = new_chemicals_list
            self.solvent_list = new_solvent_list
            # make sure that nan values are replaced with 0s
            self.df.fillna(0, inplace=True)
            # For demonstration, log a success message at the end
            self.num_call_names = 0
            self.log_mssg("StockSolutionDF updated successfully.", level="ok")
        except Exception as e:
            self.log_mssg(f"Error during update: {e}", level="error")
            raise e

    def update_from_df(self, new_df: pd.DataFrame):
        """checks that the new df is in the right format and updates the current df"""
        # Check if the new DataFrame has the right number of columns
        if new_df.empty:
            self.log_mssg("New DataFrame is empty, nothing to update.", level="warning")
            return
        if new_df.shape[1] != len(self.chemicals_list) + 2:
            self.log_mssg(
                "New DataFrame has the wrong number of columns.", level="error"
            )
            raise ValueError("New DataFrame has the wrong number of columns.")

        # Check if the Stock IDs are all filled and unique
        if not new_df["StockID"].notnull().all():
            self.log_mssg("Stock IDs are not all filled.", level="error")
            raise ValueError("Stock IDs are not all filled.")

        if not new_df["StockID"].is_unique:
            self.log_mssg("Stock IDs are not all unique.", level="error")
            raise ValueError("Stock IDs are not all unique.")

        # Checks that the chemical columns are in the right format and match the ones in the df
        concentration_columns = [
            col for col in new_df.columns if col.startswith("Conc_")
        ]
        if not all(
            [
                col.replace("Conc_", "") in self.chemicals_list
                for col in concentration_columns
            ]
        ):
            self.log_mssg(
                "Chemical columns are not correctly formatted or do not match the list of chemicals.",
                level="error",
            )
            raise ValueError(
                "Chemical columns are not correctly formatted or do not match the list of chemicals."
            )

        # updates all rows of the conc columns with the new values only if the stock_ID matches and the new row is not all 0s
        self.df = new_df

        self.log_mssg("StockSolutionDF updated successfully from df", level="ok")

    def __bool__(self):
        """returns false if 1) the dataframe is empty, 2) stock IDs are not right, 3) there exists a stock solution with
        all 0s or negative values"""
        # Check if the DataFrame is empty
        if self.df.empty:
            return False

        # Check if Stock IDs are all filled and unique
        if not self.df["StockID"].notnull().all():
            return False
        if not self.df["StockID"].is_unique:
            return False
        # Check for rows with all 0s or any negative values in chemical concentration columns
        concentration_columns = [
            col for col in self.df.columns if col.startswith("Conc_")
        ]
        for _, row in self.df.iterrows():
            if all(row[col] <= 0 for col in concentration_columns):
                return False

        # If none of the conditions are met, the instance is considered True
        return True
