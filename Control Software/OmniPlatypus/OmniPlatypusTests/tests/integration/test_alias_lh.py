"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import unittest

import pandas as pd
from omniplatypus.devices.platform import Platform
from omniplatypus.devices.knauer.autosampler_alias import (
    AutosamplerAlias,
    AliasPosition,
)
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    GenerateSampleDataframe,
    VialRecipeComponent,
)
from omniplatypus.procedures.unit_tasks.sampling.alias_control import *


class AliasSamplerTest(unittest.TestCase):
    platform: Platform = None
    sampler: AutosamplerAlias = None
    samples: pd.DataFrame = None

    @classmethod
    def setUpClass(cls):
        test_platform = Platform()
        test_platform.build(
            platform_name="Sparkie",
            devices=[
                "Sampler_cnc",
            ],
            open_gui=False,
        )
        cls.platform = test_platform
        cls.sampler = test_platform["Sampler_cnc"]
        cls.samples = pd.DataFrame(
            {
                GenerateSampleDataframe.index_name: ["S1", "S2", "C1"],
                "Conc_A": [1, 1, 0.0],
                "Volume": [900.0, 1000.0, 100.0],
                "Type": ["Stock", "Stock", "Sample"],
                "Sampler": ["Sampler_cnc", "Sampler_cnc", "Sampler_cnc"],
                "Holder": ["holder_left", "holder_left", "holder_left"],
                "Position": ["A1", "B2", "D3"],
                "Add": [0.0, 0.0, 0.0],
            }
        )
        cls.samples.set_index(
            GenerateSampleDataframe.index_name, inplace=True, verify_integrity=True
        )
        GenerateSampleDataframe.run(platform=cls.platform, samples=cls.samples)
        cls.samples = cls.platform.samples

    @classmethod
    def tearDownClass(cls):
        cls.platform.clear()

    def alias_fill_system(self):
        """
        pushes to waste for 20 times
        """
        AliasPrimeWaste.run(
            platform=self.platform, sampler=self.sampler, number_of_primes=20
        )

    @unittest.skip("Already done.")
    def test_fill(self):
        self.alias_fill_system()

    # @unittest.skip("Already done.")
    def test_move(self):
        # make some position objects:
        position1 = AliasPosition(X=1, Y=1, T="L")
        position2 = AliasPosition(X=2, Y=2, T="L")
        position3 = AliasPosition(X=4, Y=3, T="L")

        # move to position 1
        MoveAliasNeedle.run(sampler=self.sampler, destination=position1)
        # move to position 2
        MoveAliasNeedle.run(sampler=self.sampler, destination=position2)
        # move to position 3
        MoveAliasNeedle.run(sampler=self.sampler, destination=position3)

        home_position = AliasPosition(special="HOME_ALL")
        MoveAliasNeedle.run(sampler=self.sampler, destination=home_position)

    @unittest.skip("Already done.")
    def test_move_to_sample(self):
        for index, row in self.samples.iterrows():
            position = AliasPosition(X=row["X"], Y=row["Y"], T="L")
            MoveAliasNeedle.run(sampler=self.sampler, destination=position)

        home_position = AliasPosition(special="HOME_ALL")
        MoveAliasNeedle.run(sampler=self.sampler, destination=home_position)

    @unittest.skip("Already done.")
    def test_make_reaction(self):
        recipe = [VialRecipeComponent("S1", 100.0), VialRecipeComponent("S2", 100.0)]
        dest_position = "C1"
        PrepareReactionVial.run(
            platform=self.platform,
            samples=self.samples,
            recipe=recipe,
            destination=dest_position,
            clean_needle=True,
        )

    @unittest.skip("Already done.")
    def test_make_slug(self):
        AliasLoadSlug.run(
            platform=self.platform,
            samples=self.samples,
            sample_name="S1",
            slug_volume=500.0,
        )

    def test_calibration(self):
        """test method to calibrate the autosampler:
        select write down the stock_1 position and molarity (you should calculate that by hand), and select the
        required dilution concentrations. choose the position of the dilution vials and (1 ml volumes), the position of
        the 2 ml stock (highest conc) and the solvent vial positions.

        dilutions 3x 0.5, 2 , 5, 7, 12, 18
        initial stock 20mM
        """
        new_sample_df = pd.read_csv("OmniPlatypusTests/tests/integration/SamplesDF.csv")

        new_sample_df.set_index("SampleID", inplace=True, verify_integrity=True)
        new_sample_df = GenerateSampleDataframe.run(
            platform=self.platform, samples=new_sample_df
        )

        # Convert the recipe to a DataFrame
        dilution_df = pd.read_csv("OmniPlatypusTests/tests/integration/DilutionDF.csv")

        print(dilution_df)

        usr_ok = input("Is the recipe ok? (y/n): ")
        if not usr_ok.lower() in ["y", "yes"]:
            raise ValueError("Recipe not ok.")

        AliasPrimeNeedle.run(
            autosampler=self.sampler, volume_to_waste=2000, bubble_volume=50
        )

        # Perform the dilutions
        for index, row in dilution_df.iterrows():
            # position objects for the stock and solvent
            stock_pos = row["StockID"]
            solvent_pos = row["SolventID"]
            target_pos = row["DilutionID"]
            recipe = [
                VialRecipeComponent(solvent_pos, row["SolventVol"]),
                VialRecipeComponent(stock_pos, row["StockVol"]),
            ]

            PrepareReactionVial.run(
                platform=self.platform,
                samples=new_sample_df,
                recipe=recipe,
                destination=target_pos,
                clean_needle=True,
            )

            CleanNeedle.run(platform=self.platform, sampler=self.sampler)

    def test_volume_calibration(self):
        """test method to calibrate the autosampler:
        select write down the required volumes for each vial. choose the position of
        the dilution vials and the solvent vial positions.

        """
        new_sample_df = pd.DataFrame(
            {
                "SampleID": [
                    "solvent_1",
                    "solvent_2",
                    "solvent_3",
                    "solvent_4",
                    "solvent_5",
                    "solvent_6",
                    "solvent_7",
                    "solvent_8",
                    "solvent_9",
                    "dilution_1000_1",
                    "dilution_1000_2",
                    "dilution_1000_3",
                    "dilution_0750_1",
                    "dilution_0750_2",
                    "dilution_0750_3",
                    "dilution_0500_1",
                    "dilution_0500_2",
                    "dilution_0500_3",
                    "dilution_0250_1",
                    "dilution_0250_2",
                    "dilution_0250_3",
                    "dilution_0100_1",
                    "dilution_0100_2",
                    "dilution_0100_3",
                    "dilution_0050_1",
                    "dilution_0050_2",
                    "dilution_0050_3",
                    "dilution_0020_1",
                    "dilution_0020_2",
                    "dilution_0020_3",
                    "dilution_0010_1",
                    "dilution_0010_2",
                    "dilution_0010_3",
                    "dilution_0001_1",
                    "dilution_0001_2",
                    "dilution_0001_3",
                ],
                "Volume": [
                    1000,
                    1500,
                    1500,
                    1500,
                    1500,
                    1500,
                    1500,
                    1500,
                    1500,
                    1000,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                "Type": [
                    "Solvent",
                    "Solvent",
                    "Solvent",
                    "Solvent",
                    "Solvent",
                    "Solvent",
                    "Solvent",
                    "Solvent",
                    "Solvent",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                    "Sample",
                ],
                "Sampler": [
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                ],
                "Holder": [
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                    "holder_left",
                ],
                "Position": [
                    "A1",
                    "B1",
                    "C1",
                    "D1",
                    "E1",
                    "F1",
                    "A2",
                    "B2",
                    "C2",
                    "D2",
                    "E2",
                    "F2",
                    "A3",
                    "B3",
                    "C3",
                    "D3",
                    "E3",
                    "F3",
                    "A4",
                    "B4",
                    "C4",
                    "D4",
                    "E4",
                    "F4",
                    "A5",
                    "B5",
                    "C5",
                    "D5",
                    "E5",
                    "F5",
                    "A6",
                    "B6",
                    "C6",
                    "D6",
                    "E6",
                    "F6",
                ],
                "Add": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            }
        )

        new_sample_df.set_index("SampleID", inplace=True, verify_integrity=True)
        new_sample_df = GenerateSampleDataframe.run(
            platform=self.platform, samples=new_sample_df
        )

        # Define the required volumes for each dilution vial
        required_volumes = {
            # "D2":1000,
            "E2": 1000,
            "F2": 1000,
            "A3": 750,
            "B3": 750,
            "C3": 750,
            "D3": 500,
            "E3": 500,
            "F3": 500,
            "A4": 250,
            "B4": 250,
            "C4": 250,
            "D4": 100,
            "E4": 100,
            "F4": 100,
            "A5": 50,
            "B5": 50,
            "C5": 50,
            "D5": 20,
            "E5": 20,
            "F5": 20,
            "A6": 10,
            "B6": 10,
            "C6": 10,
            "D6": 1,
            "E6": 1,
            "F6": 1,
        }  # ul
        solvent_positions = {
            "A1": 600,
            "B1": 1500,
            "C1": 1500,
            "D1": 1500,
            "E1": 1500,
            "F1": 1500,
            "A2": 1500,
            "B2": 1500,
            "C2": 1500,
        }  # ul

        # Prepare a DataFrame to store the volume calibration recipe
        volume_recipe = []

        # Perform volume calibration
        for vial, required_volume in required_volumes.items():
            remaining_volume = required_volume

            while remaining_volume > 0:
                # Find an available solvent vial with enough volume
                solvent_vial = None
                for pos, vol in solvent_positions.items():
                    if vol >= remaining_volume:
                        solvent_vial = pos
                        volume_taken = remaining_volume
                        solvent_positions[pos] -= remaining_volume
                        remaining_volume = 0
                        break
                    elif vol > 0:
                        solvent_vial = pos
                        volume_taken = vol
                        solvent_positions[pos] = 0
                        remaining_volume -= vol

                if solvent_vial is None:
                    raise ValueError(
                        f"Not enough solvent available for volume calibration in {vial}."
                    )

                    # Record the volume calibration step
                dilution_vial_sample_id = new_sample_df[
                    new_sample_df["Position"] == vial
                ].index[0]
                solvent_vial_sample_id = new_sample_df[
                    new_sample_df["Position"] == solvent_vial
                ].index[0]
                volume_recipe.append(
                    {
                        "Dilution Vial": vial,
                        "Dilution Vial SampleID": dilution_vial_sample_id,
                        "Solvent Volume (ul)": volume_taken,
                        "Solvent Position": solvent_vial,
                        "Solvent Vial SampleID": solvent_vial_sample_id,
                        "Final Volume (ul)": required_volume - remaining_volume,
                    }
                )

        # Convert the recipe to a DataFrame
        volume_df = pd.DataFrame(volume_recipe)

        print(volume_df)

        usr_ok = input("Is the recipe ok? (y/n): ")
        if not usr_ok.lower() in ["y", "yes"]:
            raise ValueError("Recipe not ok.")
        # CleanNeedle.run(platform=self.platform, sampler=self.sampler)
        # Perform the volume calibration
        for index, row in volume_df.iterrows():
            solvent_pos = row["Solvent Vial SampleID"]
            target_pos = row["Dilution Vial SampleID"]
            recipe = [VialRecipeComponent(solvent_pos, row["Solvent Volume (ul)"])]

            PrepareReactionVial.run(
                platform=self.platform,
                samples=new_sample_df,
                recipe=recipe,
                destination=target_pos,
                clean_needle=False,
            )

            # CleanNeedle.run(sampler=self.sampler)

    def test_wasting(self):
        AliasPrimeNeedle.run(
            autosampler=self.sampler, volume_to_waste=3400, bubble_volume=50
        )

    # @unittest.skip("Already done.")
    # def test_fill_samples(self):
    #     # user = Logger.input("fill all with 500.0 (y/n)?")
    #     user = "y"
    #     fixed = user == "y"
    #     ctr = 0
    #     vials = self.samples[self.samples["Type"] == "Sample"]
    #     for index, row in vials.iterrows():
    #         if ctr % 6 == 0:
    #             FillPump.run(pump=self.sampler_pump)
    #         ctr += 1
    #         if fixed:
    #             vol = 500.0
    #         else:
    #             vol = row["Add"]
    #         PumpSample.run(
    #             platform=self.platform,
    #             samples=self.samples,
    #             sample_id=index,
    #             volume=vol,
    #             needle_position="top",
    #         )
    #         if fixed:
    #             FillPump.run(pump=self.sampler_pump)
    #     self.sampler["home"] = "RUN"
    # @unittest.skip("Already done.")
    # def test_slug(self):
    #     yesno = input("flush (y/n)?")
    #     if yesno.lower() in ("y", "yes"):
    #         self.prime()
    #     yesno = input("inject (y/n)?")
    #     inject = yesno.lower() in ("y", "yes")
    #
    #     ctr = 0
    #     vials = self.samples[self.samples["Type"] == "Sample"]
    #     for index, row in vials.iterrows():
    #         ctr += 1
    #         if ctr >= 7:
    #             break
    #         if ctr == 1:
    #             continue
    #         analyte_volume = row["Add"]
    #         slug_volume = 350.0
    #         bubble_volume = 100.0
    #         extra_push = 50.0
    #         print(f"sample {ctr}")
    #         print("cleaning")
    #         FillPump.run(pump=self.sampler_pump)
    #         self.flush_slug(fill=False)
    #         PumpVolume.run(pump=self.sampler_pump, volume=-100.0, reverse=True)
    #         print("make slug")
    #         recipe = [
    #             ("Solvent", slug_volume - analyte_volume),
    #             ("Analyte", analyte_volume),
    #         ]
    #         PrepareReactionSlug.run(
    #             platform=self.platform,
    #             samples=self.samples,
    #             recipe=recipe,
    #             sampling_flowrate=1.0,
    #             bubble_volume=bubble_volume,
    #             bubble_flowrate=0.5,
    #         )
    #         print("slug out")
    #         if inject:
    #             Inject.run(
    #                 platform=self.platform,
    #                 sampler=self.sampler,
    #                 sampler_pump=self.sampler_pump,
    #                 injection_port_name="injection_flow",
    #                 volume=slug_volume + extra_push,
    #                 flowrate=1.0,
    #                 retract=False,
    #                 add_gas_delimiters=True,
    #                 main_pump=self.main_pump,
    #                 gas_valve=self.n2_valve,
    #                 phase_sensor=self.phase_sensor,
    #             )
    #             # Pretend we go through a reaction
    #             PumpVolume.run(pump=self.main_pump, volume=4000.0, flowrate=5.0)
    #             # Now do some cleanup
    #             Inject.run(
    #                 platform=self.platform,
    #                 sampler=self.sampler,
    #                 sampler_pump=self.sampler_pump,
    #                 injection_port_name="injection_flow",
    #                 volume=2000.0,
    #                 flowrate=4.0,
    #                 retract=True,
    #             )
    #             FillPump.run(pump=self.sampler_pump, fill=False)
    #             PumpSample.run(
    #                 platform=self.platform,
    #                 samples=self.samples,
    #                 sample_id="N2",
    #                 volume=-500.0,
    #                 flowrate=2.0,
    #                 needle_position="top",
    #                 pause_after=2.0,
    #                 update_volume=False,
    #             )
    #             FillPump.run(pump=self.sampler_pump, fill=True)
    #             Inject.run(
    #                 platform=self.platform,
    #                 sampler=self.sampler,
    #                 sampler_pump=self.sampler_pump,
    #                 injection_port_name="injection_flow",
    #                 volume=300.0,
    #                 flowrate=2.0,
    #                 retract=True,
    #             )
    #             PumpVolume.run(pump=self.main_pump, volume=4000.0, flowrate=5.0)
    #         else:
    #             PumpSample.run(
    #                 platform=self.platform,
    #                 samples=self.samples,
    #                 sample_id=index,
    #                 volume=slug_volume + extra_push,
    #                 needle_position="bottom",
    #                 pause_after=1.0,
    #             )
    #     print("done")
    #     FillPump.run(pump=self.sampler_pump)
    #     self.sampler["home"] = "RUN"


if __name__ == "__main__":
    unittest.main()
