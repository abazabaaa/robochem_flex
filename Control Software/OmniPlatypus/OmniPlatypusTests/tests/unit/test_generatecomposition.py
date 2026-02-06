"""
File: test_generatecomposition.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for GenerateComposition unit task
"""

import unittest
import pandas as pd

from omniplatypus.devices.platform import Platform
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    DataFrameTask,
    RecipeComponent,
    VialRecipeComponent,
    GenerateComposition,
    OrderRecipe,
    NoSuitableVialError,
    RecipeError,
    OrderRecipe,
)


class TestComposition(unittest.TestCase):
    def setUp(self):
        self.samples = pd.DataFrame(
            {
                DataFrameTask.index_name: [
                    "N2",
                    "waste",
                    "vial_solvent",
                    "vial_solvent_fuller",
                    "vial_vinegar",
                    "vial_vinegar_fuller",
                    "vial_salt",
                    "vial_salt_fuller",
                    "vial_salt_and_vinegar",
                    "vial_salt_and_vinegar_high",
                ],
                DataFrameTask.vial_name: [
                    "N2",
                    "waste",
                    "vial_solvent",
                    "vial_solvent_fuller",
                    "vial_vinegar",
                    "vial_vinegar_fuller",
                    "vial_salt",
                    "vial_salt_fuller",
                    "vial_salt_and_vinegar",
                    "vial_salt_and_vinegar_high",
                ],
                "Conc_vinegar": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.5, 20.0],
                "Conc_salt": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 20.0],
                "Volume": [
                    0.0,
                    0.0,
                    10000.0,
                    11000.0,
                    10000.0,
                    11000.0,
                    10000.0,
                    11000.0,
                    10000.0,
                    10000.0,
                ],
                "Volume_min": [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ],
                "Type": [
                    "Gas",
                    "Waste",
                    "Solvent",
                    "Solvent",
                    "Stock",
                    "Stock",
                    "Stock",
                    "Stock",
                    "Stock",
                    "Stock",
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
                ],
                "Viable": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                    False,
                ],
            }
        )
        self.samples.set_index(
            DataFrameTask.index_name, inplace=True, verify_integrity=True
        )
        self.platform = Platform()
        self.platform.samples = self.samples

    def tearDown(self):
        pass

    def assertEqualRecipe(self, expected, actual, message):
        success = False
        if len(actual) == len(expected):
            for i in range(len(expected)):
                success = expected[i] == actual[i]
                if not success:
                    break
        self.assertTrue(success, message)
        return success

    def try_composition(
        self, experiment_recipe, expected_result, order=None, slug_volume=1000.0
    ):
        print("Samples:")
        print(self.samples.to_string())
        print("Desired composition:")
        print(experiment_recipe)
        result, actual = GenerateComposition.run(
            platform=self.platform,
            experiment_recipe=experiment_recipe,
            slug_volume=slug_volume,
            sampler_name="Sampler_cnc",
        )
        print("Actual:")
        print(actual)
        if order is not None:
            result = OrderRecipe.run(result, method=order)
        print("Recipe:")
        print(result)
        if expected_result is not None:
            self.assertEqualRecipe(
                expected_result, result, "Test did not give the expected result."
            )
        print("")
        return result

    def test_one_component(self):
        experiment_recipe = [RecipeComponent("vinegar", 0.5)]
        expected_result = [
            VialRecipeComponent("vial_vinegar", 500.0),
            VialRecipeComponent("vial_solvent", 500.0, is_solvent=True),
        ]
        self.try_composition(experiment_recipe, expected_result)

    def test_two_components(self):
        experiment_recipe = [
            RecipeComponent("vinegar", 0.25),
            RecipeComponent("salt", 0.25),
        ]
        expected_result = [
            VialRecipeComponent("vial_vinegar", 250.0),
            VialRecipeComponent("vial_salt", 250.0),
            VialRecipeComponent("vial_solvent", 500.0, is_solvent=True),
        ]
        self.try_composition(experiment_recipe, expected_result)

    def test_use_binary_mix(self):
        experiment_recipe = [
            RecipeComponent("vinegar", 0.25),
            RecipeComponent("salt", 0.25),
        ]
        expected_result = [
            VialRecipeComponent("vial_salt_and_vinegar", 500.0),
            VialRecipeComponent("vial_solvent", 500.0, is_solvent=True),
        ]
        self.samples.loc[
            (self.samples["Conc_salt"] != 0.0) & (self.samples["Conc_vinegar"] == 0.0),
            "Viable",
        ] = False
        self.try_composition(experiment_recipe, expected_result)

    def test_exceed_slug_volume(self):
        experiment_recipe = [
            RecipeComponent("vinegar", 0.6),
            RecipeComponent("salt", 0.6),
        ]
        expected_result = [
            VialRecipeComponent("vial_salt_and_vinegar", 600.0),
            VialRecipeComponent("vial_solvent", 600.0, is_solvent=True),
        ]
        self.assertRaises(
            RecipeError,
            self.try_composition,
            experiment_recipe,
            expected_result,
        )

    def test_missing_vial(self):
        experiment_recipe = [
            RecipeComponent("vinegar", 0.25),
            RecipeComponent("salt", 0.25),
        ]
        expected_result = [
            VialRecipeComponent("vial_salt_and_vinegar", 500.0),
            VialRecipeComponent("vial_solvent", 500.0, is_solvent=True),
        ]
        self.samples.loc[self.samples["Conc_salt"] != 0.0, "Viable"] = False
        self.assertRaises(
            NoSuitableVialError,
            self.try_composition,
            experiment_recipe,
            expected_result,
        )

    def test_impossible(self):
        experiment_recipe = [
            RecipeComponent("vinegar", 0.2),
            RecipeComponent("salt", 0.22),
        ]
        expected_result = [
            VialRecipeComponent("vial_salt_and_vinegar", 400.0),
            VialRecipeComponent("vial_solvent", 600.0, is_solvent=True),
        ]
        self.samples.loc[
            (self.samples["Conc_salt"] != 0.0) & (self.samples["Conc_vinegar"] == 0.0),
            "Viable",
        ] = False
        self.assertRaises(
            RecipeError,
            self.try_composition,
            experiment_recipe,
            expected_result,
        )

    def test_avoid_empty_vials(self):
        experiment_recipe = [
            RecipeComponent("vinegar", 0.25),
            RecipeComponent("salt", 0.25),
        ]
        expected_result = [
            VialRecipeComponent("vial_salt_and_vinegar", 500.0),
            VialRecipeComponent("vial_solvent", 500.0, is_solvent=True),
        ]
        self.samples.loc[
            (self.samples["Conc_salt"] != 0.0) & (self.samples["Conc_vinegar"] == 0.0),
            "Volume",
        ] = 110.0
        self.try_composition(experiment_recipe, expected_result)

    def test_sampling_priority(self):
        experiment_recipe = [
            RecipeComponent("vinegar", 0.25),
            RecipeComponent("salt", 0.25, sampling_priority=2000),
        ]
        expected_result = [
            VialRecipeComponent("vial_vinegar", 250.0),
            VialRecipeComponent("vial_salt", 250.0, sampling_priority=2000),
            VialRecipeComponent("vial_solvent", 500.0, is_solvent=True),
        ]
        recipe = self.try_composition(experiment_recipe, expected_result)
        recipe = OrderRecipe.run(recipe)
        print("After ordering:")
        print(recipe)
        expected_result = [
            VialRecipeComponent("vial_salt", 250.0, sampling_priority=2000),
            VialRecipeComponent("vial_vinegar", 250.0),
            VialRecipeComponent("vial_solvent", 500.0, is_solvent=True),
        ]
        self.assertEqualRecipe(expected_result, recipe, "Recipe ordering failed.")
        experiment_recipe = [
            RecipeComponent("vinegar", 0.25, sampling_priority=2000),
            RecipeComponent("salt", 0.25),
        ]
        expected_result = [
            VialRecipeComponent("vial_salt_and_vinegar", 500.0, sampling_priority=2000),
            VialRecipeComponent("vial_solvent", 500.0, is_solvent=True),
        ]
        self.samples.loc[
            (self.samples["Conc_salt"] != 0.0) & (self.samples["Conc_vinegar"] == 0.0),
            "Viable",
        ] = False
        self.try_composition(experiment_recipe, expected_result)

    def test_ordering(self):
        vial_recipe = [
            VialRecipeComponent("vial_vinegar", 250.0),
            VialRecipeComponent("vial_salt", 250.0, sampling_priority=2000),
            VialRecipeComponent("vial_solvent", 250.0, is_solvent=True),
            VialRecipeComponent("vial_co-solvent", 250.0, is_solvent=True),
        ]
        ordered_recipe = OrderRecipe.run(vial_recipe, method="middle")
        print(ordered_recipe)
        expected_result = [
            VialRecipeComponent("vial_solvent", 125.0, is_solvent=True),
            VialRecipeComponent("vial_co-solvent", 125.0, is_solvent=True),
            VialRecipeComponent("vial_vinegar", 125.0),
            VialRecipeComponent("vial_salt", 250.0, sampling_priority=2000),
            VialRecipeComponent("vial_vinegar", 125.0),
            VialRecipeComponent("vial_co-solvent", 125.0, is_solvent=True),
            VialRecipeComponent("vial_solvent", 125.0, is_solvent=True),
        ]
        self.assertEqualRecipe(
            expected_result, ordered_recipe, "Recipe ordering failed."
        )

    def test_avoid_low_volumes(self):
        experiment_recipe = [
            RecipeComponent("vinegar", 0.25),
            RecipeComponent("salt", 0.25),
        ]
        expected_result = [
            VialRecipeComponent("vial_salt_and_vinegar", 500.0),
            VialRecipeComponent("vial_solvent", 500.0, is_solvent=True),
        ]
        self.samples.loc["vial_salt_and_vinegar_high", "Viable"] = True
        self.samples.loc[
            (self.samples["Conc_salt"] != 0.0) & (self.samples["Conc_vinegar"] == 0.0),
            "Viable",
        ] = False
        self.try_composition(experiment_recipe, expected_result)

    def test_avoid_low_volumes_but_cant(self):
        experiment_recipe = [
            RecipeComponent("vinegar", 0.25),
            RecipeComponent("salt", 0.25),
        ]
        expected_result = [
            VialRecipeComponent("vial_salt_and_vinegar_high", 12.5),
            VialRecipeComponent("vial_solvent", 1000.0 - 12.5, is_solvent=True),
        ]
        self.samples.loc["vial_salt_and_vinegar_high", "Viable"] = True
        self.samples.loc["vial_salt_and_vinegar", "Viable"] = False
        self.samples.loc[
            (self.samples["Conc_salt"] != 0.0) & (self.samples["Conc_vinegar"] == 0.0),
            "Viable",
        ] = False
        self.try_composition(experiment_recipe, expected_result)

    def test_giese(self):
        self.samples = pd.DataFrame(
            {
                DataFrameTask.index_name: [
                    "N2",
                    "waste",
                    "solvent",
                    "SM",
                    "TBADT",
                    "THF",
                ],
                DataFrameTask.vial_name: [
                    "N2",
                    "waste",
                    "solvent",
                    "SM",
                    "TBADT",
                    "THF",
                ],
                "Conc_SM": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                "Conc_TBADT": [0.0, 0.0, 0.0, 0.0, 0.017, 0.0],
                "Conc_THF": [0.0, 0.0, 0.0, 0.0, 0.0, 8.0],
                "Volume": [
                    0.0,
                    0.0,
                    10000.0,
                    4000.0,
                    4000.0,
                    4000.0,
                ],
                "Volume_min": [
                    100.0,
                    100.0,
                    500.0,
                    100.0,
                    100.0,
                    100.0,
                ],
                "Type": [
                    "Gas",
                    "Waste",
                    "Solvent",
                    "Stock",
                    "Stock",
                    "Stock",
                ],
                "Sampler": [
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                    "Sampler_cnc",
                ],
                "Viable": [
                    True,
                    True,
                    True,
                    True,
                    True,
                    True,
                ],
            }
        )
        self.samples.set_index(
            DataFrameTask.index_name, inplace=True, verify_integrity=True
        )
        self.platform = Platform()
        self.platform.samples = self.samples
        conc_SM = 0.05
        experiment_recipe_lower = [
            RecipeComponent("SM", conc_SM, sampling_priority=5000.0),
            RecipeComponent("TBADT", conc_SM * 0.005, sampling_priority=3000.0),
            RecipeComponent("THF", conc_SM * 1, sampling_priority=1000.0),
        ]
        self.try_composition(
            experiment_recipe_lower, None, order="middle", slug_volume=500.0
        )
        conc_SM = 0.2
        experiment_recipe_upper = [
            RecipeComponent("SM", conc_SM, sampling_priority=5000.0),
            RecipeComponent("TBADT", conc_SM * 0.025, sampling_priority=3000.0),
            RecipeComponent("THF", conc_SM * 20, sampling_priority=1000.0),
        ]
        self.try_composition(
            experiment_recipe_upper, None, order="middle", slug_volume=500.0
        )


if __name__ == "__main__":
    unittest.main()
