"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: This is the unit tasks that control the alias liquid handler

"""

from omniplatypus.devices.knauer.autosampler_alias import (
    AutosamplerAlias,
    AliasPosition,
)
from omniplatypus.procedures.unit_tasks.base_unit_task import BaseUnitTaskTemplate
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    DataFrameTask,
    UpdateSampleViability,
    VialRecipeComponent,
)
from omniplatypus.devices.platform import Platform
import pandas as pd
import time


class MoveAliasNeedle(BaseUnitTaskTemplate):
    """
    Move the main needle of the alias sampler. safety is ensured by the alias sampler itself.
    """

    @classmethod
    def run(cls, sampler: AutosamplerAlias, destination: AliasPosition) -> None:
        """
        Move the needle to the destination.

        :param sampler: AutosamplerAlias
            The alias autosampler object
        :param destination: AliasPosition
            The destination position for the needle.
        """
        return super().run(
            sampler=sampler,
            destination=destination,
        )

    @classmethod
    def _validate_input(
        cls, sampler: AutosamplerAlias, destination: AliasPosition
    ) -> None:
        cls._validate_input_log(
            sampler, AutosamplerAlias, "Invalid type for argument 'sampler'."
        )
        cls._validate_input_log(
            destination, AliasPosition, "Invalid type for argument 'destination'."
        )

    @classmethod
    def _execute(
        cls,
        sampler: AutosamplerAlias,
        destination: AliasPosition,
    ) -> None:
        """Move to the destination"""

        sampler["position"] = destination


class AliasPumpingAction(BaseUnitTaskTemplate):
    """Simple alias autosampler pumping action, the alias has a very simple pumping system, so this is a very simple task
    this disregards the positioning of the valves and just pumps the volume
    the sampler will know if you're trying to pump more than it can handle and will throw an error so no need to check
    if the volume is negative the alias will dispense, if positive it will aspirate

    usage:
    call cls.run() to execute this task
    """

    @classmethod
    def run(
        cls, sampler: AutosamplerAlias, volume: float, needle_position: str = "DOWN"
    ) -> None:
        """
        Suck or push a determined volume.
        Note: Use negative volume values to dispensethe liquid, use positive to aspirate it.

        :param sampler: AutosamplerAlias
            The alias autosampler object
        :param volume: float
            The volume to pump in uL. Positive values add liquid to the vial, negative remove it.
        """
        return super().run(
            sampler=sampler, volume=volume, needle_position=needle_position
        )

    @classmethod
    def _validate_input(
        cls, sampler: AutosamplerAlias, volume: float, needle_position: str = "DOWN"
    ) -> None:
        cls._validate_input_log(
            sampler, AutosamplerAlias, "Invalid type for argument 'sampler'."
        )
        cls._validate_input_log(
            volume, lambda x: isinstance(x, int), "Volume must be a number."
        )
        cls._validate_input_log(
            needle_position,
            lambda x: x.lower() in ("down", "up"),
            "Needle position can be either 'DOWN' or 'UP'.",
        )

    @classmethod
    def _execute(
        cls, sampler: AutosamplerAlias, volume: float, needle_position
    ) -> None:
        """
        Suck or push a determined volume.
        Note: Use negative volume values to dispensethe liquid, use positive to aspirate it.

        :param sampler: AutosamplerAlias
            The alias autosampler object
        :param volume: float
            The volume to pump in uL. Positive values add liquid to the vial, negative remove it.
        """
        current_needle_position = sampler.meters[
            "needle_vertical_movement"
        ].last_known_value
        if current_needle_position != needle_position:
            sampler["needle_vertical_movement"] = needle_position

        if volume < 0:
            sampler["dispense"] = abs(volume)
        else:
            sampler["aspirate"] = volume


class AliasPump(DataFrameTask):
    """
    Suck or push a determined volume from a vial / sample.
    Moves to the vial specified via the sample id and returns to the travel position, unless otherwise specified.
    Note: Use negative volume values to suck from the vials, use zero to simply move to the vial (set retract=False to
    stay).

    Usage:
    call cls.run() to execute this task.
    """

    required_columns: set = {
        "Sampler",
        "Type",
        "Volume",
        "Volume_max",
        "Volume_min",
        "X",
        "Y",
        "T",
        "Viable",
    }

    @classmethod
    def run(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        sample_id: str,
        volume: int,
        needle_position: float | str = "plunge",
        retract: bool = True,
        update_volume: bool = True,
        allow_refill: bool = False,
        pause_after: float = 0.0,
    ) -> None:
        """
        Suck or push a determined volume from a vial / sample.
        First moves to the sample position, then pumps the volume, and finally returns to the travel position.
        Note: Use negative volume values to suck from the vials.

        @param platform: Platform
            The platform object running the experiment.
        @param samples: pandas.DataFrame
            Dataframe holding information on all samples and vials. See 'GenerateSampleDataframe' for more details.
        @param sample_id: str
            Unique identifier for the sample within the dataframe.
        @param volume: float
            The volume to pump from the vial in uL. Positive values add liquid to the vial, negative remove it.
        @param needle_position: float | str = "plunge"
            alias has no control on how deep the needle goes, it's either in or out,
            so if plunge is passed the needle will dive in, if no-plunge is passed the needle will stay out.
            If a string is provided (case-insensitive):
                'plunge': highest position inside the vial
                'no-plunge': The needle is left in the travel position
        @param retract: bool = True
            If set to False, the needle is left within the vial after the pumping operation.
            This can be used to keep pumping outside of this unit task.
        @param update_volume: bool = True
            If True, updates the volume of the vial in the samples dataframe.
        @param allow_refill: bool = False
            If True, allows pumping volumes larger than what is available in the syringe by refilling the syringe
            as many times as required.
        """
        return super().run(
            platform=platform,
            samples=samples,
            sample_id=sample_id,
            volume=volume,
            needle_position=needle_position,
            retract=retract,
            update_volume=update_volume,
            allow_refill=allow_refill,
            pause_after=pause_after,
        )

    @classmethod
    def _validate_input(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        sample_id: str,
        volume: int,
        needle_position: float | str = "plunge",
        retract: bool = True,
        update_volume: bool = True,
        allow_refill: bool = False,
        pause_after: float = 0.0,
    ) -> None:
        cls._validate_platform(platform)
        cls._validate_dataframe(samples)
        if isinstance(needle_position, str):
            needle_positions = ("plunge", "no-plunge")
            cls._validate_input_log(
                needle_position,
                lambda x: x.lower() in needle_positions,
                f"Needle position can be one of '{needle_positions}' (is {needle_position}).",
            )
        elif isinstance(needle_position, (int | float)):
            cls._validate_input_log(
                needle_position,
                lambda x: 0.0 <= x <= 1.0,
                f"If numerical, needle position must be 0.0 <= x <= 1.0 (is {needle_position}).",
            )

    @classmethod
    def _update_volume(
        cls,
        samples: pd.DataFrame,
        sample_id: str,
        volume_pumped: int,
    ) -> None:
        """
        Track the volume of liquid within a vial on the samples dataframe.

        @param samples: pandas.DataFrame
            DataFrame holding information on all samples and vials. See 'GenerateSampleDataframe' for more details.
        @param sample_id: str
            Unique identifier for the sample within the dataframe.
        @param volume_pumped: float
            Volume change in uL. Positive values add liquid to the vial, negative remove it.
        """
        volume = samples.loc[sample_id, "Volume"]
        volume -= volume_pumped

        if samples.loc[sample_id, "Type"] == "Sample":
            samples.loc[sample_id, "Viable"] = False
        else:
            UpdateSampleViability.run(samples_to_update=samples.loc[[sample_id]])

        samples.loc[sample_id, "Volume"] = volume

    @classmethod
    def _execute(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        sample_id: str,
        volume: int,
        needle_position: float | str = "plunge",
        retract: bool = True,
        update_volume: bool = True,
        allow_refill: bool = False,
        pause_after: float = 0.0,
    ) -> None:
        sampler = cls._get_sampler(platform=platform, sample_id=sample_id)
        sample_data = samples.loc[sample_id]

        # move to the sample position:
        MoveAliasNeedle.run(
            sampler=sampler,
            destination=AliasPosition(
                X=sample_data["X"], Y=sample_data["Y"], T=sample_data["T"]
            ),
        )

        match needle_position:
            case "plunge":
                sampler["needle_vertical_movement"] = "DOWN"
            case "no-plunge":
                sampler["needle_vertical_movement"] = "UP"

        if not volume == 0.0:
            # since pumping for the alias is a much easier command we don't need it's own class

            # make sure the valves are load and needle:
            sampler["syringe_valve_switching"] = "NEEDLE"
            sampler["injection_valve_switching"] = "LOAD"

            # pump the volume
            AliasPumpingAction.run(sampler=sampler, volume=volume)

            if update_volume:
                cls._update_volume(
                    samples=samples, sample_id=sample_id, volume_pumped=volume
                )

        if pause_after > 0.0:
            time.sleep(pause_after)

        if retract:
            sampler["needle_vertical_movement"] = "UP"


class CleanNeedle(DataFrameTask):
    """
    Cleans the needle by plunging and retracting the needle into a vial with cleaning solvent.

    Usage:
    call cls.run() to execute this task.
    """

    @classmethod
    def run(
        cls,
        platform: Platform,
        sampler: AutosamplerAlias,
    ) -> None:
        """
        Clean the needle by purging the required volume into a waste vial and rinsing into a cleaning vial.

        @param platform: Platform
            Platform running the campaign.
        @param sampler: Sampler
        the alias autosampler

        @return: None
        """
        return super().run(
            platform=platform,
            sampler=sampler,
        )

    @classmethod
    def _validate_input(
        cls,
        platform: Platform,
        sampler: AutosamplerAlias,
    ) -> None:
        cls._validate_platform(platform)
        cls._validate_input_log(
            sampler, AutosamplerAlias, "Invalid type for argument 'sampler'."
        )

    @classmethod
    def _execute(
        cls,
        platform: Platform,
        sampler: AutosamplerAlias,
    ) -> None:
        """
        Clean the needle by purging the required volume into a waste vial and rinsing into a cleaning vial.

        @param platform: Platform
            Platform running the campaign.
        @param sampler: Sampler
            Sampler device whose needle will be cleaned.
        @return: None
        """
        sampler["initial_wash"] = "START"


class AliasMix(DataFrameTask):
    """
    Mix the content of a vial by sucking and dispensing the content multiple times.

    Usage:
    call cls.run() to execute this task.
    """

    @classmethod
    def run(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        sample_id: str,
        volume: float,
        cycles: int = 5,
        needle_position: float | str = "plunge",
        retract: bool = True,
    ) -> None:
        """
        Mix the content of a vial by sucking and dispensing the content multiple times.

        @param platform: Platform
            The platform object running the experiment.
        @param samples: pandas.DataFrame
            Dataframe holding information on all samples and vials. See 'GenerateSampleDataframe' for more details.
        @param sample_id: str
            Unique identifier for the sample within the dataframe.
        @param volume: float
            The volume to pump from the vial in uL. Positive values add liquid to the vial, negative remove it.
        @param cycles: int = 5
            Number of cycles to mix the content.
        @param needle_position: float | str = "plunge"
            alias has no control on how deep the needle goes, it's either in or out,
            so if plunge is passed the needle will dive in, if no-plunge is passed the needle will stay out.
            If a string is provided (case-insensitive):
                'plunge': highest position inside the vial
                'no-plunge': The needle is left in the travel position
        @param retract: bool = True
            If set to False, the needle is left within the vial after the pumping operation.
            This can be used to keep pumping outside of this unit task.
        @param update_volume: bool = True
            If True, updates the volume of the vial in the samples dataframe.
        @param allow_refill: bool = False
            If True, allows pumping volumes larger than what is available in the syringe by refilling the syringe
            as many times as required.
        """
        return super().run(
            platform=platform,
            samples=samples,
            sample_id=sample_id,
            volume=volume,
            cycles=cycles,
            needle_position=needle_position,
            retract=retract,
        )

    @classmethod
    def _validate_input(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        sample_id: str,
        volume: float,
        cycles: int = 5,
        needle_position: float | str = "plunge",
        retract: bool = True,
    ) -> None:
        cls._validate_platform(platform)
        cls._validate_dataframe(samples)
        cls._validate_input_log(
            cycles,
            lambda x: x >= 0,
            "Number of cycles must be a positive integer.",
        )

    @classmethod
    def _execute(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        sample_id: str,
        volume: float,
        cycles: int = 5,
        retract: bool = True,
    ) -> None:
        sampler = cls._get_sampler(platform=platform, sample_id=sample_id)
        sample_data = samples.loc[sample_id]

        # move to the sample position:
        MoveAliasNeedle.run(
            sampler=sampler,
            destination=AliasPosition(
                X=sample_data["X"], Y=sample_data["Y"], T=sample_data["T"]
            ),
        )

        # plunge the needle:
        sampler["needle_vertical_movement"] = "DOWN"

        # since pumping for the alias is a much easier command we don't need it's own class
        # for mixing let's use the whole syringe:

        # make sure the valves are load and needle:
        sampler["syringe_valve_switching"] = "NEEDLE"
        sampler["injection_valve_switching"] = "LOAD"
        # make sure we are in the right position
        sampler["move_syringe"] = "HOME"
        for _ in range(cycles):
            sampler["move_syringe"] = "END"
            sampler["move_syringe"] = "HOME"

        if retract:
            sampler["needle_vertical_movement"] = "UP"


class PrepareReactionVial(DataFrameTask):
    """
    Prepare a reaction slug based on a recipe.
    The recipe is a list of VialRecipeComponents (vial ids and volumes), as returned by the GenerateComposition task.
    To maintain accuracy, does not allow the slug volume to exceed the syringe volume (no refills during slug
    preparation).

    Usage:
    call cls.run() to execute this task.
    """

    @classmethod
    def run(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        recipe: list[VialRecipeComponent],
        destination: str,
        clean_needle: bool = False,
    ) -> None:
        """
        Prepare a reaction slug based on a recipe.
        The recipe is a list of VialRecipeComponents (vial ids and volumes), as returned by the GenerateComposition task.
        To maintain accuracy, does not allow the slug volume to exceed the syringe volume (no refills during slug
        preparation).

        @param platform: Platform
            The platform performing the experiment.
        @param samples: pandas.DataFrame
            The DataFrame holding the sample information.
        @param recipe: list[VialRecipeComponent]
            List of vials components of the slug. Each component provides a volume value in uL corresponding to the
            volume which will need to be extracted from the vial to obtain the requested slug composition.
        @param destination: the vial name of the destination vial

        @param clean_needle: bool = False
            If set to true, the needle is cleaned after every ragent (not solvent) is sampled.

        """
        return super().run(
            platform=platform,
            samples=samples,
            recipe=recipe,
            clean_needle=clean_needle,
            destination=destination,
        )

    @classmethod
    def _validate_input(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        recipe: list[VialRecipeComponent],
        destination: str,
        clean_needle: bool = False,
    ) -> None:
        cls._validate_platform(platform)
        cls._validate_dataframe(samples)
        cls._validate_input_log(
            recipe, list, f"Invalid type '{type(recipe)}' for argument 'recipe'."
        )
        volumes = []
        sample_ids = []
        for vial in recipe:
            volumes.append(vial.volume)
            sample_ids.append(vial.vial_id)
        missing_ids = set(sample_ids) - set(samples.index)

        cls._validate_input_log(
            missing_ids,
            lambda x: len(x) == 0,
            "Recipe requires sample_ids which are not found in samples dataframe"
            f" (not found: {missing_ids}).",
        )
        sampler = cls._get_sampler(platform=platform, sample_id=sample_ids[0])
        sampler_name = samples.loc[sample_ids[0], "Sampler"]
        cls._validate_input_log(
            samples,
            lambda x: all(x.loc[sample_ids, "Sampler"] == sampler_name),
            "All samples used for preparing a reaction slug must be located in the same Sampler.",
        )
        cls._validate_input_log(
            volumes,
            lambda x: all([y >= 0.0 for y in x]),
            "Only positive volumes are allowed in recipe.",
        )
        # cls._validate_input_log(
        #     volumes,
        #     lambda x: sum(x) <= sampler["loop_volume"]*1.2,
        #     f"Slug volume ({sum(volumes)} uL) exceeds available sample loop volume ({sampler['loop_volume']} uL).",
        # )

        # loop through the recipe again and check that the
        # volumes we're trying to take are available in the vials
        # that the vials are viable
        for vial in recipe:
            cls._validate_input_log(
                samples.loc[vial.vial_id, "Viable"],
                lambda x: x == True,
                f"Vial '{vial.vial_id}' is not viable for sampling.",
            )
            cls._validate_input_log(
                samples.loc[vial.vial_id, "Volume"],
                lambda x: x >= vial.volume,
                f"Vial '{vial.vial_id}' does not contain enough volume for the recipe.",
            )

    @classmethod
    def _execute(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        recipe: list[VialRecipeComponent],
        destination: str,
        clean_needle: bool = False,
    ) -> None:
        sampler = cls._get_sampler(platform=platform, sample_id=recipe[0].vial_id)

        # run an initial wash, and then make sure the injection valve is set to load and the needle is set to needle
        # CleanNeedle.run(platform=platform, sampler=sampler)

        sampler["injection_valve_switching"] = "LOAD"
        sampler["syringe_valve_switching"] = "NEEDLE"
        syringe_volume = sampler["syringe_volume"]
        for vial in recipe:
            # a recipe might set an ingredient to zero, don't make a fuss about it.
            if not vial.volume == 0.0:
                remaining_volume = int(vial.volume)
                while remaining_volume > 0:
                    if remaining_volume > syringe_volume - 100:
                        transfer_volume = syringe_volume - 100
                    else:
                        transfer_volume = remaining_volume

                    # Aspirate from the source vial
                    AliasPump.run(
                        platform=platform,
                        samples=samples,
                        sample_id=vial.vial_id,
                        volume=transfer_volume,
                        retract=True,
                        update_volume=True,
                        pause_after=10,
                    )

                    # Dispense into the destination vial
                    AliasPump.run(
                        platform=platform,
                        samples=samples,
                        sample_id=destination,
                        volume=-transfer_volume,
                        retract=True,
                        update_volume=True,
                        pause_after=10,
                    )

                    # Reduce the remaining volume
                    remaining_volume -= transfer_volume

                if clean_needle and not vial.is_solvent:
                    CleanNeedle.run(
                        platform=platform,
                        sampler=sampler,
                    )


class AliasLoadSlug(DataFrameTask):
    """The to load in the alias we need to do it in a semi smart way,
    we move to the vial to load from, then we load all the solution in the buffer tube by repeatedly loading the syringe and
    emptying it to waste. Then we load the air/nitrogen bubble, then we flip the injection valve to inject and we load the
    whole slug in the sample loop, (if neeeded we can use the syringe to reload from the solvent vial to push
    more volume in the sample loop). Then we switch the vial back to laod and we make sure to wash the needle
    """

    @classmethod
    def run(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        sample_name: str,
        slug_volume: float,
        front_bubble_bool: bool = True,
        front_bubble_volume: float = 50,
        front_bubble_vial: str = "None",
        back_bubble_bool: bool = False,
        back_bubble_volume: float = 50,
        back_bubble_vial: str = "None",
    ) -> None:
        """
        Load a reaction slug based on a recipe into the sample loop of the autosampler.

        @param platform: Platform
            The platform performing the experiment.
        @param samples: pandas.DataFrame
            The DataFrame holding the sample information.
        @param sample_name: str
            The name of the vial from which the slug will be loaded.
        @param slug_volume: float
            The volume of the slug in uL.
        @param front_bubble_bool: bool
            If True, a bubble is loaded after the slug. this will mean that the bubble will be injected at the front
        @param front_bubble_volume: float
            The volume of the bubble in uL.
        @param front_bubble_vial: str
            The vial from which the front bubble will be loaded. if None air will be loaded
        @param back_bubble_bool: bool
            If True, a bubble is loaded before the slug. this will mean that the bubble will be injected at the back
        @param back_bubble_volume: float
            The volume of the bubble in uL.
        @param back_bubble_vial: str
            The vial from which the back bubble will be loaded. if None air will be loaded


        """
        return super().run(
            platform=platform,
            samples=samples,
            sample_name=sample_name,
            slug_volume=slug_volume,
            front_bubble_bool=front_bubble_bool,
            front_bubble_volume=front_bubble_volume,
            front_bubble_vial=front_bubble_vial,
            back_bubble_bool=back_bubble_bool,
            back_bubble_volume=back_bubble_volume,
            back_bubble_vial=back_bubble_vial,
        )

    @classmethod
    def _validate_input(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        sample_name: str,
        slug_volume: float,
        front_bubble_bool: bool = True,
        front_bubble_volume: float = 50,
        front_bubble_vial: str = "None",
        back_bubble_bool: bool = False,
        back_bubble_volume: float = 50,
        back_bubble_vial: str = "None",
    ) -> None:
        cls._validate_platform(platform)
        cls._validate_dataframe(samples)
        sampler = cls._get_sampler(platform=platform, sample_id=sample_name)

        cls._validate_input_log(
            sample_name,
            lambda x: x in samples.index,
            f"Sample '{sample_name}' not found in samples dataframe.",
        )
        cls._validate_input_log(
            slug_volume,
            lambda x: x > 0,
            "Slug volume must be a positive number.",
        )

        volume_sum = slug_volume

        if front_bubble_bool:
            cls._validate_input_log(
                front_bubble_vial,
                lambda x: x in samples.index or x == "None",
                f"Front bubble vial '{front_bubble_vial}' not found in samples dataframe.",
            )
            cls._validate_input_log(
                front_bubble_volume,
                lambda x: x >= 0,
                "Front bubble volume must be a non-negative number.",
            )
            volume_sum += front_bubble_volume
        if back_bubble_bool:
            cls._validate_input_log(
                back_bubble_vial,
                lambda x: x in samples.index or x == "None",
                f"Back bubble vial '{back_bubble_vial}' not found in samples dataframe.",
            )
            cls._validate_input_log(
                back_bubble_volume,
                lambda x: x >= 0,
                "Back bubble volume must be a non-negative number.",
            )
            volume_sum += back_bubble_volume

        cls._validate_input_log(
            volume_sum,
            lambda x: x <= sampler["loop_volume"],
            f"Total volume of slug and bubbles ({volume_sum} uL) exceeds available sample loop volume ",
        )

    @classmethod
    def _execute(
        cls,
        platform: Platform,
        samples: pd.DataFrame,
        sample_name: str,
        slug_volume: float,
        front_bubble_bool: bool = True,
        front_bubble_volume: float = 50,
        front_bubble_vial: str = "None",
        back_bubble_bool: bool = False,
        back_bubble_volume: float = 50,
        back_bubble_vial: str = "None",
    ) -> None:
        # step 1: get the autosampler
        sampler = cls._get_sampler(platform=platform, sample_id=sample_name)

        # step 2: clean the needle
        CleanNeedle.run(platform=platform, sampler=sampler)

        # make sure the injection valve is set to load and the needle is set to needle
        sampler["injection_valve_switching"] = "LOAD"

        sampler["syringe_valve_switching"] = "WASTE"
        sampler["move_syringe"] = "HOME"
        sampler["syringe_valve_switching"] = "NEEDLE"
        # set the wait parameter:
        wait = 3 if back_bubble_bool else 1
        syringe_volume = sampler["syringe_volume"]
        dead_volume = sampler["tubing_volume"]

        # if we have a back bubble we need to load it first
        if back_bubble_bool is True:
            if back_bubble_vial == "None":
                AliasPumpingAction.run(
                    sampler=sampler,
                    volume=int(back_bubble_volume),
                    needle_position="UP",
                )
            else:
                AliasPump.run(
                    platform=platform,
                    samples=samples,
                    sample_id=back_bubble_vial,
                    volume=int(back_bubble_volume),
                    needle_position="plunge",
                    retract=True,
                    update_volume=False,
                )
            time.sleep(wait)

            # return the syringe to emtpy
            sampler["syringe_valve_switching"] = "WASTE"
            sampler["move_syringe"] = "HOME"
            sampler["syringe_valve_switching"] = "NEEDLE"

        # step 3: load the slug
        remaining_volume = int(slug_volume)
        if front_bubble_bool == False:
            remaining_volume = int(remaining_volume + dead_volume)
        while remaining_volume > 0:
            if remaining_volume > syringe_volume - 100:
                transfer_volume = syringe_volume - 100
            else:
                transfer_volume = remaining_volume

            AliasPump.run(
                platform=platform,
                samples=samples,
                sample_id=sample_name,
                volume=transfer_volume,
                retract=True,
                update_volume=True,
            )

            time.sleep(wait)

            # switch the syringe valve to waste and empty it:
            sampler["syringe_valve_switching"] = "WASTE"
            sampler["move_syringe"] = "HOME"
            sampler["syringe_valve_switching"] = "NEEDLE"

            remaining_volume -= transfer_volume

        # if we have a front bubble we need to load it now
        if front_bubble_bool is True:
            front_bubble_volume = int(front_bubble_volume) + dead_volume
            if front_bubble_vial == "None":
                AliasPumpingAction.run(
                    sampler=sampler, volume=front_bubble_volume, needle_position="UP"
                )
            else:
                AliasPump.run(
                    platform=platform,
                    samples=samples,
                    sample_id=front_bubble_vial,
                    volume=front_bubble_volume,
                    needle_position="plunge",
                    retract=True,
                    update_volume=False,
                )
            time.sleep(wait)

        # step 4: now do the reverset to load the slug in the sample loop
        sampler["injection_valve_switching"] = "INJECT"
        sampler["syringe_valve_switching"] = "WASH"
        sampler["move_syringe"] = "END"

        total_volume = slug_volume + front_bubble_volume + back_bubble_volume
        remaining_volume = int(total_volume)

        while remaining_volume > 0:
            if remaining_volume > syringe_volume - 100:
                transfer_volume = syringe_volume - 100
            else:
                transfer_volume = remaining_volume

            AliasPumpingAction.run(
                sampler=sampler, volume=-transfer_volume, needle_position="UP"
            )

            time.sleep(wait)

            # switch the syringe valve to waste and empty it:
            sampler["syringe_valve_switching"] = "WASH"
            sampler["move_syringe"] = "END"
            sampler["syringe_valve_switching"] = "NEEDLE"

            remaining_volume -= transfer_volume

        # step 5: switch the vial back to load and clean the needle
        sampler["injection_valve_switching"] = "LOAD"
        sampler["syringe_valve_switching"] = "NEEDLE"

        CleanNeedle.run(platform=platform, sampler=sampler)


class AliasPrimeWaste(BaseUnitTaskTemplate):
    """fills syringe with solvent and empties to waste a bunch of times, just
    to fill the system with solvent"""

    @classmethod
    def run(
        cls,
        platform: Platform,
        sampler: AutosamplerAlias,
        number_of_primes: int = 5,
    ) -> None:
        """
        Clean the needle by purging the required volume into a waste vial and rinsing into a cleaning vial.

        @param platform: Platform
            Platform running the campaign.
        @param sampler: Sampler
        the alias autosampler

        @return: None
        """
        return super().run(
            platform=platform,
            sampler=sampler,
            number_of_primes=number_of_primes,
        )

    @classmethod
    def _validate_input(
        cls,
        platform: Platform,
        sampler: AutosamplerAlias,
        number_of_primes: int,
    ) -> None:
        cls._validate_input_log(
            platform, Platform, "platform argument must be a Platform object."
        )
        cls._validate_input_log(
            sampler, AutosamplerAlias, "sampler argument must be a Sampler object."
        )
        cls._validate_input_log(
            number_of_primes,
            lambda x: isinstance(x, (int)),
            "Number of primes must be an integer.",
        )

    @classmethod
    def _execute(
        cls,
        platform: Platform,
        sampler: AutosamplerAlias,
        number_of_primes: int,
    ) -> None:
        """
        Clean the needle by purging the required volume into a waste vial and rinsing into a cleaning vial.

        @param platform: Platform
            Platform running the campaign.
        @param sampler: Sampler
            Sampler device whose needle will be cleaned.
        @return: None
        """
        # check where the syringe is, if it's not to home switch
        # valve to waste and home:

        sampler["syringe_valve_switching"] = "WASTE"
        sampler["move_syringe"] = "HOME"
        sampler["syringe_valve_switching"] = "WASH"

        # check if the valve is in the right position

        sampler["syringe_valve_switching"] = "WASH"

        for _ in range(number_of_primes):
            # switch the valve to load and load the syringe
            sampler["move_syringe"] = "END"
            # switch the valve to waste and empty the syringe
            sampler["syringe_valve_switching"] = "WASTE"
            sampler["move_syringe"] = "HOME"
            sampler["syringe_valve_switching"] = "WASH"


class AliasPrimeNeedle(BaseUnitTaskTemplate):
    """flushes solvent through the needle to remove bubbles, then loads a bubble of known volume air and puts the
    syringe to home
    """

    @classmethod
    def run(
        cls,
        autosampler: AutosamplerAlias,
        volume_to_waste: int = 5000,
        bubble_volume: int = 50,
    ):
        """
        flushes solvent through the needle to remove bubbles, then loads a bubble of known volume air and puts the
        syringe to home

        @param autosampler: AutosamplerAlias
            The alias autosampler object
        @param volume_to_waste: int
            The volume to waste in uL
        @param bubble_volume: int
            The volume of the bubble in uL
        """
        return super().run(
            autosampler=autosampler,
            volume_to_waste=volume_to_waste,
            bubble_volume=bubble_volume,
        )

    @classmethod
    def _validate_input(
        cls, autosampler: AutosamplerAlias, volume_to_waste: int, bubble_volume: int
    ) -> None:
        cls._validate_input_log(
            autosampler, AutosamplerAlias, "Invalid type for argument 'autosampler'."
        )
        cls._validate_input_log(
            volume_to_waste,
            lambda x: isinstance(x, int),
            "Volume to waste must be an integer.",
        )
        cls._validate_input_log(
            bubble_volume,
            lambda x: isinstance(x, int) and x < 500,
            "Bubble volume must be an integer and less than 500 uL.",
        )

    @classmethod
    def _execute(
        cls, autosampler: AutosamplerAlias, volume_to_waste: int, bubble_volume: int
    ) -> None:
        """
        flushes solvent through the needle to remove bubbles, then loads a bubble of known volume air and puts the
        syringe to home

        @param autosampler: AutosamplerAlias
            The alias autosampler object
        @param volume_to_waste: int
            The volume to waste in uL
        @param bubble_volume: int
            The volume of the bubble in uL
        """
        # check where the syringe is, if it's not to home switch
        # valve to waste and home:
        autosampler["syringe_valve_switching"] = "WASTE"
        autosampler["move_syringe"] = "HOME"
        autosampler["syringe_valve_switching"] = "WASH"

        # put the needle to waste:
        autosampler["position"] = AliasPosition(special="WASTE")

        # flush the needle

        remaining_volume = int(volume_to_waste)
        syringe_volume = autosampler["syringe_volume"]
        while remaining_volume > 0:
            if remaining_volume > syringe_volume - 100:
                transfer_volume = syringe_volume - 100
            else:
                transfer_volume = remaining_volume
            AliasPumpingAction.run(autosampler, transfer_volume, needle_position="DOWN")
            autosampler["syringe_valve_switching"] = "NEEDLE"

            AliasPumpingAction.run(
                autosampler, -transfer_volume, needle_position="DOWN"
            )
            autosampler["syringe_valve_switching"] = "WASH"
            # Reduce the remaining volume
            remaining_volume -= transfer_volume

        autosampler["syringe_valve_switching"] = "NEEDLE"
        # load the bubble
        AliasPumpingAction.run(autosampler, bubble_volume, needle_position="UP")
        time.sleep(5)
        autosampler["syringe_valve_switching"] = "WASTE"
        # put the syringe to home
        autosampler["move_syringe"] = "HOME"
        autosampler["syringe_valve_switching"] = "WASH"
