import pandas as pd
from functools import lru_cache
from gluonts.time_feature import time_features_from_frequency_str, get_lags_for_frequency
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    ValidationSplitSampler,
    SetFieldIfNotPresent,
    TestSplitSampler,
    RemoveFields,
    VstackFeatures,
    RenameFields,
    InstanceSplitter,
)
from gluonts.transform.sampler import InstanceSampler
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches
from typing import Optional, Iterable
import torch

class DataLoaderCreator:
    """
    Class responsible for creating training and production datasets.
    """
    def __init__(self, config):
        self.config = config
        self.freq = config.freq
        self.time_features = time_features_from_frequency_str(self.freq)
        self.lags_sequence = get_lags_for_frequency(self.freq)
        self.transformation = self.create_transformation()

    @staticmethod
    @lru_cache(10_000)
    def convert_to_pandas_period(date, freq):
        """
        Converts a date string to a pandas Period object.

        Args:
            date (str): Date string.
            freq (str): Frequency string.

        Returns:
            pd.Period: Pandas Period object.
        """
        return pd.Period(date, freq)

    def transform_start_field(self, batch):
        """
        Transforms the 'start' field in the batch to pandas Period objects.

        Args:
            batch (dict): Batch of data.

        Returns:
            dict: Batch with transformed 'start' field.
        """
        batch["start"] = [self.convert_to_pandas_period(date, self.freq) for date in batch["start"]]
        return batch

    def create_transformation(self):
        """
        Creates a transformation chain for the dataset.

        Returns:
            Transformation: GluonTS transformation chain.
        """
        config = self.config
        remove_field_names = []
        if config.num_static_real_features == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if config.num_dynamic_real_features == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if config.num_static_categorical_features == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_CAT)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_CAT,
                        expected_ndim=1,
                        dtype=int,
                    )
                ]
                if config.num_static_categorical_features > 0
                else []
            )
            + (
                [
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_REAL,
                        expected_ndim=1,
                    )
                ]
                if config.num_static_real_features > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=1 if config.input_size == 1 else 2,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=config.prediction_length,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=config.prediction_length,
                    log_scale=True,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if config.num_dynamic_real_features > 0
                        else []
                    ),
                ),
                RenameFields(
                    mapping={
                        FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                        FieldName.FEAT_STATIC_REAL: "static_real_features",
                        FieldName.FEAT_TIME: "time_features",
                        FieldName.TARGET: "values",
                        FieldName.OBSERVED_VALUES: "observed_mask",
                    }
                ),
            ]
        )

    def create_instance_splitter(
        self,
        mode: str,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ):
        """
        Creates an instance splitter for the dataset.

        Args:
            mode (str): Mode of operation ('train' or 'test').
            train_sampler (InstanceSampler, optional): Sampler for training data.
            validation_sampler (InstanceSampler, optional): Sampler for validation data.

        Returns:
            Transformation: InstanceSplitter transformation.
        """
        assert mode in ["train", "test", "validation"]
        config = self.config

        instance_sampler = {
            "train": train_sampler
            or ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=config.prediction_length
            ),
            "validation": validation_sampler
            or ValidationSplitSampler(min_future=config.prediction_length),
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field="values",
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=config.context_length + max(config.lags_sequence),
            future_length=config.prediction_length,
            time_series_fields=["time_features", "observed_mask"],
        )
    
    def create_backtest_dataloader(
        self,
        data,
        batch_size: int
    ):
        config = self.config
        PREDICTION_INPUT_NAMES = [
            "past_time_features",
            "past_values",
            "past_observed_mask",
            "future_time_features",
        ]
        if config.num_static_categorical_features > 0:
            PREDICTION_INPUT_NAMES.append("static_categorical_features")

        if config.num_static_real_features > 0:
            PREDICTION_INPUT_NAMES.append("static_real_features")

        transformed_data = data.with_transform(self.transform_start_field)
        transformed_data = self.transformation.apply(transformed_data)
        
        # we create a Validation Instance splitter which will sample the very last
        # context window seen during training only for the encoder.
        instance_sampler = self.create_instance_splitter("validation")

        # we apply the transformations in train mode
        testing_instances = instance_sampler.apply(transformed_data, is_train=True)
        
        return as_stacked_batches(
            testing_instances,
            batch_size=batch_size,
            output_type=torch.tensor,
            field_names=PREDICTION_INPUT_NAMES,
        )
    
    def create_train_dataloader(
        self,
        data,
        batch_size: int,
        num_batches_per_epoch: int,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = True,
    ) -> Iterable:
        """
        Creates a DataLoader for training.

        Args:
            data (Dataset): Dataset to load data from.
            batch_size (int): Batch size.
            num_batches_per_epoch (int): Number of batches per epoch.
            shuffle_buffer_length (int, optional): Length of shuffle buffer. Defaults to None.
            cache_data (bool, optional): If True, caches the data. Defaults to True.

        Returns:
            Iterable: DataLoader for training.
        """
        config = self.config
        PREDICTION_INPUT_NAMES = [
            "past_time_features",
            "past_values",
            "past_observed_mask",
            "future_time_features",
        ]
        if config.num_static_categorical_features > 0:
            PREDICTION_INPUT_NAMES.append("static_categorical_features")

        if config.num_static_real_features > 0:
            PREDICTION_INPUT_NAMES.append("static_real_features")

        TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
            "future_values",
            "future_observed_mask",
        ]

        transformed_data = data.with_transform(self.transform_start_field)
        transformed_data = self.transformation.apply(transformed_data, is_train=True)
        if cache_data:
            transformed_data = Cached(transformed_data)

        instance_splitter = self.create_instance_splitter("train")

        stream = Cyclic(transformed_data).stream()
        training_instances = instance_splitter.apply(stream)

        return as_stacked_batches(
            training_instances,
            batch_size=batch_size,
            shuffle_buffer_length=shuffle_buffer_length,
            field_names=TRAINING_INPUT_NAMES,
            output_type=torch.tensor,
            num_batches_per_epoch=num_batches_per_epoch,
        )

    def create_prod_dataloader(
        self,
        data,
        batch_size: int,
    ) -> Iterable:
        """
        Creates a DataLoader for testing/production.

        Args:
            data (Dataset): Dataset to load data from.
            batch_size (int): Batch size.

        Returns:
            Iterable: DataLoader for testing.
        """
        config = self.config
        PREDICTION_INPUT_NAMES = [
            "past_time_features",
            "past_values",
            "past_observed_mask",
            "future_time_features",
        ]
        if config.num_static_categorical_features > 0:
            PREDICTION_INPUT_NAMES.append("static_categorical_features")

        if config.num_static_real_features > 0:
            PREDICTION_INPUT_NAMES.append("static_real_features")

        transformed_data = data.with_transform(self.transform_start_field)
        transformed_data = self.transformation.apply(transformed_data, is_train=False)

        instance_splitter = self.create_instance_splitter("test")

        testing_instances = instance_splitter.apply(transformed_data, is_train=False)

        return as_stacked_batches(
            testing_instances,
            batch_size=batch_size,
            output_type=torch.tensor,
            field_names=PREDICTION_INPUT_NAMES,
        )
