import torch
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction
import numpy as np
import pandas as pd
import datetime
from typing import Dict, Optional, List
import random
from evaluate import load
from gluonts.time_feature import get_seasonality
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from gluonts.dataset.field_names import FieldName


class TimeSeriesModel:
    """
    Class for training and predicting with the Time Series Transformer model.
    """
    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose
        self.model = TimeSeriesTransformerForPrediction(config)
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.model.to(self.device)
        self.freq = config.freq  # Frequency of the time series (e.g., 'H' for hourly)
        # Get mappings from config
        self.zone_to_int = config.zone_to_int
        self.int_to_zone = config.int_to_zone
        self.zones_list = config.zones_list
    
    def train(
        self,
        train_dataloader,
        epochs=40,
        lr=6e-4,
        betas=(0.9, 0.95),
        weight_decay=1e-1,
        print_every=100,
    ):
        """
        Trains the model.

        Args:
            train_dataloader (Iterable): DataLoader for training data.
            epochs (int, optional): Number of epochs. Defaults to 40.
            lr (float, optional): Learning rate. Defaults to 6e-4.
            betas (tuple, optional): Betas for AdamW optimizer. Defaults to (0.9, 0.95).
            weight_decay (float, optional): Weight decay for optimizer. Defaults to 1e-1.
            print_every (int, optional): Print loss every 'print_every' steps. Defaults to 100.
        """
        # Configure the optimizer
        optimizer = AdamW(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    
        # Prepare for distributed training
        self.model, optimizer, train_dataloader = self.accelerator.prepare(
            self.model,
            optimizer,
            train_dataloader,
        )
    
        # Set model to training mode
        self.model.train()
    
        # Training loop
        for epoch in range(epochs):
            if self.verbose:
                print(f"--- Epoch {epoch + 1}/{epochs} ---")
            for idx, batch in enumerate(train_dataloader):
                optimizer.zero_grad()  # Reset gradients
    
                # Prepare inputs
                static_categorical_features = (
                    batch["static_categorical_features"].to(self.device)
                    if self.config.num_static_categorical_features > 0 and "static_categorical_features" in batch
                    else None
                )
                static_real_features = (
                    batch["static_real_features"].to(self.device)
                    if self.config.num_static_real_features > 0 and "static_real_features" in batch
                    else None
                )
                past_time_features = batch["past_time_features"].to(self.device)
                past_values = batch["past_values"].to(self.device)
                future_time_features = batch["future_time_features"].to(self.device)
                future_values = batch["future_values"].to(self.device)
                past_observed_mask = batch["past_observed_mask"].to(self.device)
                future_observed_mask = batch["future_observed_mask"].to(self.device)
    
                # Forward pass
                outputs = self.model(
                    static_categorical_features=static_categorical_features,
                    static_real_features=static_real_features,
                    past_time_features=past_time_features,
                    past_values=past_values,
                    future_time_features=future_time_features,
                    future_values=future_values,
                    past_observed_mask=past_observed_mask,
                    future_observed_mask=future_observed_mask,
                )
                loss = outputs.loss  # Compute loss
    
                # Backpropagation
                self.accelerator.backward(loss)
                optimizer.step()  # Update model parameters
    
                # Print loss every 'print_every' steps
                if self.verbose and idx % print_every == 0:
                    print(f"Epoch {epoch + 1}, Step {idx}, Loss: {loss.item():.4f}")
    
    def predict(self, test_dataloader, last_observed_time: Optional[datetime.datetime] = None):
        """
        Generates predictions using the test DataLoader.
        Returns a dictionary containing median, std, dates (if last_observed_time is provided), etc.

        Args:
            test_dataloader (Iterable): DataLoader for test data.
            last_observed_time (datetime, optional): The datetime representing the last observed data point.
                                                     If None, dates will not be generated.

        Returns:
            Dict: Dictionary containing predictions and associated information.
        """
        self.model.eval()
        
        # Lists to store results
        predictions_median = []
        predictions_mean = []
        predictions_std = []
        sequences_list = []  # To store all sequences for potential plotting
        # past_values_list = []
        #past_dates_list = []
        item_id_list = []
        
        # Time increment based on frequency
        freq = self.freq
        prediction_length = self.config.prediction_length
        
        # Determine time increment
        time_increment = pd.tseries.frequencies.to_offset(freq)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                batch_size = batch["past_values"].shape[0]
                
                # Move tensors to device
                static_categorical_features = (
                    batch["static_categorical_features"].to(self.device)
                    if self.config.num_static_categorical_features > 0 and "static_categorical_features" in batch
                    else None
                )
                static_real_features = (
                    batch["static_real_features"].to(self.device)
                    if self.config.num_static_real_features > 0 and "static_real_features" in batch
                    else None
                )
                past_time_features = batch["past_time_features"].to(self.device)
                past_values = batch["past_values"].to(self.device)
                future_time_features = batch["future_time_features"].to(self.device)
                past_observed_mask = batch["past_observed_mask"].to(self.device)
                
                # Generate predictions
                outputs = self.model.generate(
                    static_categorical_features=static_categorical_features,
                    static_real_features=static_real_features,
                    past_time_features=past_time_features,
                    past_values=past_values,
                    future_time_features=future_time_features,
                    past_observed_mask=past_observed_mask,
                )
                
                # Get generated sequences
                sequences = outputs.sequences.cpu().numpy()  # Shape: (batch_size, num_samples, prediction_length)
                sequences_list.append(sequences)
                
                # Calculate median and std along samples axis
                median_prediction = np.median(sequences, axis=1)  # Shape: (batch_size, prediction_length)
                mean_prediction = np.mean(sequences, axis=1)
                std_prediction = np.std(sequences, axis=1)
                
                # Store results
                predictions_median.append(median_prediction)
                predictions_mean.append(mean_prediction)
                predictions_std.append(std_prediction)
                
                # Store past values
                # past_values_np = batch["past_values"].cpu().numpy()  # Shape: (batch_size, past_length)
                # past_values_list.append(past_values_np)
                
                # Reconstruct item_ids from static_categorical_features
                if static_categorical_features is not None:
                    static_cat_feat = static_categorical_features.cpu().numpy()
                    for idx_in_batch in range(static_cat_feat.shape[0]):
                        zone_idx = static_cat_feat[idx_in_batch][0]  # Assuming shape (batch_size, 1)
                        item_id = self.int_to_zone.get(zone_idx, f"Zone_{zone_idx}")
                        item_id_list.append(item_id)
                else:
                    item_id_list.extend([f"Series_{i}" for i in range(batch_size)])
        
        # Concatenate results
        predictions_median = np.concatenate(predictions_median, axis=0)  # Shape: (num_items, prediction_length)
        predictions_mean= np.concatenate(predictions_mean, axis=0)
        predictions_std = np.concatenate(predictions_std, axis=0)
        sequences_array = np.concatenate(sequences_list, axis=0)  # Shape: (num_items, num_samples, prediction_length)
        #past_values_array = np.concatenate(past_values_list, axis=0)  # Shape: (num_items, past_length)
        
        # Assemble results
        results = {}
        for idx, item_id in enumerate(item_id_list):
            # past_dates = past_dates_list[idx]
            if last_observed_time is not None:
                # Use generated dates
                prediction_dates = [
                    last_observed_time + (i + 1) * time_increment
                    for i in range(prediction_length)
                ]
            else:
                # Use numerical indices
                prediction_dates = list(range(prediction_length))
            
            results[item_id] = {
                'dates': prediction_dates,
                'median_prediction': predictions_median[idx],
                'std_prediction': predictions_std[idx],
                'mean_prediction': predictions_mean[idx],
                'lower_bound': predictions_median[idx] - predictions_std[idx],
                'upper_bound': predictions_median[idx] + predictions_std[idx],
                'sequences': sequences_array[idx],  # All sequences for this item
                #'past_dates': past_dates,
                #'past_values': past_values_array[idx],
            }
        return results

    def predict_test(self, test_dataloader):
        """
        Generates predictions using the test DataLoader for testing purposes.
        Returns arrays of predictions without dates or item IDs.

        Args:
            test_dataloader (Iterable): DataLoader for test data.

        Returns:
            Tuple: A tuple containing arrays of median predictions, mean predictions, standard deviations,
                   and the sequences of generated samples.
                   (median_predictions, mean_predictions, std_predictions, sequences_array)
        """
        self.model.eval()
        
        # Lists to store results
        predictions_median = []
        predictions_mean = []
        predictions_std = []
        sequences_list = []  # To store all sequences
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                batch_size = batch["past_values"].shape[0]
                
                # Move tensors to device
                static_categorical_features = (
                    batch["static_categorical_features"].to(self.device)
                    if self.config.num_static_categorical_features > 0 and "static_categorical_features" in batch
                    else None
                )
                static_real_features = (
                    batch["static_real_features"].to(self.device)
                    if self.config.num_static_real_features > 0 and "static_real_features" in batch
                    else None
                )
                past_time_features = batch["past_time_features"].to(self.device)
                past_values = batch["past_values"].to(self.device)
                future_time_features = batch["future_time_features"].to(self.device)
                past_observed_mask = batch["past_observed_mask"].to(self.device)
                
                # Generate predictions
                outputs = self.model.generate(
                    static_categorical_features=static_categorical_features,
                    static_real_features=static_real_features,
                    past_time_features=past_time_features,
                    past_values=past_values,
                    future_time_features=future_time_features,
                    past_observed_mask=past_observed_mask,
                )
                
                # Get generated sequences
                sequences = outputs.sequences.cpu().numpy()  # Shape: (batch_size, num_samples, prediction_length)
                sequences_list.append(sequences)
                
                # Calculate median, mean, and std along samples axis
                median_prediction = np.median(sequences, axis=1)  # Shape: (batch_size, prediction_length)
                mean_prediction = np.mean(sequences, axis=1)
                std_prediction = np.std(sequences, axis=1)
                
                # Store results
                predictions_median.append(median_prediction)
                predictions_mean.append(mean_prediction)
                predictions_std.append(std_prediction)
        
        # Concatenate results
        predictions_median = np.concatenate(predictions_median, axis=0)  # Shape: (num_samples, prediction_length)
        predictions_mean = np.concatenate(predictions_mean, axis=0)
        predictions_std = np.concatenate(predictions_std, axis=0)
        sequences_array = np.concatenate(sequences_list, axis=0)  # Shape: (num_samples, num_samples_per_series, prediction_length)
        
        return predictions_median, predictions_mean, predictions_std, sequences_array

    def evaluate_test(self, predictions_median, test_dataset):
        """
        Evaluates the model on the test data using MASE and sMAPE metrics.

        Args:
            predictions_median (List): List of median predictions.
            test_dataset (Dataset): The test dataset used to create the DataLoader.

        Returns:
            Dict: A dictionary containing the MASE and sMAPE metrics.
        """

        # Load evaluation metrics
        mase_metric = load("mase")
        smape_metric = load("smape")

        # Get seasonality (periodicity)
        periodicity = get_seasonality(self.freq)

        mase_metrics = []
        smape_metrics = []

        # Extract ground truth and training data from the test dataset
        # Note: Since we have sliding windows, we need to extract the appropriate parts
        for idx, ts in enumerate(test_dataset):
            target = ts["target"]
            window_size = len(target)
            prediction_length = self.config.prediction_length
            training_data = target[:-prediction_length]
            ground_truth = target[-prediction_length:]

            # Compute MASE
            mase = mase_metric.compute(
                predictions=predictions_median[idx],
                references=ground_truth,
                training=training_data,
                periodicity=periodicity,
            )
            mase_metrics.append(mase["mase"])

            # Compute sMAPE
            smape = smape_metric.compute(
                predictions=predictions_median[idx],
                references=ground_truth,
            )
            smape_metrics.append(smape["smape"])

        # Plot metrics
        plt.figure(figsize=(8, 6))
        plt.scatter(mase_metrics, smape_metrics, alpha=0.3)
        plt.xlabel("MASE")
        plt.ylabel("sMAPE")
        plt.title("MASE vs sMAPE on Test Data")
        plt.show()

        # Return metrics
        return {
            "mase": mase_metrics,
            "smape": smape_metrics,
        }

    def plot_test_predictions(self, test_dataset, sequences_array, num_plots=4):
        """
        Plots a selection of test predictions alongside the actual values.

        Args:
            test_dataset (Dataset): The test dataset used for predictions.
            sequences_array (np.ndarray): The sequences of generated samples.
            num_plots (int, optional): Number of plots to generate. Defaults to 4.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # Randomly select indices to plot
        num_samples = len(test_dataset)
        indices = random.sample(range(num_samples), min(num_plots, num_samples))

        # Plot settings
        num_rows = int(np.ceil(num_plots / 2))
        fig, axs = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
        axs = axs.flatten()

        for i, idx in enumerate(indices):
            ts = test_dataset[idx]
            target = ts["target"]
            prediction_length = self.config.prediction_length
            context_length = self.config.context_length
            freq = self.freq

            # Prepare dates
            start_time = pd.to_datetime(ts["start"])
            full_index = pd.date_range(start=start_time, periods=len(target), freq=freq)

            # Split target into training data (context) and ground truth
            training_data = target[:-prediction_length]
            ground_truth = target[-prediction_length:]

            # For plotting, take some context from the training data
            # Adjust the number of context points to display (e.g., last 2 * prediction_length)
            context_values = training_data[-2 * prediction_length:]
            context_index = full_index[-(prediction_length + len(context_values)):-prediction_length]
            prediction_index = full_index[-prediction_length:]

            # Predictions
            sequences = sequences_array[idx]  # Shape: (num_samples_per_series, prediction_length)
            median_prediction = np.median(sequences, axis=0)
            std_prediction = np.std(sequences, axis=0)

            # Plot
            ax = axs[i]
            ax.plot(context_index, context_values, label="Context", color="blue")
            ax.plot(prediction_index, ground_truth, label="Actual", color="green", linestyle="--")
            ax.plot(prediction_index, median_prediction, label="Median Prediction", color="orange")
            ax.fill_between(
                prediction_index,
                median_prediction - std_prediction,
                median_prediction + std_prediction,
                color="orange",
                alpha=0.3,
                label="+/- 1 std dev",
            )

            ax.set_title(f"Sample {idx} for {ts['item_id']}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax.get_xticklabels(), rotation=45)

        plt.tight_layout()
        plt.show()

    def save(self, path):
        """
        Saves the model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        self.model.save_pretrained(path)
    
    def load(self, path):
        """
        Loads the model from the specified path.

        Args:
            path (str): Path to load the model from.
        """
        self.model = TimeSeriesTransformerForPrediction.from_pretrained(path)
        self.model.to(self.device)
    