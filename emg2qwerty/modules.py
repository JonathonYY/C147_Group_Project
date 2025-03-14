# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
import torch
import logging
from torch import nn, Tensor


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C

        x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)


class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [
            mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
        ]
        return torch.stack(outputs_per_band, dim=self.stack_dim)


class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC

        # TNC -> NCT -> NcwT
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

        # Skip connection after downsampling
        T_out = x.shape[0]
        x = x + inputs[-T_out:]

        # Layer norm over C
        return self.layer_norm(x)  # TNC


class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
    """

    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # TNC
        x = self.fc_block(x)
        x = x + inputs
        return self.layer_norm(x)  # TNC


class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class TDSLSTMEncoder(nn.Module):
    def __init__(
        self,
        num_features: int,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 4,
    ) -> None:
        super().__init__()

        self.lstm_layers = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,  # Input shape: (batch_size, sequence_length, num_features)
            bidirectional=True,
        )

        # Fully connected block (remains the same)
        self.fc_block = TDSFullyConnectedBlock(lstm_hidden_size * 2)
        self.out_layer = nn.Linear(lstm_hidden_size * 2, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Forward pass through the LSTM
        x, _ = self.lstm_layers(inputs)  # (batch_size, seq_len, lstm_hidden_size * 2)
        
        # Apply FC transformation
        x = self.fc_block(x)  
        x = self.out_layer(x)
        
        return x

class CNNRNNHybrid(nn.Module):
    def __init__(
        self,
        input_channels: int,  # Change back to input_channels for compatibility
        cnn_features: int,
        rnn_hidden_size: int,
        num_rnn_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        # Treat input_channels as feature size since there's no spatial structure
        input_features = input_channels  

        # CNN Encoder (1D Convolution for sequential data)
        self.cnn_encoder = nn.Conv1d(
            in_channels=input_features, out_channels=cnn_features, kernel_size=3, padding=1
        )

        # RNN Encoder
        self.rnn_encoder = nn.LSTM(
            input_size=cnn_features,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,  # Input shape: (batch_size, sequence_length, features)
            bidirectional=True,
        )

        # Fully connected layers
        self.fc_block = TDSFullyConnectedBlock(rnn_hidden_size * 2)
        self.out_layer = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size, seq_len, features = inputs.shape  # (batch_size, sequence_length, features)

        # CNN expects (batch_size, features, sequence_length)
        x = inputs.permute(0, 2, 1)

        # CNN Encoder
        x = self.cnn_encoder(x)

        # Reshape for RNN: (batch_size, cnn_features, sequence_length) → (batch_size, sequence_length, cnn_features)
        x = x.permute(0, 2, 1)

        # RNN Encoder
        x, _ = self.rnn_encoder(x)

        # Take last time step (or use other strategy for RNN output)
        x = x[:, -1, :]  # (batch_size, rnn_hidden_size * 2)

        # Fully connected layers
        x = self.fc_block(x)
        x = self.out_layer(x)

        return x

class CNNGRUHybrid(nn.Module):
    """A hybrid model combining CNN and GRU for sequential data with spatial structure.

    Args:
        input_channels (int): Number of input channels (e.g., for spectrograms or images).
        cnn_features (int): Number of output features from the CNN encoder.
        gru_hidden_size (int): Hidden size of the GRU.
        num_gru_layers (int): Number of GRU layers.
        num_classes (int): Number of output classes for classification.
    """

    def __init__(
        self,
        input_channels: int,
        cnn_features: int,
        gru_hidden_size: int,
        num_gru_layers: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        # CNN Encoder 
        self.cnn_encoder = TDSConvEncoder(
            num_features=input_channels,
            block_channels=[cnn_features] * 4,  # Example: 4 CNN blocks
            kernel_width=3,  # Example kernel width
        )

        # GRU Encoder
        self.gru_encoder = nn.GRU(
            input_size=input_channels,
            hidden_size=gru_hidden_size,
            num_layers=num_gru_layers,
            batch_first=False,  # Input shape: (T, N, num_features)
            bidirectional=True,
        )

        # Fully connected block
        self.fc_block = TDSFullyConnectedBlock(gru_hidden_size * 2)
        self.out_layer = nn.Linear(gru_hidden_size * 2, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass for the hybrid CNN + GRU model.

        Args:
            inputs (Tensor): Input tensor of shape (T, N, num_features).

        Returns:
            Tensor: Output logits of shape (T, N, num_classes).
        """
        T, N, num_features = inputs.shape

        # Pass through CNN encoder
        x = self.cnn_encoder(inputs) 

        # Pass through GRU encoder
        x, _ = self.gru_encoder(x)

        # Apply fully connected block
        x = self.fc_block(x)

        # Final output layer
        x = self.out_layer(x)

        return x