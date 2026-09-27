import torch
import torch.nn as nn
import torch.nn.functional as F

class PClayer(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.layer = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()
        self.state = None # Re-initialized per batch

    def init_state(self, batch_size, device, init_value=None, requires_grad=True):
        if init_value is not None:
            self.state = init_value.clone().to(device).requires_grad_(requires_grad)
        else:
            self.state = torch.zeros(batch_size, self.layer.out_features, device=device, requires_grad=requires_grad)
    def forward(self, state_above):
        if self.state is None:
            raise RuntimeError("initialise state first")
        return self.activation(self.layer(state_above))

    def get_energy(self, state_above):
        if self.state is None:
            raise RuntimeError("initialise state first")
        prediction = self.activation(self.layer(state_above))
        return F.mse_loss(self.state, prediction)

class PCmodel(nn.Module):

    def __init__(self, input_dim=28*28, hidden_dim=128, output_dim=10):
        super().__init__()
        # Note: Generative direction is Top (10) -> Hidden (128) -> Bottom (784)
        self.hidden_layer = PClayer(output_dim, hidden_dim) # Predicts hidden from label
        self.data_layer = PClayer(hidden_dim, input_dim)    # Predicts data from hidden

    def init_hidden(self, x_data, batch_size, device):
        # The bottom state is clamped to the actual image data (no gradients needed)
        self.data_layer.init_state(batch_size, device, init_value=x_data, requires_grad=False)
        # The hidden state is free to be optimized (requires gradients)
        self.hidden_layer.init_state(batch_size, device, requires_grad=True)

    def forward(self, y_label):
        prediction_hidden = self.hidden_layer(y_label)
        prediction_data = self.data_layer(self.hidden_layer.state)
        return prediction_hidden, prediction_data

    def get_energy(self, y_label):
        # Total network energy is the sum of prediction errors at every layer
        energy_hidden = self.hidden_layer.get_energy(y_label)
        energy_data = self.data_layer.get_energy(self.hidden_layer.state)
        return energy_hidden + energy_data

class StatePredictor(nn.Module):

    def __init__(self, upstream_dim, downstream_dim, state_dim) -> None:
        super().__init__()
        self.upstream_predictor = nn.Sequential(
            nn.Linear(upstream_dim, state_dim),
            nn.GELU()
        )
        self.downstream_predictor= nn.Sequential(
            nn.Linear(downstream_dim, state_dim),
            nn.GELU()
        )

        self.state_combiner = nn.Sequential(
            nn.Linear(2 * state_dim, state_dim),
            nn.GELU()
        )
        
    def forward(self, upstream_pred, downstream_pred):
        state1 = self.upstream_predictor(upstream_pred)
        state2 = self.downstream_predictor(downstream_pred)
        cat_state = torch.cat([state1,state2], dim=1)
        return self.state_combiner(cat_state)

