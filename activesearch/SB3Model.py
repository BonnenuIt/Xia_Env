import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 80, 80), dtype=int)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # nn.MaxPool2d(2),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim * 2), nn.ReLU())

        self.linear_uav = nn.Sequential(nn.Linear(6, 32), 
                                         nn.ReLU(),
                                         nn.Linear(32, 64), 
                                         nn.ReLU(),
                                         nn.Linear(64, 16), 
                                         nn.ReLU(),
                                         )
        
        self.linear_total = nn.Sequential(nn.Linear(16 + features_dim * 2, 512), 
                                         nn.ReLU(),
                                         nn.Linear(512, 512), 
                                         nn.ReLU(),
                                         nn.Linear(512, features_dim), 
                                         nn.ReLU(),
                                         )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        BMap = torch.reshape(observations[:, 6:], ((observations.shape[0], 1, 80, 80)))
        UAV_locs = observations[:, :6]
        BMap_res = self.linear(self.cnn(BMap))
        UAV_res = self.linear_uav(UAV_locs)
        res = self.linear_total(torch.cat((BMap_res, UAV_res), dim=1))
        return res

def model_policy_kwargs():
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )
    return policy_kwargs
