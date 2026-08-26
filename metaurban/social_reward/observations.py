"""Collector-specific observations that keep policy state separate from VLM input."""

import gymnasium as gym

from metaurban.component.vehicle.base_vehicle import BaseVehicle
from metaurban.obs.mix_obs import ImageObservation, LidarStateObservation
from metaurban.obs.observation_base import BaseObservation


class MainCameraLidarStateObservation(BaseObservation):
    """Main-camera RGB plus the 271-D lidar state used by the reference PPO."""

    IMAGE = "image"
    STATE = "state"

    def __init__(self, config):
        super().__init__(config)
        self.img_obs = ImageObservation(config, "main_camera", config["norm_pixel"])
        self.state_obs = LidarStateObservation(config)

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                self.IMAGE: self.img_obs.observation_space,
                self.STATE: self.state_obs.observation_space,
            }
        )

    def observe(self, vehicle: BaseVehicle):
        return {
            self.IMAGE: self.img_obs.observe(),
            self.STATE: self.state_obs.observe(vehicle),
        }

    def destroy(self):
        self.img_obs.destroy()
        self.state_obs.destroy()
        super().destroy()
