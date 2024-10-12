import torch
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.envs.utils import RandomPolicy
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record import VideoRecorder
from torchrl.envs import HabitatEnv
from torchrl.envs.transforms import TransformedEnv


def main():

    print(f"Exploring env...")

    env_name = "HabitatRenderNavToObj-v0"

    video_logger = CSVLogger(
        exp_name=env_name, log_dir="videos", video_format="mp4", video_fps=30
    )
    recorder = VideoRecorder(logger=video_logger, tag="iteration", skip=2)

    env = TransformedEnv(
        HabitatEnv(env_name=env_name, from_pixels=True, pixels_only=True)
    )
    env.append_transform(recorder)

    policy = RandomPolicy(action_spec=env.action_spec)

    device = torch.device("cuda")

    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy,
        total_frames=300,
        max_frames_per_traj=100,
        frames_per_batch=100,
        device=device,
        storing_device=device,
    )

    for _ in collector:
        continue

    env.transform.dump()

    env.close()


if __name__ == "__main__":
    main()
