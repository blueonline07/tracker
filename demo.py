from visualizer import read_video_from_path, Visualizer
from predictor import Predictor
import torch
from train_on_kubric import Lite
from tracker import Tracker
from torchinfo import summary

model = Lite.load_from_checkpoint('epoch=9-step=10.ckpt', tracker = Tracker(resolution = (96, 128)))
summary(model, input = (1, 1, 3, 96, 128))

video = read_video_from_path('assets/apple.mp4')
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
predictor = Predictor(model)
pred_tracks, pred_visibilities = predictor.run(video=video, grid_size = 3)

vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
vis.visualize(
    video,
    pred_tracks,
    pred_visibilities,
    query_frame=0
)

