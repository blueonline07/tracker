import torch
from tracker import Tracker
from model_utils import get_points_on_a_grid
import torch.nn.functional as F 

class Predictor:
    def __init__(self, model=None):
        if model is None:
            model = Tracker(resolution = (96, 128))
        self.model = model
        self.interp_shape = model.resolution

    def run(self, video, queries = None, grid_size = 0, grid_query_frame = 0):
        B, T, C, H, W = video.shape

        video = video.reshape(B * T, C, H, W)
        video = F.interpolate(
            video, tuple(self.interp_shape), mode="bilinear", align_corners=True 
        )
        video = video.reshape(B, T, 3, self.interp_shape[0], self.interp_shape[1])


        if queries is not None:
            B, N, D = queries.shape
            assert D == 3
            queries = queries.clone()
            queries[:, :, 1:] *= queries.new_tensor(
                [
                    (self.interp_shape[1] - 1) / (W - 1),
                    (self.interp_shape[0] - 1) / (H - 1),
                ]
            )
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(
                grid_size, self.interp_shape, device=video.device
            )
            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            ).repeat(B, 1, 1)
            
        tracks, visibilities, *_ = self.model(video, queries)

        thr = 0.9
        visibilities = visibilities > thr
        for i in range(len(queries)):
            queries_t = queries[i, : tracks.size(2), 0].to(torch.int64)
            arange = torch.arange(0, len(queries_t))

            # overwrite the predictions with the query points
            tracks[i, queries_t, arange] = queries[i, : tracks.size(2), 1:]

            # correct visibilities, the query points should be visible
            visibilities[i, queries_t, arange] = True

        tracks *= tracks.new_tensor(
            [(W - 1) / (self.interp_shape[1] - 1), (H - 1) / (self.interp_shape[0] - 1)]
        )
        return tracks, visibilities