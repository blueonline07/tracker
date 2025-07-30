import os
from tracker import Tracker
import torch
from my_datasets.utils import collate_fn_train
from torch.utils.data import DataLoader
from my_datasets import kubric_movif_dataset
import lightning as L


class Lite(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.tracker = Tracker()

    def training_step(self, batch):
        data, *_ = batch
        video = data.video[:, :16]        # shape: B, T, C, H, W → keep first 8 frames
        queries = data.trajectory[:, :16] # shape: B, T, N, 2 → match video

        out = torch.cat([torch.zeros_like(queries[:, 0, :, :1]), queries[:, 0]], dim=-1)
        pred_tracks, *_ = self.tracker(video, out)

        loss = torch.nn.functional.mse_loss(pred_tracks, queries)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.tracker.parameters(), lr=1e-4)

    

if __name__ == "__main__":
    dataset = kubric_movif_dataset.KubricMovifDataset(
        data_root=os.path.join(
            "/content/drive/MyDrive/data", "kubric/kubric_movi_f_120_frames_dense/movi_f"
        )
    )
    data_loader = DataLoader(dataset, collate_fn=collate_fn_train)
    model = Lite()
    trainer = L.Trainer(
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=4,
        max_epochs=10,
        limit_train_batches=1,
        accelerator="gpu"
    )
    trainer.fit(model=model, train_dataloaders=data_loader)