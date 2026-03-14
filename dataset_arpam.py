# %%
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Typo in sphincter :P
LABEL_MAP_BINARY = {"Normal": 0, "Sphincter": 0, "Scar": 0, "Tumor": 1}
LABELS_BINARY = ["Normal", "Tumor"]

LABEL_MAP_MULTI = {"Normal": 0, "Sphincter": 1, "Scar": 2, "Tumor": 3}
LABELS_MULTI = list(LABEL_MAP_MULTI.keys())


# Define a helper function to unnormalize the images for display
def unnormalize(image, mean, std):
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)  # Unnormalize

    return np.moveaxis(image.cpu().numpy(), 0, -1)


class ArpamBScanDataset(Dataset):
    df: pd.DataFrame
    image_type: str
    target_type: str

    transform: Callable | None
    target_transform: Callable | None

    # Radiomics features
    rf: np.ndarray | None
    rf_augment: bool

    def __init__(
        self,
        dataset_df: pd.DataFrame,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        target_type: Literal["pathology", "response"] = "response",
        image_type: str = "USradial",
        force_3chan: bool = False,
        rf: np.ndarray | None = None,
        rf_augment=False,
    ):
        """
        image_type:
            Existing images:
                - USradial
                - USrect
                - PAradial
                - PArect

            Combined to 3 channels. Returns ([US, US, PA], y)
                - PAUSradial
                - PAUSrect

            Returns 2 images, ([PA], [US], y)
                - PAUSradial-pair
                - PAUSrect-pair

        target_type:
            "pathology": vector of [T, N, TRG]
            "response": 0 or 1 (normal, tumor)
        """
        self.df = dataset_df
        self.transform = transform
        self.target_transform = target_transform
        self.image_type = image_type
        self.target_type = target_type
        self.force_3chan = force_3chan

        self.rf = rf
        self.rf_augment = rf_augment

        assert image_type in (
            "PAradial",
            "USradial",
            "PAUSradial",
            "PAUSradial-pair",
            "PArect",
            "USrect",
            "PAUSrect",
            "PAUSrect-pair",
        )

        if target_type == "response":
            self.y_columns = "has_tumor"
        else:
            self.y_columns = ["T", "TRG"]

    def __len__(self):
        return len(self.df)

    def _crop_rect(self, x):
        # Crop out the left side transducer artifact
        # and the top/bottom rotation join artifact
        h, w = x.shape[-2:]
        x1 = 150
        x2 = w
        y1 = 30
        y2 = h - 30
        x = x[..., y1:y2, x1:x2]
        # crop_to = (150, 30, w, h - 30)
        # x = x.crop(crop_to)
        return x

    def __getitem__(self, idx):
        # Images are loaded as gray scale (H, W) arrays
        # For PAUS, stacked into (C, H, W)
        row = self.df.iloc[idx]
        y = np.array(row[self.y_columns], dtype=int)
        if self.target_transform:
            y = self.target_transform(y)

        if self.image_type.startswith("PAUS"):
            # "PAUS*"
            if "radial" in self.image_type:
                PA = cv2.imread(row["PAradial"], cv2.IMREAD_GRAYSCALE)
                US = cv2.imread(row["USradial"], cv2.IMREAD_GRAYSCALE)
            else:
                PA = cv2.imread(row["PArect"], cv2.IMREAD_GRAYSCALE)
                US = cv2.imread(row["USrect"], cv2.IMREAD_GRAYSCALE)

            x = np.stack((US, US, PA), dtype=np.float32) / 255.0  # (3, H, W)
        else:
            # "US*" or "PA*"
            x = cv2.imread(row[self.image_type], cv2.IMREAD_GRAYSCALE)  # type: ignore

            if self.force_3chan:
                x = np.stack((x, x, x), dtype=np.float32) / 255.0
            else:
                x = x[None, ...].astype(np.float32) / 255.0  # (1, H, W)

        if "rect" in self.image_type:
            x = self._crop_rect(x)

        if self.transform:
            x = self.transform(torch.tensor(x, dtype=torch.float32))

        if self.image_type.endswith("pair"):
            # (US, PA, y)
            if self.force_3chan:
                if self.transform:
                    # x is torch.tensor
                    return (
                        x[None, 0, ...].repeat(3, 1, 1),
                        x[None, 2, ...].repeat(3, 1, 1),
                        y,
                    )
                else:
                    # x is np.array
                    return (
                        np.repeat(x[None, 0, ...], 3, axis=0),
                        np.repeat(x[None, 2, ...], 3, axis=0),
                        y,
                    )

            # TODO: support RF

            return x[None, 0, ...], x[None, 2, ...], y

        if self.rf is not None:
            rf = self.rf[idx]
            rf = torch.tensor(rf, dtype=torch.float32)

            if self.rf_augment:
                # Add 0.05 gaussian noise
                noise = torch.randn_like(rf) * 0.05
                rf += noise
            return x, rf, y

        return x, y


# %%
if __name__ == "__main__":
    import pandas as pd

    # image_size = (256, 256)
    image_size = (512, 512)

    image_type = "PAUSradial"
    target_type = "pathology"
    # target_type = "response"

    # root = "~/data/roi_to_281"
    dataset_df = pd.read_csv("~/data/arpam_roi_select_281/bscan_dataset.csv")
    ds = ArpamBScanDataset(dataset_df, image_type=image_type, target_type=target_type)

    # %%
    import matplotlib.pyplot as plt

    x, y = ds[200]
    if len(x) > 1:
        f, ax = plt.subplots(1, len(x))
        for i in range(len(x)):
            ax[i].set_axis_off()
            ax[i].imshow(x[i], "gray")
            ax[i].set_title(f"Channel {i}")
    else:
        f, ax = plt.subplots()
        ax.imshow(x[0], "gray")
        ax.set_axis_off()

    # %%
    img = x[0]
    print(img.dtype, img.max(), img.min())

    img = img * 1.5 - 0.2
    print(img.dtype, img.max(), img.min())
    img = np.clip(img, 0.0, 1.0)

    plt.imshow(img, "gray")

    # %%
    import torchvision.transforms.v2 as T

    train_transform = T.Compose(
        [
            T.Lambda(lambda x: x * 1.5 - 0.2),  # increase contrast
            T.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
            T.RandomRotation(degrees=(-20, 20)),
            T.RandomResizedCrop(image_size, scale=(0.9, 1.0)),
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            T.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        ]
    )

    train_ds = ArpamBScanDataset(
        dataset_df,
        image_type=image_type,
        target_type=target_type,
        transform=train_transform,
    )

    x, y = train_ds[200]
    img = x[0]
    plt.figure(figsize=(7, 7))
    plt.imshow(img, "gray")
    plt.title(f"{y}")
    plt.gca().set_axis_off()

    x[:5, :5]

    # %%
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, 4, True)
    X, Y = next(iter(dl))

    # %%
    plt.imshow(X[0][2].numpy(), "gray")

    # %%
    """
    Test BScan dataset with RF
    """
    from mri.preprocess_radiomics import load_MR_radiomics_features

    mri_root = "~/data/mr-radiomics-20250910-noimage"

    rf_mr1, rf_mr2 = load_MR_radiomics_features(
        Path(mri_root) / "nrrd_t2_mask/results_pai.csv"
    )

    ds = ArpamBScanDataset(
        dataset_df, image_type=image_type, target_type=target_type, rf=rf_mr1
    )

    pids_with_mr1 = sorted(set(dataset_df["pid"].unique()).intersection(rf_mr1.index))

    df = dataset_df.loc[dataset_df["pid"].isin(pids_with_mr1)]

    rf_mr1 = rf_mr1.loc[df["pid"]]

    # %%
    # Preprocess radiomics
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # standard scaler scales to mean 0 variance 1
    preprocess_rf = Pipeline([("scaler", StandardScaler())])
    preprocess_rf.fit(rf_mr1)

    rf_mr1_transformed = preprocess_rf.transform(rf_mr1)

    # %%
    ds_with_mr = ArpamBScanDataset(
        dataset_df,
        image_type=image_type,
        target_type=target_type,
        rf=rf_mr1_transformed.astype(np.float32),
        rf_augment=True,
    )

    _, mr_feat, _ = ds_with_mr[0]
    mr_feat = mr_feat.cpu().numpy()
    plt.subplot(211)
    plt.plot(mr_feat)

    noise = torch.randn(mr_feat.size) * 0.05
    plt.subplot(212)
    plt.plot(mr_feat + noise.numpy())
