import numpy as np
import torch
from sklearn.impute import KNNImputer


def dataloaders_to_numpy(dataloader: torch.utils.data.DataLoader) -> np.ndarray:
    """Convert a PyTorch DataLoader to a numpy array."""
    data = []
    for i, batch in enumerate(dataloader):
        data.append(batch["clean_data"])
    data = torch.cat(data, dim=0)
    return data.numpy()


def convert_to_nan(
    masked_data_batch: torch.tensor, mask_data_batch: torch.tensor
) -> torch.tensor:
    """Convert masked portion of data to NaN values, given a mask."""
    masked_data_batch = masked_data_batch.clone()
    for masked_data, mask in zip(masked_data_batch, mask_data_batch):
        mask_indices = [i for i in range(len(mask)) if mask[i] == 0]
        masked_data[mask_indices] = np.nan
    return masked_data_batch


def impute_knn(
    train_data_batch: np.ndarray,
    test_masked_data_batch: torch.tensor,
    test_mask_data_batch: torch.tensor,
    k: int = 3,
) -> torch.tensor:
    """Impute missing values in test data using KNN imputation."""
    test_masked_data_batch = convert_to_nan(
        test_masked_data_batch, test_mask_data_batch
    )
    imputer = KNNImputer(n_neighbors=k)
    imputer.fit(train_data_batch.squeeze(-1))

    imputed_data = imputer.transform(test_masked_data_batch.squeeze(-1))
    imputed_data = torch.tensor(imputed_data)

    return imputed_data.unsqueeze(-1)


# knn_imputed_batch = impute_knn(dataloaders_to_numpy(train_dataloader),
#                                masked_data_batch, mask_batch, k=3)
