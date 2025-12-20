import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_optimal_num_workers():
    # Detect CPU cores and use N-1 (min 1)
    cores = os.cpu_count() or 1
    return max(1, cores - 1)

class OxfordPetLoader:
    def __init__(self, root='./data', batch_size=8, image_size=256, download=True, cat_only=True):
        """
        Oxford-IIIT Pet Dataset Loader.
        cat_only: If True, only load cat breeds (Oxford Pet labels 1-12 are cats).
        """
        self.root = root
        self.batch_size = batch_size
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(image_size), # Resize to image_size
            transforms.CenterCrop(image_size), # Center crop
            transforms.ToTensor(), # Convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
        ])
        
        # Download and load full dataset
        full_dataset = datasets.OxfordIIITPet(
            root=root, 
            split='trainval',  # Use trainval split
            target_types='category',  # Load category labels
            download=download, # Download if not present
            transform=self.transform # Apply transformations
        )
        
        if cat_only:
            # Species mapping: 1 is Cat, 2 is Dog
            # In Oxford Pet, you can filter by species
            cat_indices = [i for i, (_, species) in enumerate(full_dataset) if species == 1]
            self.dataset = torch.utils.data.Subset(full_dataset, cat_indices) # Filter for cats only
            print(f"Filtered Oxford-Pet for Cats. Total Images: {len(self.dataset)}")
        else:
            self.dataset = full_dataset

    def get_loader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True, # Shuffle for training
            num_workers=get_optimal_num_workers(), # Get CPU cores - 1
            pin_memory=True # Pin memory for faster transfers
        )
