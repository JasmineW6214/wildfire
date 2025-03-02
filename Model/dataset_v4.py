import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os

LABELS = ['targetY', 'targetY_prob', 'targetY_o1', 'targetY_o1_prob', 
          'targetY_o2', 'targetY_o2_prob', 'targetY_o3', 'targetY_o3_prob']

class FireSequenceDataset(Dataset):
    """Dataset for fire prediction sequences"""
    def __init__(self, X, y):
        """
        Args:
            X (np.ndarray): Input sequences of shape (N, window_size, feature_size)
            y (np.ndarray): Target labels of shape (N,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BaselineSequenceDataset(Dataset):
    """Dataset for baseline models that converts sequence data to point data"""
    def __init__(self, X, y):
        """
        Args:
            X (np.ndarray): Input sequences of shape (N, window_size, feature_size)
            y (np.ndarray): Target labels of shape (N,)
        """
        # Take only the last timestep of each sequence for baseline models
        self.X = torch.FloatTensor(X[:, -1, :])  # Shape: (N, feature_size)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FireSeqMultiYsDataset(Dataset):
    """Dataset for fire prediction sequences with multiple target labels"""
    def __init__(self, X, y_dict, target_col='targetY'):
        """
        Args:
            X (np.ndarray): Input sequences of shape (N, window_size, feature_size)
            y_dict (dict): Dictionary containing different target labels, each of shape (N,)
            target_col (str): Which target column to use (e.g., 'targetY', 'targetY_o1', etc.)
        """
        self.X = torch.FloatTensor(X)
        self.y_dict = {k: torch.FloatTensor(v) for k, v in y_dict.items()}
        self.target_col = target_col
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_dict[self.target_col][idx]
    
    def set_target(self, target_col):
        """Change which target label to use"""
        if target_col not in self.y_dict:
            raise ValueError(f"Invalid target column: {target_col}")
        self.target_col = target_col

class BaselineMultiYsDataset(Dataset):
    """Dataset for baseline models with multiple target labels"""
    def __init__(self, X, y_dict, target_col='targetY'):
        """
        Args:
            X (np.ndarray): Input sequences of shape (N, window_size, feature_size)
            y_dict (dict): Dictionary containing different target labels, each of shape (N,)
            target_col (str): Which target column to use (e.g., 'targetY', 'targetY_o1', etc.)
        """
        # Take only the last timestep of each sequence for baseline models
        self.X = torch.FloatTensor(X[:, -1, :])  # Shape: (N, feature_size)
        self.y_dict = {k: torch.FloatTensor(v) for k, v in y_dict.items()}
        self.target_col = target_col
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_dict[self.target_col][idx]
    
    def set_target(self, target_col):
        """Change which target label to use"""
        if target_col not in self.y_dict:
            raise ValueError(f"Invalid target column: {target_col}")
        self.target_col = target_col

def create_dataloaders(data_path, batch_size=32, train_ratio=0.9, val_data_path=None):
    """
    Create data loaders for training and validation with optimized settings
    
    Args:
        data_path: Path to training data
        batch_size: Batch size for the dataloaders
        train_ratio: Ratio to split training data (ignored if val_data_path is provided)
        val_data_path: Optional path to separate validation data
    """
    # Load training data
    print("Loading training data from disk...")
    train_data = np.load(data_path)
    X_train, y_train = train_data['X'], train_data['y']
    
    if val_data_path:
        # Use separate validation data
        print("Loading validation data from disk...")
        val_data = np.load(val_data_path)
        X_val, y_val = val_data['X'], val_data['y']
    else:
        # Split training data
        n_samples = len(y_train)
        train_size = int(train_ratio * n_samples)
        X_val = X_train[train_size:]
        y_val = y_train[train_size:]
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]
    
    # Create datasets
    train_dataset = FireSequenceDataset(X_train, y_train)
    val_dataset = FireSequenceDataset(X_val, y_val)
    
    # Determine optimal number of workers
    num_workers = min(4, os.cpu_count())
    
    # Create optimized loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # Parallel data loading
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch next batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader

def create_test_loader(test_data_path, batch_size=32):
    """
    Create data loader for separate test dataset
    
    Args:
        test_data_path: Path to the test data
        batch_size: Batch size for the dataloader
        
    Returns:
        test_loader
    """
    # Load test data
    data = np.load(test_data_path)
    X_test, y_test = data['X'], data['y']
    
    # Create test dataset
    test_dataset = FireSequenceDataset(X_test, y_test)
    
    # Determine optimal number of workers
    num_workers = min(4, os.cpu_count())
    
    # Create loader with optimized settings
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # Parallel data loading
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2  # Prefetch next batches
    )
    
    return test_loader

def create_baseline_dataloaders(data_path, batch_size=32, train_ratio=0.9, val_data_path=None):
    """Create optimized data loaders for baseline models
    
    Args:
        data_path: Path to training data
        batch_size: Batch size for the dataloaders
        train_ratio: Ratio to split training data (ignored if val_data_path is provided)
        val_data_path: Optional path to separate validation data
    """
    # Load training data
    print("Loading training data from disk...")
    train_data = np.load(data_path)
    X_train, y_train = train_data['X'], train_data['y']
    
    if val_data_path:
        # Use separate validation data
        print("Loading validation data from disk...")
        val_data = np.load(val_data_path)
        X_val, y_val = val_data['X'], val_data['y']
    else:
        # Split training data
        n_samples = len(y_train)
        train_size = int(train_ratio * n_samples)
        X_val = X_train[train_size:]
        y_val = y_train[train_size:]
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]
    
    # Create datasets
    train_dataset = BaselineSequenceDataset(X_train, y_train)
    val_dataset = BaselineSequenceDataset(X_val, y_val)
    
    # Determine optimal number of workers
    num_workers = min(4, os.cpu_count())
    
    # Create optimized loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader

def create_baseline_test_loader(test_data_path, batch_size=32):
    """Create test data loader for baseline models
    
    Args:
        test_data_path: Path to the test data
        batch_size: Batch size for the dataloader
        
    Returns:
        test_loader
    """
    # Load test data
    print("Loading test data from disk...")
    data = np.load(test_data_path)
    X_test, y_test = data['X'], data['y']
    
    # Create test dataset
    test_dataset = BaselineSequenceDataset(X_test, y_test)
    
    # Determine optimal number of workers
    num_workers = min(4, os.cpu_count())
    
    # Create loader with optimized settings
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return test_loader

def create_multi_target_dataloaders(data_path, target_col='targetY', batch_size=32, 
                                  train_ratio=0.9, val_data_path=None):
    """
    Create data loaders for training and validation with multiple target labels
    
    Args:
        data_path: Path to training data
        target_col: Which target column to use
        batch_size: Batch size for the dataloaders
        train_ratio: Ratio to split training data (ignored if val_data_path is provided)
        val_data_path: Optional path to separate validation data
    """
    # Load training data
    print("Loading training data from disk...")
    train_data = np.load(data_path)
    X_train = train_data['X']
    y_train_dict = {col: train_data[col] for col in LABELS}
    
    if val_data_path:
        # Use separate validation data
        print("Loading validation data from disk...")
        val_data = np.load(val_data_path)
        X_val = val_data['X']
        y_val_dict = {col: val_data[col] for col in LABELS}
    else:
        # Split training data
        n_samples = len(X_train)
        train_size = int(train_ratio * n_samples)
        
        X_val = X_train[train_size:]
        X_train = X_train[:train_size]
        
        y_val_dict = {}
        for col in y_train_dict:
            y_val_dict[col] = y_train_dict[col][train_size:]
            y_train_dict[col] = y_train_dict[col][:train_size]
    
    # Add validation
    missing_cols = [col for col in LABELS if col not in train_data]
    if missing_cols:
        raise ValueError(f"Missing target columns in data: {missing_cols}")
    
    # Create datasets
    train_dataset = FireSeqMultiYsDataset(X_train, y_train_dict, target_col)
    val_dataset = FireSeqMultiYsDataset(X_val, y_val_dict, target_col)
    
    # Determine optimal number of workers
    num_workers = min(4, os.cpu_count())
    
    # Create optimized loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader 

def create_multi_target_test_loader(test_data_path, target_col='targetY', batch_size=32):
    """
    Create test data loader for multi-target datasets
    
    Args:
        test_data_path: Path to the test data
        target_col: Which target column to use
        batch_size: Batch size for the dataloader
        
    Returns:
        test_loader
    """
    # Load test data
    print("Loading test data from disk...")
    test_data = np.load(test_data_path)
    X_test = test_data['X']
    y_test_dict = {col: test_data[col] for col in LABELS}
    
    # Create test dataset
    test_dataset = FireSeqMultiYsDataset(X_test, y_test_dict, target_col)
    
    # Determine optimal number of workers
    num_workers = min(4, os.cpu_count())
    
    # Create loader with optimized settings
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return test_loader 

def create_multi_target_baseline_dataloaders(data_path, target_col='targetY', batch_size=32, 
                                           train_ratio=0.9, val_data_path=None):
    """
    Create data loaders for baseline models with multiple target labels
    
    Args:
        data_path: Path to training data
        target_col: Which target column to use
        batch_size: Batch size for the dataloaders
        train_ratio: Ratio to split training data (ignored if val_data_path is provided)
        val_data_path: Optional path to separate validation data
    """
    # Load training data
    print("Loading training data from disk...")
    train_data = np.load(data_path)
    X_train = train_data['X']
    y_train_dict = {col: train_data[col] for col in LABELS}
    
    if val_data_path:
        # Use separate validation data
        print("Loading validation data from disk...")
        val_data = np.load(val_data_path)
        X_val = val_data['X']
        y_val_dict = {col: val_data[col] for col in LABELS}
    else:
        # Split training data
        n_samples = len(X_train)
        train_size = int(train_ratio * n_samples)
        
        X_val = X_train[train_size:]
        X_train = X_train[:train_size]
        
        y_val_dict = {}
        for col in y_train_dict:
            y_val_dict[col] = y_train_dict[col][train_size:]
            y_train_dict[col] = y_train_dict[col][:train_size]
    
    # Add validation
    missing_cols = [col for col in LABELS if col not in train_data]
    if missing_cols:
        raise ValueError(f"Missing target columns in data: {missing_cols}")
    
    # Create datasets
    train_dataset = BaselineMultiYsDataset(X_train, y_train_dict, target_col)
    val_dataset = BaselineMultiYsDataset(X_val, y_val_dict, target_col)
    
    # Determine optimal number of workers
    num_workers = min(4, os.cpu_count())
    
    # Create optimized loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, val_loader

def create_multi_target_baseline_test_loader(test_data_path, target_col='targetY', batch_size=32):
    """
    Create test data loader for baseline models with multiple target labels
    
    Args:
        test_data_path: Path to the test data
        target_col: Which target column to use
        batch_size: Batch size for the dataloader
        
    Returns:
        test_loader
    """
    # Load test data
    print("Loading test data from disk...")
    test_data = np.load(test_data_path)
    X_test = test_data['X']
    y_test_dict = {col: test_data[col] for col in LABELS}
    
    # Create test dataset
    test_dataset = BaselineMultiYsDataset(X_test, y_test_dict, target_col)
    
    # Determine optimal number of workers
    num_workers = min(4, os.cpu_count())
    
    # Create loader with optimized settings
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return test_loader 
