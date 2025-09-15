import math
import random
import os
import numpy as np
from neurograd import xp, Tensor, float32, int64
from typing import Optional, List, Tuple, Union, Callable, Dict
import glob
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import cv2
import os
import psutil
import warnings


# Try to import DALI first - this determines our capabilities
try:
    import nvidia.dali as dali
    from nvidia.dali import pipeline_def, Pipeline
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    # FIX 1: Import LastBatchPolicy from its location in older DALI versions
    from nvidia.dali.plugin.base_iterator import LastBatchPolicy
    from nvidia.dali.backend import PreallocateDeviceMemory, PreallocatePinnedMemory
    DALI_AVAILABLE = True

    class DALIGenericIterator:
        """
        A generic DALI iterator that is not tied to any specific framework.
        It iterates over a DALI pipeline and yields batches of data.
        """
        def __init__(self,
                     pipeline: Pipeline,
                     output_map: List[str],
                     last_batch_policy: LastBatchPolicy,
                     auto_reset: bool = False,
                     reader_name: Optional[str] = None,
                     prepare_first_batch: bool = True):
            self._pipeline = pipeline
            self._output_map = output_map
            self._last_batch_policy = last_batch_policy
            self._auto_reset = auto_reset
            self._reader_name = reader_name

            # Try to find reader name if not provided
            if not self._reader_name:
                readers = [op.name for op in self._pipeline.ops if "readers" in op.spec.name]
                if len(readers) == 1:
                    self._reader_name = readers[0]
                else:
                    raise ValueError(f"Could not automatically determine the reader name. "
                                     f"Found {len(readers)} readers: {readers}. Please specify 'reader_name'.")

            self._size = self._pipeline.epoch_size(self._reader_name)
            self._batch_size = self._pipeline.max_batch_size
            
            if self._last_batch_policy == LastBatchPolicy.DROP:
                self._num_batches = self._size // self._batch_size
            else:  # PARTIAL or FILL
                self._num_batches = math.ceil(self._size / self._batch_size)
            
            self._counter = 0

        def __iter__(self):
            return self

        def __len__(self):
            return self._num_batches

        def __next__(self):
            if self._counter >= self._num_batches:
                if self._auto_reset:
                    self.reset()
                raise StopIteration

            try:
                outputs = self._pipeline.run()
                self._counter += 1
                
                # FIX 2: Correctly map the tuple of outputs to the output_map keys.
                batch_dict = {key: outputs[i] for i, key in enumerate(self._output_map)}
                return [batch_dict]

            except StopIteration:
                if self._auto_reset:
                    self.reset()
                raise

        def reset(self):
            self._pipeline.reset()
            self._counter = 0

except ImportError:
    DALI_AVAILABLE = False
    warnings.warn("INFO: NVIDIA DALI not available. Falling back to OpenCV-based implementation."
                  "For maximum performance, install with: "
                  "pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120")

# Image file extensions
IMG_EXTS = (
    '.png', '.jpg', '.jpeg', '.bmp', '.gif',
    '.tif', '.tiff', '.webp', '.jfif', '.avif',
    '.heif', '.heic'
)


class Dataset:
    """Base dataset class for simple tensor data"""
    def __init__(self, X, y, dtype=float32):
        assert len(X) == len(y), "Mismatched input and label lengths"
        self.X = Tensor(X, dtype=dtype)
        self.y = Tensor(y, dtype=dtype)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]
    
    def shuffle(self, seed: Optional[int] = None):
        indices = list(range(len(self)))
        rng = random.Random(seed) if seed is not None else random.Random()
        rng.shuffle(indices)
        self.X = self.X[indices]
        self.y = self.y[indices]
    
    def __repr__(self):
        return f"<Dataset: {len(self)} samples, dtype={self.X.data.dtype}>"
    
    def __str__(self):
        preview_x = self.X[:1]
        preview_y = self.y[:1]
        return (f"Dataset:\n"
                f"  Total samples: {len(self)}\n"
                f"  Input preview: {preview_x}\n"
                f"  Target preview: {preview_y}")


class ImageFolder(Dataset):
    """
    ImageFolder dataset class that handles image loading and preprocessing.
    All data loading parameters are now handled by the DataLoader.
    """
    def __init__(
        self,
        root: str,
        img_shape: tuple = None,          # (H, W)
        img_mode: str = "RGB",            # "RGB", "L", etc.
        img_normalize: bool = True,       # /255 -> float
        img_transform: callable = None,   # DALI pipeline or callable
        one_hot_targets: bool = True,     # Convert targets to one-hot encoding
        img_dtype=xp.float32,
        target_dtype=xp.int64,
        chw: bool = True,                 # return CxHxW if True, else HxWxC
    ):
        self.root = root
        self.img_shape = img_shape
        self.img_mode = img_mode
        self.img_normalize = img_normalize
        self.img_transform = img_transform
        self.one_hot_targets = one_hot_targets
        self.img_dtype = img_dtype
        self.target_dtype = target_dtype
        self.chw = chw

        self.images: List[str] = []
        self.targets: List[str] = []
        self.dont_use_mmap = is_on_network_drive(self.root)
        if self.dont_use_mmap:
            warnings.warn("ImageFolder dataset directory appears to be on a network drive / non-local storage. "
                          "Performance will be severely impacted. Consider copying data locally.", UserWarning)
        self._collect_paths()

        # Check if we have any images
        if len(self.images) == 0:
            raise ValueError(f"No images found in {root} with supported extensions: {IMG_EXTS}")

        # Stable class mapping
        self.target_names = sorted(set(self.targets))
        self.target_mapping = {name: i for i, name in enumerate(self.target_names)}
        self.num_classes = len(self.target_names)
        
        # Create one-hot mapping
        self.one_hot_mapping: Dict[int, np.ndarray] = {}
        for class_idx in range(self.num_classes):
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[class_idx] = 1.0
            self.one_hot_mapping[class_idx] = one_hot
        
        # Convert targets to numeric labels
        self.numeric_targets = [self.target_mapping[t] for t in self.targets]

        # print(f"ImageFolder initialized: {len(self)} samples, {self.num_classes} classes")

    def get_class_name(self, class_idx: int) -> str:
        """Get class name from class index"""
        if class_idx < 0 or class_idx >= len(self.target_names):
            raise ValueError(f"Class index {class_idx} out of range [0, {len(self.target_names)-1}]")
        return self.target_names[class_idx]
    
    def get_class_index(self, class_name: str) -> int:
        """Get class index from class name"""
        if class_name not in self.target_mapping:
            raise ValueError(f"Class name '{class_name}' not found in dataset")
        return self.target_mapping[class_name]
    
    def get_one_hot(self, class_idx: int) -> np.ndarray:
        """Get one-hot encoding from class index"""
        if class_idx not in self.one_hot_mapping:
            raise ValueError(f"Class index {class_idx} not found in one-hot mapping")
        return self.one_hot_mapping[class_idx]
    
    def get_class_from_one_hot(self, one_hot: np.ndarray) -> int:
        """Get class index from one-hot encoding"""
        if one_hot.shape != (self.num_classes,):
            raise ValueError(f"One-hot encoding must have shape ({self.num_classes},), got {one_hot.shape}")
        
        class_idx = np.argmax(one_hot)
        if one_hot[class_idx] != 1.0:
            raise ValueError("One-hot encoding must have exactly one element set to 1.0")
            
        return class_idx

    def _collect_paths(self):
        """Collect image paths and their class labels"""
        if not os.path.exists(self.root) or not os.path.isdir(self.root):
            raise ValueError(f"Root directory {self.root} does not exist or is not a directory")
        
        # Method 1: Class folders (ImageNet style)
        for class_name in os.listdir(self.root):
            class_path = os.path.join(self.root, class_name)
            if not os.path.isdir(class_path):
                continue
            
            for ext in IMG_EXTS:
                pattern = os.path.join(class_path, f"*{ext}")
                for img_path in glob.glob(pattern):
                    self.images.append(img_path)
                    self.targets.append(class_name)
        
        # Method 2: If no class folders found, walk directory tree
        if not self.images:
            for r, _, files in os.walk(self.root):
                for f in files:
                    if any(f.lower().endswith(ext) for ext in IMG_EXTS):
                        p = os.path.join(r, f)
                        cls = os.path.basename(os.path.dirname(p))
                        self.images.append(p)
                        self.targets.append(cls)


    def get_dali_pipeline(self, batch_size: int, shuffle: bool = True, 
                          device: str = "cpu", num_threads: int = 4, 
                          prefetch: int = 2, seed: int = 42):
        """Create optimized DALI pipeline for this dataset (only when DALI is available)"""
        if not DALI_AVAILABLE:
            return None
            
        if isinstance(self.img_transform, Pipeline):
            return self.img_transform

        is_gpu = device == "gpu"
        h, w = self.img_shape or (224, 224)
        @pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=0 if is_gpu else None, seed=seed,
                      prefetch_queue_depth=prefetch)
        def image_pipeline():
            images, labels = fn.readers.file(files=self.images, labels=self.numeric_targets, 
                                             random_shuffle=shuffle, name="Reader",
                                             initial_fill=4096, read_ahead=True,
                                             dont_use_mmap=self.dont_use_mmap, # Crucial for network storage
                                             )
            images = fn.decoders.image(images, device="mixed" if is_gpu else "cpu", output_type=types.RGB)
            images = fn.resize(images, resize_x=w, resize_y=h, interp_type=types.INTERP_LINEAR)
            
            if self.img_transform and callable(self.img_transform):
                images = self.img_transform(images)
            
            if self.img_normalize:
                images = fn.crop_mirror_normalize(
                    images, dtype=types.FLOAT, output_layout="CHW" if self.chw else "HWC",
                    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255]
                )
            else:
                if self.chw: images = fn.transpose(images, perm=[2, 0, 1])
            
            if self.one_hot_targets:
                labels = fn.one_hot(labels, num_classes=self.num_classes)

            # A DALI pipeline function should return the DataNode objects directly.
            return images, labels

        pipeline = image_pipeline()
        return pipeline

    def _apply_img_transform(self, arr: np.ndarray) -> np.ndarray:
        """Apply image transforms (OpenCV fallback path)"""
        if self.img_transform is None:
            return arr
            
        if isinstance(self.img_transform, Pipeline):
            warnings.warn("WARNING: DALI Pipeline transform provided but running in fallback numpy/OpenCV mode. "
                          "Cannot apply Pipeline to ndarray; skipping transform.", UserWarning)
            return arr
            
        try:
            out = self.img_transform(image=arr)
            if isinstance(out, dict) and "image" in out:
                return out["image"]
            return out
        except TypeError:
            pass

        try:
            return self.img_transform(arr)
        except Exception as e:
            warnings.warn(f"WARNING: img_transform callable raised an exception: {e}. Returning original image.", UserWarning)
            return arr

    def _load_image_opencv(self, path: str) -> np.ndarray:
        """Load image using OpenCV (fallback implementation)"""
        mode = (self.img_mode or "RGB").upper()
        if mode in ("L", "GRAY", "GREY", "GRAYSCALE"):
            flag = cv2.IMREAD_GRAYSCALE
        elif mode == "RGBA":
            flag = cv2.IMREAD_UNCHANGED
        else:
            flag = cv2.IMREAD_COLOR
        
        try:
            flag |= cv2.IMREAD_IGNORE_ORIENTATION
        except Exception:
            pass

        arr = cv2.imread(path, flag)
        if arr is None:
            raise ValueError(f"Failed to read image: {path}")

        if mode == "RGB" and arr.ndim == 3 and arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        elif mode == "RGBA" and arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)

        if self.img_shape is not None:
            h, w = self.img_shape
            arr = cv2.resize(arr, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
        
        if arr.ndim == 2:
            arr = arr[:, :, None]
        
        if self.img_transform:
            arr = self._apply_img_transform(arr)
        
        if self.chw and arr.ndim == 3:
            arr = np.transpose(arr, (2, 0, 1))
        
        if self.img_normalize:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
        
        return arr

    def __getitem__(self, idx: int):
        """Get single item - uses OpenCV fallback"""
        img_path = self.images[idx]
        target = self.numeric_targets[idx]
        image = self._load_image_opencv(img_path)
        
        if self.one_hot_targets:
            target = self.one_hot_mapping[target]
            target_dtype = float32
        else:
            target_dtype = self.target_dtype
        
        return Tensor(image, dtype=self.img_dtype), Tensor(target, dtype=target_dtype)

    def shuffle(self, seed: Optional[int] = None):
        """Shuffle the dataset"""
        rng = random.Random(seed) if seed is not None else random.Random()
        combined = list(zip(self.images, self.targets, self.numeric_targets))
        rng.shuffle(combined)
        self.images, self.targets, self.numeric_targets = [list(t) for t in zip(*combined)]

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __str__(self):
        return (f"ImageFolder(root='{self.root}', samples={len(self)}, "
                f"classes={self.num_classes})")

    def __repr__(self):
        shape = tuple(self[0][0].shape) if len(self) > 0 else None
        return (f"ImageFolder(root='{self.root}', samples={len(self)}, "
                f"classes={self.num_classes}, shape={shape})")


class DataLoader:
    """
    DataLoader that handles all data loading parameters, with an efficient DALI backend
    and a multithreaded OpenCV fallback.
    """
    def __init__(
        self,
        dataset: Union[ImageFolder, Dataset],
        batch_size: int = 32,
        shuffle: bool = True,
        device: str = "gpu",
        num_workers: int = None,
        prefetch_batches: int = 2,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = "gpu" if device in ["gpu", "cuda"] else "cpu"
        self.num_workers = os.cpu_count() if num_workers is None else max(0, int(num_workers))
        self.prefetch_batches = max(0, int(prefetch_batches))
        self.drop_last = drop_last
        self.seed = seed
        
        self._pipeline = None
        self._dali_iter = None
        self._executor: Optional[ThreadPoolExecutor] = None
        
        if DALI_AVAILABLE:
            try:
                if self.device == "gpu": PreallocateDeviceMemory(int(0.5 * 1024**3), 0)
                PreallocatePinnedMemory(int(0.25 * 1024**3))
            except Exception as e: warnings.warn(f"WARNING: DALI memory preallocation failed: {e}")
        
        self.use_dali = DALI_AVAILABLE and isinstance(dataset, ImageFolder)
        if self.use_dali:
            self._init_dali_pipeline()
        
        if self.device == "gpu" and not self.use_dali:
            warnings.warn("WARNING: GPU device requested but DALI not available. Using CPU fallback.")
            self.device = "cpu"

    def _init_dali_pipeline(self):
        if not self.use_dali: return
        self._pipeline = self.dataset.get_dali_pipeline(
            batch_size=self.batch_size, shuffle=self.shuffle, device=self.device,
            num_threads=self.num_workers, prefetch=self.prefetch_batches, seed=self.seed)
        if self._pipeline:
            policy = LastBatchPolicy.DROP if self.drop_last else LastBatchPolicy.PARTIAL
            self._dali_iter = DALIGenericIterator(self._pipeline, ["images", "labels"], policy, reader_name="Reader")
            # print(f"DALI pipeline initialized: batch_size={self.batch_size}, device={self.device}, num_threads={self.num_workers}")

    def __len__(self):
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        if self.use_dali and self._dali_iter:
            self._dali_iter.reset()
            return self._dali_iterator()
        return self._regular_iterator()

    def reset(self):
        """
        Resets the internal iterator, particularly for the DALI pipeline.
        This allows re-iteration over the dataset from the beginning without
        recreating the DataLoader.
        """
        if self.use_dali and self._dali_iter:
            self._dali_iter.reset()
        # For the regular iterator, a new one is created on each __iter__ call,
        # so an explicit reset is not strictly necessary in the same way.

    
    def _dali_iterator(self):
        for data in self._dali_iter:
            # The pipeline outputs DALI Tensors which are extracted from the output dictionary.
            images_tensor = data[0]["images"]  # This is a TensorListGPU containing the batch
            labels_tensor = data[0]["labels"]  # This is a TensorListGPU containing the batch
            
            # Convert DALI TensorList to proper arrays
            if self.device == 'gpu':
                import cupy as cp
                # For GPU TensorList, convert to cupy arrays
                images_array = cp.asarray(images_tensor.as_tensor())
                labels_array = cp.asarray(labels_tensor.as_tensor())
                
                X = Tensor(images_array, dtype=self.dataset.img_dtype)
                y = Tensor(labels_array, dtype=self.dataset.target_dtype if not self.dataset.one_hot_targets else float32)
            else: # CPU device
                # For CPU TensorList, convert to numpy array
                X = Tensor(images_tensor.as_array(), dtype=self.dataset.img_dtype)
                y = Tensor(labels_tensor.as_array(), dtype=self.dataset.target_dtype if not self.dataset.one_hot_targets else float32)
    
            yield X, y
            
    def _regular_iterator(self):
        """Fallback iterator with threading and prefetching."""
        batches = list(self._batch_indices())
        window = deque()
        next_to_submit = 0
        total = len(batches)

        for _ in range(min(self.prefetch_batches, total)):
            futs = self._schedule_batch(batches[next_to_submit])
            window.append(futs)
            next_to_submit += 1

        for _ in range(total):
            if not window:
                futs = self._schedule_batch(batches[next_to_submit])
                window.append(futs)
                next_to_submit += 1

            futs = window.popleft()

            if next_to_submit < total:
                next_futs = self._schedule_batch(batches[next_to_submit])
                window.append(next_futs)
                next_to_submit += 1

            yield self._gather_batch(futs)

    def _batch_indices(self):
        n = len(self.dataset)
        order = list(range(n))
        if self.shuffle:
            random.Random(self.seed).shuffle(order)
        
        limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
        for start in range(0, limit, self.batch_size):
            end = min(start + self.batch_size, n)
            yield order[start:end]

    def _ensure_executor(self):
        if self.num_workers > 0 and self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.num_workers)

    def _schedule_batch(self, idxs):
        if self.num_workers > 0:
            self._ensure_executor()
            return [self._executor.submit(self.dataset.__getitem__, i) for i in idxs]
        return [self.dataset[i] for i in idxs]

    def _gather_batch(self, futures_or_results):
        if self.num_workers > 0:
            batch = [f.result() for f in futures_or_results]
        else:
            batch = futures_or_results
        
        Xs, ys = zip(*batch)
        return Tensor(xp.stack([x.data for x in Xs])), Tensor(xp.stack([y.data for y in ys]))

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


def create_dali_transforms(
    device: str = "gpu",
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    horizontal_flip_prob: float = 0.0,
    vertical_flip_prob: float = 0.0,
    rotation_angle: float = 0.0
):
    """
    Create common DALI augmentation transforms.
    Returns None if DALI is not available.
    """
    if not DALI_AVAILABLE:
        warnings.warn("WARNING: DALI not available. Transform creation skipped.")
        return None
        
    def apply_transforms(images):
        if brightness or contrast or saturation or hue:
            images = fn.color_twist(
                images, device=device,
                brightness=fn.random.uniform(range=[-brightness, brightness]),
                contrast=fn.random.uniform(range=[1-contrast, 1+contrast]),
                saturation=fn.random.uniform(range=[1-saturation, 1+saturation]),
                hue=fn.random.uniform(range=[-hue, hue]),
            )
        if horizontal_flip_prob > 0.0:
            images = fn.flip(images, device=device, horizontal=fn.random.coin_flip(probability=horizontal_flip_prob))
        if vertical_flip_prob > 0.0:
            images = fn.flip(images, device=device, vertical=fn.random.coin_flip(probability=vertical_flip_prob))
        if rotation_angle != 0.0:
            images = fn.rotate(
                images, device=device, angle=fn.random.uniform(range=[-rotation_angle, rotation_angle]),
                keep_size=True, fill_value=0
            )
        return images
    
    return apply_transforms




def is_on_network_drive(path_to_check: str) -> bool:
    """
    Detects if the given path is on a network-mounted filesystem.
    """
    if not os.path.exists(path_to_check):
        raise FileNotFoundError(f"Path does not exist: {path_to_check}")
    NETWORK_FS_TYPES = [
        "nfs", "nfs4", "nfsd", "cifs", "smbfs", "smb", "smb2", "smb3",
        "fuse.sshfs", "fuse.gcsfuse", "fuse.s3fs"
    ]
    target_path = os.path.abspath(path_to_check)
    partitions = psutil.disk_partitions(all=True)
    # Find the most specific mount point for the path
    longest_match = None
    for p in partitions:
        if target_path.startswith(p.mountpoint):
            if longest_match is None or len(p.mountpoint) > len(longest_match.mountpoint):
                longest_match = p
    if longest_match is None:
        # Could not find any mount, assume local
        return False
    fs_type = longest_match.fstype.lower()
    if fs_type in NETWORK_FS_TYPES:
        return True
    # Overlay is usually local (Docker /tmp), so treat it as local
    return False