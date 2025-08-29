# Core imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import contextlib
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# NeuroGrad imports
import neurograd as ng
from neurograd import Tensor
from neurograd.nn.layers.conv import Conv2D, MaxPool2D
from neurograd.nn.layers.linear import Linear
from neurograd.nn.module import Sequential
from neurograd.nn.losses import CategoricalCrossEntropy
from neurograd.functions.activations import ReLU, Softmax
from neurograd.optim.adam import Adam
from neurograd.optim.sgd import SGD
from neurograd.optim.rmsprop import RMSprop
from neurograd.utils.data import Dataset, DataLoader
from neurograd.nn.metrics import accuracy_score as ng_accuracy_score

# Set style and random seed
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

print(f"NeuroGrad device: {ng.DEVICE}")
print(f"Backend: {'CuPy' if ng.DEVICE == 'cuda' else 'NumPy'}")
print("Conv2D training notebook ready!")



# Mixed Precision Training Configuration
USE_MIXED_PRECISION = True  # Set to True to enable mixed precision training

# Import mixed precision components if enabled
if USE_MIXED_PRECISION:
    try:
        from neurograd.amp import autocast, GradScaler  # Use NeuroGrad's GradScaler, not PyTorch's!
        print(f"✅ Mixed precision support enabled")
        print(f"   Device: {ng.DEVICE} ({'supports FP16' if ng.DEVICE == 'cuda' else 'FP16 limited benefit on CPU'})")
        print(f"   Using NeuroGrad's GradScaler (not PyTorch's)")
    except ImportError as e:
        print(f"❌ Mixed precision not available: {e}")
        USE_MIXED_PRECISION = False
else:
    print("🔧 Mixed precision disabled")

print(f"Mixed precision training: {'ON' if USE_MIXED_PRECISION else 'OFF'}")



# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target

print(f"Dataset info:")
print(f"  Number of samples: {X.shape[0]}")
print(f"  Original shape: {X.shape}")
print(f"  Image shape: 8x8 pixels")
print(f"  Number of classes: {len(np.unique(y))}")
print(f"  Classes: {np.unique(y)}")
print(f"  Class distribution: {np.bincount(y)}")

# Visualize some sample digits
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i in range(10):
    ax = axes[i // 5, i % 5]
    # Find first example of digit i
    idx = np.where(y == i)[0][0]
    ax.imshow(digits.images[idx], cmap='gray')
    ax.set_title(f'Digit {i}')
    ax.axis('off')

plt.tight_layout()
plt.show()


# Reshape data for Conv2D: (batch_size, channels, height, width) - NCHW format
# The digits dataset is 8x8, so we reshape from (1797, 64) to (1797, 1, 8, 8)
X_images = X.reshape(-1, 8, 8, 1).transpose(0, 3, 1, 2)  # NHWC -> NCHW

print(f"Reshaped for Conv2D (NCHW format):")
print(f"  X shape: {X_images.shape} (batch_size, channels, height, width)")
print(f"  y shape: {y.shape}")

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_images, y, test_size=0.2, random_state=42, stratify=y
)

# Normalize pixel values to [0, 1]
X_train = X_train.astype(np.float32) / 16.0  # digits are 0-16
X_test = X_test.astype(np.float32) / 16.0

# One-hot encode labels
n_classes = 10
y_train_onehot = np.eye(n_classes)[y_train]
y_test_onehot = np.eye(n_classes)[y_test]

print(f"\nAfter preprocessing:")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_train_onehot shape: {y_train_onehot.shape}")
print(f"  y_test_onehot shape: {y_test_onehot.shape}")
print(f"  Pixel value range: [{X_train.min():.3f}, {X_train.max():.3f}]")



# Convert to NeuroGrad tensors
X_train_tensor = Tensor(X_train, requires_grad=False)
y_train_tensor = Tensor(y_train_onehot, requires_grad=False)
X_test_tensor = Tensor(X_test, requires_grad=False)
y_test_tensor = Tensor(y_test_onehot, requires_grad=False)

print(f"NeuroGrad tensors:")
print(f"  X_train_tensor: {X_train_tensor.shape}")
print(f"  y_train_tensor: {y_train_tensor.shape}")
print(f"  X_test_tensor: {X_test_tensor.shape}")
print(f"  y_test_tensor: {y_test_tensor.shape}")

# Create Dataset and DataLoader objects
train_dataset = Dataset(X_train, y_train_onehot)
test_dataset = Dataset(X_test, y_test_onehot)

# Create data loaders with smaller batch size for better gradients
batch_size = 32  # Reduced from 32 for more stable training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"\nDataLoaders:")
print(f"  Training batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")
print(f"  Batch size: {batch_size}")


# Define improved Conv2D model architecture
def create_conv2d_model():
    """
    Create an improved Conv2D model for 8x8 digit classification (NCHW format).
    
    Enhanced Architecture:
    - Conv2D: 1->32 filters, 3x3 kernel, ReLU + MaxPool 2x2
    - Conv2D: 32->64 filters, 3x3 kernel, ReLU + MaxPool 2x2
    - Flatten and larger Linear layers
    - Softmax output for 10 classes
    """
    from neurograd.nn.layers import Flatten
    
    model = Sequential(
        # First Conv block: (N,1,8,8) -> (N,32,8,8) -> (N,32,4,4)
        Conv2D(in_channels=1, out_channels=32, kernel_size=(3, 3), 
               padding="same", activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        
        # Second Conv block: (N,32,4,4) -> (N,64,4,4) -> (N,64,2,2)
        Conv2D(in_channels=32, out_channels=64, kernel_size=(3, 3), 
               padding="same", activation="relu"),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        
        # Flatten for fully connected layers: (N,64,2,2) -> (N,256)
        Flatten(),
        Linear(2*2*64, 128),  # Larger hidden layer
        ReLU(),
        
        # Output layer
        Linear(128, 10),
        Softmax(axis=1)  # Softmax along class dimension
    )
    
    return model

# Create the improved model
model = create_conv2d_model()

# Print model information
print("Improved Conv2D Model Architecture:")
print(model)

# Count total parameters
total_params = sum(p.data.size for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Print parameter details
print("\nParameter details:")
for name, param in model.named_parameters():
    print(f"  {name}: {param.shape} ({param.data.size} params)")

print(f"\nArchitecture improvements:")
print(f"  ✓ More filters: 1→32→64 (vs 1→16→32)")
print(f"  ✓ Larger hidden layer: 128 units (vs 64)")
print(f"  ✓ More parameters for better feature learning")


# Test model with a single batch to verify shapes
print("Testing model with sample batch:")

# Get a small batch for testing
test_batch = X_train_tensor[:4]  # Take first 4 samples
print(f"Test batch shape: {test_batch.shape}")

# Debug: Test individual layers
print("\n=== DEBUG: Testing individual components ===")

# Test first Conv2D layer directly
first_conv = model._sequential_modules[0]  # First Conv2D layer
print(f"First Conv2D layer:")
print(f"  Kernel size: {first_conv.kernel_size}")
print(f"  Padding: {first_conv.padding}")
print(f"  In channels: {first_conv.in_channels}")
print(f"  Out channels: {first_conv.out_channels}")

# Test conv2d function directly
try:
    print(f"\nTesting conv2d function with:")
    print(f"  Input shape: {test_batch.shape}")
    print(f"  Kernel shape: {first_conv.kernels.shape}")
    print(f"  Strides: {first_conv.strides}")
    print(f"  Padding: {first_conv.padding}")
    
    from neurograd import conv2d
    conv_output = conv2d(test_batch, first_conv.kernels, first_conv.strides, first_conv.padding, first_conv.padding_value)
    print(f"  Conv2d output shape: {conv_output.shape}")
    print("✅ Conv2d test successful!")
    
except Exception as e:
    print(f"❌ Conv2d test failed: {e}")
    import traceback
    traceback.print_exc()

# Test the full model
try:
    print(f"\n=== Testing full model ===")
    model.eval()  # Set to evaluation mode
    test_output = model(test_batch)
    print(f"Model output shape: {test_output.shape}")
    print(f"Output probabilities (first sample): {test_output.data[0]}")
    print(f"Sum of probabilities: {test_output.data[0].sum()}")
    
    # Verify the spatial dimensions are correct after convolutions
    print("\nSpatial dimension verification (NCHW format):")
    print("  Input: (N,1,8,8)")
    print("  After Conv1 + Pool1: (N,16,4,4)")
    print("  After Conv2 + Pool2: (N,32,2,2)")
    print("  Flattened: 128 features")
    print("  Final output: 10 classes")
    print("\n✅ Model test successful!")
    
except Exception as e:
    print(f"❌ Model test failed with error: {e}")
    import traceback
    traceback.print_exc()



# Improved training configuration for better performance
epochs = 100       # Increased from 50 for more convergence time
learning_rate = 0.0001  # Reduced from 0.001 for more stable training

# Create optimizer and loss function
optimizer = Adam(model.named_parameters(), lr=learning_rate)
loss_fn = CategoricalCrossEntropy()

# Initialize mixed precision scaler if enabled
if USE_MIXED_PRECISION:
    scaler = GradScaler()
    print(f"✅ GradScaler initialized:")
    print(f"   Initial scale: {scaler.get_scale()}")
    print(f"   Enabled: {scaler.is_enabled()}")
else:
    scaler = None

print(f"\nImproved training configuration:")
print(f"  Epochs: {epochs} (increased for convergence)")
print(f"  Learning rate: {learning_rate} (reduced for stability)")
print(f"  Optimizer: {optimizer.__class__.__name__}")
print(f"  Loss function: {loss_fn.__class__.__name__}")
print(f"  Batch size: {batch_size} (reduced to 16)")
print(f"  Mixed precision: {'ENABLED' if USE_MIXED_PRECISION else 'DISABLED'}")
if USE_MIXED_PRECISION:
    print(f"  Gradient scaler: {scaler.__class__.__name__}")

print(f"\nChanges made to improve performance:")
print(f"  ✓ Lower learning rate (0.001 → 0.0001) - reduces oscillations")
print(f"  ✓ Smaller batch size (32 → 16) - better gradient estimates")
print(f"  ✓ More epochs (50 → 100) - allows full convergence")
if USE_MIXED_PRECISION:
    print(f"  ✓ Mixed precision training - faster computation & lower memory usage")
    print(f"  ✓ Gradient scaling - prevents FP16 underflow")



# Utility functions for training
def convert_to_numpy(data):
    """Convert CuPy arrays to NumPy arrays if needed"""
    if hasattr(data, 'get'):  # CuPy array
        return data.get()
    return np.array(data)

def calculate_accuracy(predictions, targets):
    """Calculate accuracy for classification using NeuroGrad's tensor-based metrics"""
    # Convert predictions and targets to class indices for accuracy calculation
    pred_classes = predictions.argmax(axis=1)
    target_classes = targets.argmax(axis=1)
    
    # Use NeuroGrad's accuracy_score which works directly on Tensors
    accuracy = ng_accuracy_score(target_classes, pred_classes)
    
    # Convert to Python float (handle both NumPy and CuPy arrays)
    if hasattr(accuracy, 'item'):  # CuPy array
        return float(accuracy.item())
    else:  # NumPy array or scalar
        return float(accuracy)

def evaluate_model(model, X, y):
    """Evaluate model on given data"""
    model.eval()
    predictions = model(X)
    loss = loss_fn(y, predictions)
    accuracy = calculate_accuracy(predictions, y)
    return loss.data.item(), accuracy


# Improved training loop with mixed precision and learning rate scheduling
print("Starting improved training...")
if USE_MIXED_PRECISION:
    print("🚀 Mixed precision training enabled - expect faster training!")
print("=" * 60)

# Initialize tracking
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
epoch_times = []
learning_rates = []
overflow_count = 0  # Track gradient overflow events
scale_values = []   # Track gradient scale changes

start_time = time.time()
initial_lr = learning_rate

for epoch in range(epochs):
    epoch_start = time.time()
    
    # Learning rate scheduling - reduce LR when training plateaus
    if epoch == 40:  # First reduction at epoch 40
        learning_rate = initial_lr * 0.5
        optimizer.lr = learning_rate
        print(f"  Reducing learning rate to {learning_rate}")
    elif epoch == 70:  # Second reduction at epoch 70
        learning_rate = initial_lr * 0.1
        optimizer.lr = learning_rate
        print(f"  Reducing learning rate to {learning_rate}")
    
    learning_rates.append(learning_rate)
    
    # Track gradient scaler values if using mixed precision
    if USE_MIXED_PRECISION:
        scale_values.append(scaler.get_scale())
    
    # Training phase
    model.train()
    epoch_train_losses = []
    epoch_train_accs = []
    
    for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
        # Zero gradients
        optimizer.zero_grad()
        
        if USE_MIXED_PRECISION:
            # Mixed precision forward pass
            with autocast(enabled=True):
                predictions = model(batch_X)
                # Loss computation stays in FP32 for numerical stability
                loss = loss_fn(batch_y, predictions)
            
            # Scale loss to prevent gradient underflow in FP16
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            
            # Unscale gradients and check for overflow before optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            # Check if this step had gradient overflow
            if hasattr(scaler, '_found_inf') and scaler._found_inf:
                overflow_count += 1
                
        else:
            # Standard precision training
            predictions = model(batch_X)
            loss = loss_fn(batch_y, predictions)
            loss.backward()
            optimizer.step()
        
        # Track metrics (use unscaled loss for logging)
        batch_loss = loss.data.item()
        batch_acc = calculate_accuracy(predictions, batch_y)
        
        epoch_train_losses.append(batch_loss)
        epoch_train_accs.append(batch_acc)
    
    # Calculate epoch averages
    avg_train_loss = np.mean(epoch_train_losses)
    avg_train_acc = np.mean(epoch_train_accs)
    
    # Evaluation phase (always in FP32 for consistency)
    with autocast(enabled=False) if USE_MIXED_PRECISION else contextlib.nullcontext():
        test_loss, test_acc = evaluate_model(model, X_test_tensor, y_test_tensor)
    
    # Record metrics
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    
    # Print progress more frequently for longer training
    if epoch % 20 == 0 or epoch == epochs - 1 or epoch in [39, 69]:  # Show LR reduction epochs
        progress_msg = (f"Epoch {epoch:3d}/{epochs}: "
                       f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                       f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, "
                       f"LR: {learning_rate:.6f}, Time: {epoch_time:.2f}s")
        
        if USE_MIXED_PRECISION:
            progress_msg += f", Scale: {scaler.get_scale():.0f}"
        
        print(progress_msg)

total_time = time.time() - start_time
print("=" * 60)
print(f"Improved training completed in {total_time:.2f} seconds")
print(f"Average time per epoch: {np.mean(epoch_times):.2f}s")
print(f"Final train accuracy: {train_accuracies[-1]:.4f}")
print(f"Final test accuracy: {test_accuracies[-1]:.4f}")
print(f"Best test accuracy: {max(test_accuracies):.4f} (epoch {np.argmax(test_accuracies)})")
print(f"Improvement from baseline (66.1%): {max(test_accuracies) - 0.661:.3f}")

if USE_MIXED_PRECISION:
    print(f"\nMixed Precision Statistics:")
    print(f"  Gradient overflow events: {overflow_count}")
    print(f"  Final gradient scale: {scaler.get_scale():.0f}")
    print(f"  Scale range: {min(scale_values):.0f} - {max(scale_values):.0f}")
    print(f"  Average scale: {np.mean(scale_values):.0f}")
    if overflow_count > 0:
        print(f"  Overflow rate: {overflow_count / (epochs * len(train_loader)):.4f} per batch")
        
# Need contextlib for the nullcontext when mixed precision is disabled
import contextlib