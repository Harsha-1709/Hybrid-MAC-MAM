import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button
import numpy as np
from models.network import HybridNN
import random
import os

# CIFAR-10 classes
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model_path = "hybrid_model.pth"
    model = HybridNN().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model weights from {model_path}")
        except RuntimeError as e:
            print(f"Warning: Architecture mismatch for {model_path}. Could not load weights.")
            print("This happens because the model was upgraded to a CNN. Please run train.py to generate new weights.")
    else:
        print(f"Warning: {model_path} not found. Using untrained model.")
        print("Please run train.py first to train the model.")
        
    model.eval()

    # Load Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    print("Loading test dataset...")
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    display_dataset = datasets.CIFAR10('./data', train=False, download=True)
    
    print("Dataset loaded. Initializing UI...")

    # Set up the matplotlib figure using GridSpec for a nice layout
    # Left side: 4x4 grid of selectible images
    # Right side: Top for zoomed image + prediction, Bottom for probability bar chart
    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title('Hybrid MAC-MAM CNN CIFAR-10 Grid Classifier')
    
    gs = gridspec.GridSpec(4, 6, figure=fig)
    
    # Grid axes (left 4 columns)
    grid_axs = []
    for i in range(4):
        for j in range(4):
            ax = fig.add_subplot(gs[i, j])
            ax.axis('off')
            grid_axs.append(ax)
            
    # Detail axes (right 2 columns)
    ax_detail_img = fig.add_subplot(gs[0:2, 4:6])
    ax_detail_bar = fig.add_subplot(gs[2:4, 4:6])
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
    
    # Store currently displayed indices to map clicks
    current_indices = []
    
    def load_new_grid(event=None):
        nonlocal current_indices
        current_indices = random.sample(range(len(test_dataset)), 16)
        
        for idx, ax in zip(current_indices, grid_axs):
            ax.clear()
            img_display, _ = display_dataset[idx]
            ax.imshow(img_display)
            ax.axis('off')
            # Store the index in the axes object for click retrieval
            ax.dataset_idx = idx
            
        fig.canvas.draw()
        
        # Select the first image in the grid by default
        update_detail(current_indices[0])

    def update_detail(dataset_idx):
        ax_detail_img.clear()
        ax_detail_bar.clear()
        
        img_tensor, label_idx = test_dataset[dataset_idx]
        actual_label = CLASSES[label_idx]
        img_display, _ = display_dataset[dataset_idx]
        
        # Inference
        with torch.no_grad():
            img_batch = img_tensor.unsqueeze(0).to(device)
            output = model(img_batch)
            probs = F.softmax(output, dim=1).squeeze().cpu().numpy()
            
            pred_idx = int(np.argmax(probs))
            pred_label = CLASSES[pred_idx]
            max_prob = float(probs[pred_idx])
            
        # Display Image
        ax_detail_img.imshow(img_display)
        ax_detail_img.axis('off')
        
        is_correct = pred_idx == label_idx
        color = 'green' if is_correct else 'red'
        title = f"Actual: {actual_label} | Pred: {pred_label}"
        ax_detail_img.set_title(title, color=color, fontweight='bold', fontsize=14)
        
        # Display Bar Chart showing only the predicted class with actual probability
        y_pos = np.array([0])
        # Make the bar thinner by passing height=0.3
        bars = ax_detail_bar.barh(y_pos, [max_prob], height=0.3, align='center', color='lightgray')
        
        if pred_idx == label_idx:
            bars[0].set_color('green')
        else:
            bars[0].set_color('red')
            
        ax_detail_bar.set_yticks(y_pos, labels=[pred_label])
        ax_detail_bar.invert_yaxis()
        ax_detail_bar.set_xlabel('Probability')
        ax_detail_bar.set_xlim([0, 1.0])
        ax_detail_bar.set_ylim([-1, 1]) # Adds padding above and below the single bar
        ax_detail_bar.spines['top'].set_visible(False)
        ax_detail_bar.spines['right'].set_visible(False)
        
        fig.canvas.draw()

    def on_click(event):
        # Check if the click was in one of the grid axes
        for ax in grid_axs:
            if event.inaxes == ax:
                update_detail(ax.dataset_idx)
                break

    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Initial grid load
    load_new_grid()

    # Add button to refresh grid
    ax_button = plt.axes([0.15, 0.02, 0.2, 0.05])
    btn = Button(ax_button, 'Refresh Grid')
    btn.on_clicked(load_new_grid)

    plt.show()

if __name__ == '__main__':
    main()
