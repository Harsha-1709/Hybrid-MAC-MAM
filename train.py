import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
from models.network import HybridNN, BaselineNN
from tqdm import tqdm
import argparse

class LiveVisualizer:
    def __init__(self):
        plt.ion()
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.canvas.manager.set_window_title('Training Process Visualizer')
        
        self.baseline_losses = []
        self.hybrid_losses = []
        
        self.epochs = []
        self.baseline_accs = []
        self.hybrid_accs = []
        
        self.alpha_history = []
        
        self._setup_plots()
        
    def _setup_plots(self):
        for ax in self.axs:
            ax.clear()
        
        self.axs[0].set_title('Training Loss (Moving Avg)')
        self.axs[0].set_xlabel('Batches')
        self.axs[0].set_ylabel('Loss')
        
        self.axs[1].set_title('Test Accuracy (%)')
        self.axs[1].set_xlabel('Epoch')
        self.axs[1].set_ylabel('Accuracy')
        self.axs[1].set_xlim([0.5, 3.5])
        self.axs[1].set_ylim([80, 100])
        
        self.axs[2].set_title('Hybrid Alphas (MAC/MAM Balance)')
        self.axs[2].set_xlabel('Batches')
        self.axs[2].set_ylabel('Alpha Value (1.0=MAC, 0.0=MAM)')
        
        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def smooth(self, scalars, weight=0.9):
        if len(scalars) == 0:
            return []
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def update_loss(self, baseline_loss=None, hybrid_loss=None):
        if baseline_loss is not None:
            self.baseline_losses.append(baseline_loss)
        if hybrid_loss is not None:
            self.hybrid_losses.append(hybrid_loss)
            
        if len(self.baseline_losses) % 50 == 0 or len(self.hybrid_losses) % 50 == 0:
            self.axs[0].clear()
            self.axs[0].set_title('Training Loss (Moving Avg)')
            self.axs[0].set_xlabel('Batches')
            self.axs[0].set_ylabel('Loss')
            
            if len(self.baseline_losses) > 0:
                self.axs[0].plot(self.smooth(self.baseline_losses), label='Baseline', color='blue')
            if len(self.hybrid_losses) > 0:
                self.axs[0].plot(self.smooth(self.hybrid_losses), label='Hybrid', color='orange')
            self.axs[0].legend(loc='upper right')
            
            with torch.no_grad():
                plt.draw()
                plt.pause(0.001)

    def update_alphas(self, alphas):
        self.alpha_history.append(alphas)
        if len(self.alpha_history) % 50 == 0:
            self.axs[2].clear()
            self.axs[2].set_title('Hybrid Alphas (MAC/MAM Balance)')
            self.axs[2].set_xlabel('Batches')
            self.axs[2].set_ylabel('Alpha Value (1.0=MAC, 0.0=MAM)')
            self.axs[2].set_ylim([-0.05, 1.05])
            
            history_np = np.array(self.alpha_history)
            for i in range(history_np.shape[1]):
                self.axs[2].plot(history_np[:, i], label=f'Layer {i+1}')
            self.axs[2].legend(loc='upper right')
            
            with torch.no_grad():
                plt.draw()
                plt.pause(0.001)

    def update_accuracy(self, epoch, baseline_acc, hybrid_acc):
        self.epochs.append(epoch)
        self.baseline_accs.append(baseline_acc)
        self.hybrid_accs.append(hybrid_acc)
        
        self.axs[1].clear()
        self.axs[1].set_title('Test Accuracy (%)')
        self.axs[1].set_xlabel('Epoch')
        self.axs[1].set_ylabel('Accuracy')
        self.axs[1].set_xlim([0.5, max(3.5, epoch + 0.5)])
        
        self.axs[1].plot(self.epochs, self.baseline_accs, 'o-', label='Baseline', color='blue')
        self.axs[1].plot(self.epochs, self.hybrid_accs, 'o-', label='Hybrid', color='orange')
        self.axs[1].legend(loc='lower right')
        
        with torch.no_grad():
            plt.draw()
            plt.pause(0.001)

    def finish(self):
        plt.ioff()
        plt.show()

def train(model, device, train_loader, optimizer, criterion, epoch, visualizer, is_hybrid=False):
    model.train()
    total_loss = 0
    correct = 0
    
    desc = f"Epoch {epoch} [{'Hybrid' if is_hybrid else 'Baseline'}]"
    pbar = tqdm(train_loader, desc=desc, leave=False)
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        if is_hybrid:
            visualizer.update_loss(hybrid_loss=loss.item())
            
            # Record alphas
            alphas = []
            for name, module in model.named_modules():
                if hasattr(module, 'alpha_logits'):
                    # alpha_logits might be a tensor of shape [out_channels, 1, 1], take the mean for visualization
                    mean_alpha = torch.sigmoid(module.alpha_logits).mean().item()
                    alphas.append(mean_alpha)
            if alphas:
                visualizer.update_alphas(alphas)
        else:
            visualizer.update_loss(baseline_loss=loss.item())
            
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(train_loader)
    acc = 100. * correct / len(train_loader.dataset)
    return avg_loss, acc

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    return test_loss, acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fast', action='store_true', help='Run on a small subset for quick testing')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    if args.fast:
        print("Running in fast mode with small dataset subset...")
        train_dataset = torch.utils.data.Subset(train_dataset, range(100))
        test_dataset = torch.utils.data.Subset(test_dataset, range(100))
        epochs_total = 1
    else:
        epochs_total = 10

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    baseline_model = BaselineNN().to(device)
    optimizer_b = optim.Adam(baseline_model.parameters(), lr=0.001)
    
    hybrid_model = HybridNN().to(device)
    params = [
        {'params': [p for n, p in hybrid_model.named_parameters() if 'alpha' not in n], 'lr': 0.001},
        {'params': [p for n, p in hybrid_model.named_parameters() if 'alpha' in n], 'lr': 0.01}
    ]
    optimizer_h = optim.Adam(params)
    
    criterion = nn.CrossEntropyLoss()
    
    # Initialize Visualizer Output Window
    visualizer = LiveVisualizer()
    print("Visualizer window initialized.")
    
    epochs = epochs_total
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch} ---")
        
        # Train & Test Baseline
        train_loss_b, train_acc_b = train(baseline_model, device, train_loader, optimizer_b, criterion, epoch, visualizer, is_hybrid=False)
        test_loss_b, test_acc_b = test(baseline_model, device, test_loader, criterion)
        print(f"[Baseline] Train Loss: {train_loss_b:.4f}, Acc: {train_acc_b:.2f}% | Test Loss: {test_loss_b:.4f}, Acc: {test_acc_b:.2f}%")
        
        # Train & Test Hybrid
        train_loss_h, train_acc_h = train(hybrid_model, device, train_loader, optimizer_h, criterion, epoch, visualizer, is_hybrid=True)
        test_loss_h, test_acc_h = test(hybrid_model, device, test_loader, criterion)
        print(f"[Hybrid]   Train Loss: {train_loss_h:.4f}, Acc: {train_acc_h:.2f}% | Test Loss: {test_loss_h:.4f}, Acc: {test_acc_h:.2f}%")
        
        # Update epoch-level graph
        visualizer.update_accuracy(epoch, test_acc_b, test_acc_h)
        
    torch.save(baseline_model.state_dict(), "baseline_model.pth")
    torch.save(hybrid_model.state_dict(), "hybrid_model.pth")
    print("\nTraining complete. Models saved.")
    print("Close the visualizer window to exit the script.")
    visualizer.finish()
