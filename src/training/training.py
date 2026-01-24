import time
import os
import copy
import torch
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2


def visualize_attention_map(image, rollout_map, mask_gt, config_name, idx, output_dir='./results/heatmaps'):
    """
    Partie 4b: 可视化注意力热力图并与原图、Mask 对比
    Args:
        image: 原始图像 [3, H, W] (Tensor)
        rollout_map: calculate_rollout 算出的热力图 [H, W] (Tensor)
        mask_gt: 真实的分割图 [3, H, W] (Tensor)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 转换图像格式以便绘图 (C,H,W) -> (H,W,C)
    img = image.permute(1, 2, 0).cpu().numpy()
    # 归一化到 0-1
    img = (img - img.min()) / (img.max() - img.min())

    # 处理 rollout_map (热力图)
    heatmap = rollout_map.cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化

    # 生成彩色热力图 (使用 OpenCV 的 JET 映射)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) / 255.0

    # 叠加图：原图 * 0.6 + 热力图 * 0.4
    overlay = img * 0.6 + heatmap_color * 0.4

    # 绘图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")

    axes[1].imshow(mask_gt.permute(1, 2, 0).cpu().numpy())
    axes[1].set_title("Ground Truth Mask")

    axes[2].imshow(heatmap, cmap='jet')
    axes[2].set_title("Attention Heatmap (Rollout)")

    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (Explanation)")

    for ax in axes: ax.axis('off')

    plt.suptitle(f"Config {config_name} - Sample {idx}", fontsize=15)

    output_path = os.path.join(output_dir, f'heatmap_{config_name}_{idx}.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def train_model_crossvit(model, dataloaders, criterion, optimizer, scheduler, config, device, num_epochs=25,lambda_iou=0.5):
    """
    Entraîne le modèle CrossViT avec routage des inputs selon la configuration.
    
    Args:
        model: Modèle CrossViT
        dataloaders: dict avec 'train' et 'val' DataLoaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        config (str): 'A', 'B', 'C1', ou 'C2' pour diriger les inputs
        device: torch device
        num_epochs: Nombre d'epochs
    
    Returns:
        model: Meilleur modèle entraîné
        history: Dict avec les métriques
    """
    since = time.time()

    os.makedirs("checkpoints", exist_ok=True)
    best_model_path = f"checkpoints/best_model_config_{config}.pth"
    
    # Historique pour les courbes
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_model_wts = None

    best_epoch_idx = 0

    history.update({'train_iou': [], 'val_iou': []})

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs} [Config {config}]')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []


            running_iou = 0.0
            epoch_iou_list = []

            # Barre de progression
            pbar = tqdm(dataloaders[phase], desc=f'{phase.upper()}')
            
            # Iterate over data
            #for inputs_non_seg, inputs_seg, labels in pbar:
                #inputs_non_seg = inputs_non_seg.to(device)
                #inputs_seg = inputs_seg.to(device)
                #labels = labels.to(device).long()

            for batch_idx, (inputs_non_seg, inputs_seg, labels) in enumerate(pbar):
                inputs_non_seg = inputs_non_seg.to(device)
                inputs_seg = inputs_seg.to(device)
                labels = labels.to(device).long()

                # On décide qui va dans la branche Large (L) et Small (S)
                if config == 'A':
                    # Config A : images non segmentées uniquement
                    input_L, input_S = inputs_non_seg, inputs_non_seg
                elif config == 'B':
                    # Config B : images segmentées uniquement
                    input_L, input_S = inputs_seg, inputs_seg
                elif config == 'C1':
                    # Config C1 : segmentées -> Large, non segmentées -> Small
                    input_L, input_S = inputs_seg, inputs_non_seg
                elif config == 'C2':
                    # Config C2 : segmentées -> Small, non segmentées -> Large
                    input_L, input_S = inputs_non_seg, inputs_seg
                elif config == 'Partie3_C':
                    # Config Partie 3 C : segmentées -> Small, non segmentées -> Large
                    input_L, input_S = inputs_non_seg, inputs_seg
                else:
                    raise ValueError(f"Config {config} inconnue. Utilisez 'A', 'B', 'C1' ou 'C2'")

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Le modèle prend 2 entrées : (input_small, input_large)
                    #outputs = model(input_S, input_L)
                    need_attn = (phase == 'val') or (lambda_iou > 0)
                    outputs, iou_score, rollout_map = model(input_S, input_L, return_attention=True)

                    _, preds = torch.max(outputs, 1)
                    #loss = criterion(outputs, labels)
                    ce_loss = criterion(outputs, labels)
                    mean_iou = iou_score.mean()
                    loss = ce_loss + lambda_iou * (1.0 - mean_iou)


                    # backward + optimize (seulement en train)
                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                    if phase == 'val' and epoch % 5 == 0 and batch_idx == 0:
                        visualize_attention_map(
                            image=inputs_non_seg[0],
                            rollout_map=rollout_map[0],
                            mask_gt=inputs_seg[0],
                            config_name=config,
                            idx=epoch,  # 这里用 epoch 作为文件名的唯一标识
                            output_dir='./results/heatmaps'
                        )

                # Statistics
                running_loss += loss.item() * inputs_non_seg.size(0)
                running_corrects += torch.sum(preds == labels.data)

                running_iou += mean_iou.item() * inputs_non_seg.size(0)
                epoch_iou_list.append(mean_iou.item())
                
                # Pour le F1-score
                all_preds.extend(preds.cpu().detach().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})

            if phase == 'train':
                scheduler.step()

            # Calcul des métriques
            dataset_size = len(dataloaders[phase].dataset)
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

            print(f'{phase.upper()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | F1: {epoch_f1:.4f}')

            epoch_iou_mean = np.mean(epoch_iou_list)
            epoch_iou_std = np.std(epoch_iou_list)

            history[f'{phase}_iou'].append(epoch_iou_mean)

            print(f'{phase.upper()} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | IoU: {epoch_iou_mean:.4f} ± {epoch_iou_std:.4f}')
            
            # Sauvegarde l'historique
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            history[f'{phase}_f1'].append(epoch_f1)

            # Sauvegarde du meilleur modèle (basé sur F1-score en validation)
            # Note: On peut aussi utiliser Accuracy si souhaité
            if phase == 'val' and epoch_f1 > best_val_f1:
                best_val_f1 = epoch_f1
                best_val_acc = epoch_acc.item()
                best_model_wts = copy.deepcopy(model.state_dict())
                best_preds = all_preds.copy()
                best_labels = all_labels.copy()

                best_epoch_idx = epoch

                torch.save(model.state_dict(), best_model_path)
                print(f'Meilleur modèle sauvegardé (Acc: {best_val_acc:.4f}, F1: {best_val_f1:.4f})')


    time_elapsed = time.time() - since
    print(f'\n{"-"*10}')
    print(f'Training complete in {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
    print(f'Best val Accuracy: {best_val_acc:.4f}')
    print(f'Best val F1-Score: {best_val_f1:.4f}')
    print(f'{"-"*10}')
    print(f"Best Val IoU: {history['val_iou'][best_epoch_idx]:.4f}")


    # Load best model weights
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    final_preds = {
        'preds': best_preds,
        'labels': best_labels
    }

    return model, history, final_preds