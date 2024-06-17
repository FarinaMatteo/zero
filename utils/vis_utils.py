import os
import math
import torch
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms import ToPILImage
from utils.tools import entropy

def get_colors(num_colors):
    return sns.color_palette('colorblind', num_colors)


def denormalize(image: torch.Tensor, mean: list = [0.48145466, 0.4578275, 0.40821073], std: list = [0.26862954, 0.26130258, 0.27577711]):
    mean = torch.tensor(mean).view(3,1,1).to(image.device)
    std = torch.tensor(std).view(3,1,1).to(image.device)
    image = image * std + mean
    return image


def back_to_image_space(image):
    norm_image = denormalize(image)
    return ToPILImage(mode='RGB')(norm_image)

# def visualize_batch_unravel(batch: torch.Tensor, folder: str):
#     assert len(batch.shape) == 4, f'Expected 4D tensor, got {batch.shape}'
#     assert os.path.isdir(folder), f'Expected valid directory, got {folder}'

#     for i in range(batch.shape[0]):
#         image = denormalize(batch[i])
#         image = ToPILImage(mode='RGB')(image)
#         image.save(os.path.join(folder, f'{i}.png'))


def visualize_batch_unravel(batch: torch.Tensor, logits: torch.Tensor, label: torch.Tensor, classnames: str, folder: str):
    assert len(batch.shape) == 4, f'Expected 4D tensor, got {batch.shape}'
    assert os.path.isdir(folder), f'Expected valid directory, got {folder}'

    probs = F.softmax(logits, dim=-1)

    # dump the groundtruth on disk
    with open(os.path.join(folder, 'gt.txt'), 'w') as f:
        f.write(f'{classnames[label.item()]}\n')


    for i in range(batch.shape[0]):
        image = denormalize(batch[i])
        image = ToPILImage(mode='RGB')(image)
        image.save(os.path.join(folder, f'{i}.png'))
        # dump the prediction (with confidence) on disk
        with open(os.path.join(folder, f'{i}.txt'), 'w') as f:
            f.write(f'{classnames[probs[i].argmax()]}, {probs[i].max()}\n')


def visualize_wrongly_predicted(image: torch.Tensor, preds: torch.Tensor, target: torch.Tensor, class_names: list, save_to: str, topk: int = 10):
    assert len(image.shape) == 3, f'Expected 3D tensor, got {image.shape}'
    assert len(preds.shape) == 2 and preds.shape[0] == 1, f'Expected 2D tensor with shape (1, num_classes), got {preds.shape}'
    assert target.shape == (1,), f'Expected 1D tensor with shape (1,), got {target.shape}'
    assert os.path.isdir(save_to), f'Expected valid directory, got {save_to}'

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,10))

    # rearrange image for visualization
    image = denormalize(image)
    image = ToPILImage(mode='RGB')(image)

    # grab index of predicted class
    pred = preds.squeeze().argmax().item()
    confidence = F.softmax(preds, dim=-1).squeeze().amax().item()

    # plot image to the left portion of the figure
    ax1.imshow(image)
    ax1.set_title(f'y_hat: {class_names[pred]}\n(confidence={confidence:.2f})\nGT: {class_names[target.item()]}')
    ax1.axis('off')

    # plot bar chart to the right portion of the figure
    topk_preds, topk_indices = F.softmax(preds, dim=-1).squeeze().topk(topk)
    topk_classnames = [class_names[i] for i in topk_indices.tolist()]

    color = ['r' if i == target.item() else 'b' for i in topk_indices]
    fill = [True if i == target.item() else False for i in topk_indices]

    ax2.bar(list(range(topk)), topk_preds.tolist(), color=color, fill=fill)
    ax2.set_xticks(list(range(topk)), topk_classnames, rotation=45)

    for i in range(topk):
        if topk_indices[i] == target.item():
            color = 'red'
        else:
            color = 'black'
        ax2.text(i, topk_preds[i].item(), round(topk_preds[i].item(), 2), ha='center', va='bottom', color=color)

    ax2.set_ylim(0, 1)

    # save to directory
    num_items = len(os.listdir(save_to))
    fig.savefig(os.path.join(save_to, f'{num_items}.png'), bbox_inches='tight')
    plt.close(fig)


def attn_map_from_attn_weights(attn_weights: torch.Tensor, patch_size: int = 16, resolution: int = 224):
    mi, ma = attn_weights.amin(), attn_weights.amax()
    # min-max normalization
    attn_weights = (attn_weights - mi) / (ma - mi)
    attn_weights = attn_weights.view(resolution//patch_size, resolution//patch_size)
    attn_weights = ToPILImage(mode='RGB')(attn_weights.repeat(3,1,1))
    attn_weights = attn_weights.resize((resolution, resolution))
    arr = np.array(attn_weights)[..., 0]
    return arr


def visualize_attention_maps(image: torch.Tensor, 
                             attn_weights: torch.Tensor, 
                             preds: torch.Tensor, 
                             target: torch.Tensor, 
                             class_names: list, 
                             save_to: str, 
                             topk: int = 10):
    
    assert len(image.shape) == 3, f'Expected 3D tensor, got {image.shape}'
    assert len(attn_weights.shape) in (2,4), f'Expected 2D or 4D tensor, got {attn_weights.shape}'
    assert os.path.isdir(save_to), f'Expected valid directory, got {save_to}'

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,10))

    # rearrange image for visualization
    image = denormalize(image)
    image = ToPILImage(mode='RGB')(image)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'pred: {class_names[preds.argmax().item()]}\nGT: {class_names[target.item()]}')

    # plot bar chart to the right portion of the figure
    if len(attn_weights.shape) == 4:
        attn_weights = F.softmax(attn_weights[-1,0,0,1:], dim=-1)
    attn_map = attn_map_from_attn_weights(attn_weights)
    ax1.imshow(attn_map, alpha=0.3, cmap='jet')

    # plot bar chart to the right portion of the figure
    topk_preds, topk_indices = F.softmax(preds, dim=-1).squeeze().topk(topk)
    topk_classnames = [class_names[i] for i in topk_indices.tolist()]

    color = ['r' if i == target.item() else 'b' for i in topk_indices]
    fill = [True if i == target.item() else False for i in topk_indices]

    ax2.bar(list(range(topk)), topk_preds.tolist(), color=color, fill=fill)
    ax2.set_xticks(list(range(topk)), topk_classnames, rotation=45)

    for i in range(topk):
        if topk_indices[i] == target.item():
            color = 'red'
        else:
            color = 'black'
        ax2.text(i, topk_preds[i].item(), round(topk_preds[i].item(), 2), ha='center', va='bottom', color=color)

    ax2.set_ylim(0, 1)

    # save to directory
    num_items = len(os.listdir(save_to))
    fig.savefig(os.path.join(save_to, f'{num_items}.jpg'), bbox_inches='tight')
    plt.close(fig)


def visualize_batch(batch: torch.Tensor, preds: torch.Tensor = None, target: torch.Tensor = None, class_names: list = [], save_to: str = '', plot_preds=-1):
    assert len(batch.shape) == 4, f'Expected 4D tensor, got {batch.shape}'
    try:
        assert os.path.isdir(save_to), f'Expected valid directory, got {save_to}'
    except:
        assert os.path.isdir(os.path.dirname(save_to)), f'Expected valid directory, got {save_to}'
        assert save_to.endswith('.png') or save_to == '', f'Expected valid file path, got {save_to}'


    # create a grid where each cell is a batch item
    num_elems_per_side = math.ceil(math.sqrt(batch.shape[0]))
    fig, axes = plt.subplots(num_elems_per_side, num_elems_per_side, figsize=(18,18))

    # # get the majority prediction
    # if preds is not None:
    #     predicted_classes = preds.argmax(dim=-1)
    #     unique, counts = torch.unique(predicted_classes, return_counts=True)
    #     majority_pred = unique[counts.argmax()].item()


    for i in range(num_elems_per_side):
        for j in range(num_elems_per_side):
            idx = i*num_elems_per_side + j
            if idx < batch.shape[0]:
                image = denormalize(batch[idx])
                image = ToPILImage(mode='RGB')(image)
                axes[i,j].imshow(image)
                axes[i,j].axis('off')

                # if the predictions are provided, set the title of each cell with '{pred}, {confidence}'
                if preds is not None:
                    pred = preds[idx].argmax().item()
                    confidence = F.softmax(preds[idx], dim=-1).amax().item()
                    color = 'black'
                    axes[i,j].set_title(f'{class_names[pred]}\n({confidence:.2f})', pad=5, color=color, fontsize=16)

                    # and also set a border around the cell if it still lies within the MI range
                    # if idx != 0 and mis[idx-1] > 0:
                    #     # Add a border around the image
                    #     border_color = 'green'  # Change this to the desired color
                    #     border_thickness = 5   # Change this to the desired thickness

                    #     # Get the dimensions of the image
                    #     height, width = image.size[:2]

                    #     # Create a rectangle patch with the same dimensions as the image plus the desired border thickness
                    #     border_patch = patches.Rectangle((0, 0), width, height, linewidth=border_thickness, edgecolor=border_color, facecolor='none')

                    #     # Add the border patch to the current axes
                    #     axes[i,j].add_patch(border_patch)

    plt.subplots_adjust(top=0.925, bottom=0.1, left=0.1, right=0.9, hspace=0.5, wspace=0.5)

    # set a title if the target is provided
    if target is not None:
        fig.suptitle(f'GT: {class_names[target.item()]}', fontsize=20)
    
    # fake artists to include a legend in the figure
    # correct_label = plt.Line2D([0], [0], color='green', lw=1, label='Majority Correct.')
    # incorrect_label = plt.Line2D([0], [0], color='orange', lw=1, label='Majority Incorrect.')
    # almost_correct_label = plt.Line2D([0], [0], color='cyan', lw=1, label='Correct but not majority.')

    # # Add legend with custom labels
    # plt.legend(handles=[correct_label, incorrect_label, almost_correct_label], loc='upper center', bbox_to_anchor=(1.5, 0.25))

    # save to directory
    if save_to != '':
        if save_to.endswith('.png'):
            full_name_without_ext, ext = os.path.splitext(save_to)
            idx = 0
            while True:
                save_to = f'{full_name_without_ext}_{idx}{ext}'
                if os.path.exists(save_to):
                    idx += 1
                else:
                    break
            print("Saving figure to", save_to)
            fig.savefig(save_to, bbox_inches='tight')
        else:
            num_items = len(os.listdir(save_to))
            fig.savefig(os.path.join(save_to, f'{num_items}.jpg'), bbox_inches='tight')
    plt.close(fig) 


def visualize_class_histogram(preds, classnames, titles, target, top=10, save_to=None):
    # get the top-k predictions
    topk_preds, topk_classes = preds.topk(k=top, largest=True, dim=1)

    # create a grid where each cell is a batch item
    fig, axes = plt.subplots(preds.size(0), 1, figsize=(14,14))

    for i in range(preds.size(0)):
        # get the top-k predictions for the current image
        curr_preds = topk_preds[i]
        topk_classnames = [classnames[i] for i in topk_classes[i].tolist()]

        # plot the bar chart
        colors = ['r' if cls_pred == target else 'b' for cls_pred in topk_classes[i]]
        axes[i].bar(list(range(top)), curr_preds.tolist(), color=colors)
        axes[i].set_xticks(list(range(top)), topk_classnames, rotation=90, fontsize=8)
        axes[i].set_title(titles[i])
        
    if save_to is not None:
        assert os.path.isdir(save_to), f'Expected valid directory, got {save_to}'
        num_items = len(os.listdir(save_to))
        fig.savefig(os.path.join(save_to, f'{num_items}.jpg'))
    
    plt.close(fig)