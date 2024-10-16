import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from utils.tools import (
    Summary, AverageMeter, ProgressMeter, 
    accuracy, display_results, print, arg_in_results, seed_everything
)
from data.cls_to_names import *
from data import prepare_dataset

from ttas import get_tta_module


def create_results_filename(tta_module, args):
    alg_name = str(tta_module)
    name = f"{alg_name}_{args.arch.replace('/', '-')}" 
    if args.maple:
        name += f"_maple"
    name += f"_seed{args.seed}"
    name = name.replace("/", "_")
    if args.reward_arch is not None:
        name += f"_r{args.reward_arch.replace('/', '-')}"
    if args.templates:
        name += f"_templates"
    return name


def augment_results(results, args):
    results = arg_in_results(results, "seed", args.seed)
    results = arg_in_results(results, "arch", args.arch)
    results = arg_in_results(results, "templates", bool(args.templates))
    results = arg_in_results(results, "maple", bool(args.maple))
    return results


def main(args):
    
    # reproducibility
    seed_everything(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True) # warn_only needed to allow for torch.topk
    cudnn.benchmark = True

    # create TTA-ed module
    tta_module = get_tta_module(
        args.tuner, 
        **dict(model=args.model, 
        arch=args.arch, 
        use_templates=args.templates,
        pretrained=args.pretrained,
        gpu=args.gpu, 
        ctx_init=args.ctx_init,
        maple_weights=args.maple,
        reward_arch=args.reward_arch,
        reward_pretrained=args.reward_pretrained,
        seed=args.seed)
    )
    tta_module = tta_module.to(args.gpu)
    
    # iterating through eval datasets
    set_id = args.set_id
    results = {}
    print("=> Evaluating on testset [{}]".format(set_id))

    # create dataset
    val_dataset = prepare_dataset(tta_module, set_id, args.num_views, args.resolution, args.dataset_mode)
        
    # create dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, # episodic TTA!
        shuffle=True, # irrelevant 
        num_workers=args.workers, 
        pin_memory=True,
        drop_last=False
    )
    print("=> Number of test samples {} (# classes = {})".format(len(val_loader), len(val_dataset.classes)))

    # prepare the model for the current dataset
    # (set the class names and their embeddings for the current dataset)
    tta_module.prepare_for_training(set_id, args.arch)
    
    # run evaluation over the dataset
    results[set_id] = test_time_adapt_eval(val_loader, tta_module, args)
    
    # log and release memory
    accs = [results[set_id][k] for k in results[set_id] if "Acc" in k]
    print(f"=> Testset [{set_id}]: Acc@1 {accs[0]:.2f}/ Acc@5 {accs[1]:.2f} / Acc@10 {accs[2]:.2f}\n")
    del val_dataset, val_loader

    # create the folder
    results_dir = args.results_dir if args.debug_steps ==-1 else args.results_dir+"_debug"
    os.makedirs(results_dir, exist_ok=True)
    
    # create the filename (including parameters)
    name = create_results_filename(tta_module, args)
    output_path = os.path.join(results_dir, f"{name}.csv")
    
    results = augment_results(results, args)
    display_results(results, save_to=output_path)


def test_time_adapt_eval(val_loader, tta_module, args):
    
    # initialize meters
    batch_time = AverageMeter('Time[s]', ':4.3f', Summary.AVERAGE)
    top1 = AverageMeter('Acc@1[%]', ':4.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5[%]', ':4.2f', Summary.AVERAGE)
    top10 = AverageMeter('Acc@10[%]', ':4.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        num_batches=len(val_loader),
        meters=[batch_time, top1, top5, top10],
        prefix='Test: '
    )

    # reset model and switch to evaluate mode
    tta_module.eval()
    
    # iterate through the validation set
    for i, (images, target) in enumerate(val_loader):
        
        # enable debug mode
        if args.debug_steps!=-1 and (i+1) == args.debug_steps:
            print("Debug mode is on. Quitting early...")
            break
        
        # move the data to the GPU
        target = target.to(args.gpu, non_blocking=True)
        images = [img.to(args.gpu, non_blocking=True) for img in images]
        images = torch.cat(images, dim=0)

        # tta with zero temp is implemented in the forward pass of the model
        with torch.cuda.amp.autocast():
            
            # measure tta time with cuda events
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()

            # actual tta here
            output = tta_module(images)

            # finish measuring time
            end_event.record()
            torch.cuda.synchronize()
            
        # measure accuracy
        acc1, acc5, acc10 = accuracy(output, target, topk=(1, 5, 10))
        top1.update(val=acc1[0], n=1)
        top5.update(val=acc5[0], n=1)
        top10.update(val=acc10[0], n=1)

        # measure elapsed time and display updates
        batch_time.update(start_event.elapsed_time(end_event)/1000, n=1)
        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            progress.display(i+1)

    # when evaluation over the shard of each rank is finished, we must gather the AverageMeters from the different ranks
    print("=> Finished evaluation.")
    results_dict = dict(zip([m.name for m in progress.meters], [m.avg for m in progress.meters]))
    return results_dict


if __name__ == '__main__':
    import ttas as ttas
    parser = argparse.ArgumentParser(description='Test-Time Adaptation with Zero Temperature for Vision-Language Models.')

    # parameters for the TTA method
    parser.add_argument('-t', '--tuner', type=str, default='TPT', help='tuner to use: TPT/MEMO', choices=ttas.__all__)
    parser.add_argument('--ctx_init', required=True, type=str, help="underscore separated context, such as 'a_photo_of_a' ")
    parser.add_argument('--num_views', default=64, type=int, help='number of views for TTA')
    parser.add_argument('--seed', type=int, default=0)

    # model parameters
    parser.add_argument('-m', '--model', type=str, default='clip', help='model to use: clip/vit', choices=['clip'])
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B-16')
    parser.add_argument('--pretrained', type=str, default="openai", help="Pretrained Keyword for the OpenCLIP repo. \
                        Default: \"openai\", will also use OpenAI's implementation of CLIP.")
    parser.add_argument('--templates', action="store_true", help="Use textual templates (+Ensemble in the paper).")
    
    # data parameters
    from data.datautils import ID_to_DIRNAME
    parser.add_argument('--set_id', type=str, required=True, help='ID of the Test Dataset (case sensitive).', choices=list(ID_to_DIRNAME.keys()))
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')

    # hardware arguments
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
    parser.add_argument('--results_dir', type=str, default='results', help='directory to save results')

    # development arguments
    parser.add_argument('-p', '--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--debug_steps', type=int, default=-1)

    # arguments for Reinforcement Learning from CLIP Feedback (optional)
    parser.add_argument('--reward_arch', type=str, default=None, help='reward model to use (optional)')
    parser.add_argument('--reward_pretrained', type=str, default=None, help="Enables using OpenCLIP models with ZeroRLCF. Please see the --pretrained flag for more details.")

    parser.add_argument('--maple', action="store_true", help='Use MaPLe weights. Will load a different pretraining based on the seed.')
    args = parser.parse_args()
    
    # setup tensor cores 
    torch.set_float32_matmul_precision("medium")

    main(args)