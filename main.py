"""Training example.
"""
import argparse
import datetime
from collections import OrderedDict
import json
import os
import random
import sys
import time
import warnings
import threading
from contextlib import contextmanager
import contextlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import StepLR
from warmup_scheduler import GradualWarmupScheduler
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
import torchvision
from transformers import AutoTokenizer

from gill import data
from gill import losses as losses_utils
from gill import models
from gill import utils
from gill import validate


llm_models = ['facebook/opt-125m', 'facebook/opt-350m', 'facebook/opt-1.3b', 'facebook/opt-2.7b',
              'facebook/opt-6.7b', 'facebook/opt-13b', 'facebook/opt-30b', 'facebook/opt-66b']
datasets = ['cc3m', 'wukong']
best_acc1 = 0  # Variable to keep track of best model so far.


def parse_args(args):
  parser = argparse.ArgumentParser(description='GILL training')
  parser.add_argument('--opt-version', default='facebook/opt-6.7b',
                      help='OPT versions: ' +
                        ' | '.join(llm_models) +
                        ' (default: "facebook/opt-6.7b")')
  parser.add_argument('--visual-model', default='openai/clip-vit-large-patch14', type=str,
                      help="Visual encoder to use.")
  parser.add_argument('--num-tokens', default=8, type=int, metavar='N', help='Number of [IMG] tokens to use.')
  parser.add_argument('--num-clip-tokens', default=77, type=int, metavar='N', help='Number of CLIP token to use for generation.')

  parser.add_argument('-d', '--dataset', metavar='DATASET',  help='Delimited list of datasets:' +
                      ' | '.join(datasets), default='cc3.1m',
                      type=lambda s: [x for x in s.split(',')])

  parser.add_argument('--val-dataset', metavar='DATASET', default='cc3.1m',
            type=lambda s: [x for x in s.split(',')],
            help='Validation dataset: ' +
              ' | '.join(datasets) +
              ' (default: cc3.1m)')
  parser.add_argument('--dataset-dir', default='datasets', type=str,
            help='Dataset directory containing .tsv files.')
  parser.add_argument('--image-dir', default='data/', type=str,
            help='Dataset directory containing image folders.')
  parser.add_argument('--log-base-dir', default='./runs', type=str,
            help='Base directory to write logs and ckpts to.')
  parser.add_argument('--exp-name', default='frozen', type=str,
            help='Name of experiment, used for saving checkpoints.')

  parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
            help='number of data loading workers (default: 4)')
  parser.add_argument('--epochs', default=10, type=int, metavar='N',
            help='number of total epochs to run')
  parser.add_argument('--steps_per_epoch', default=2000, type=int, metavar='N',
            help='number of training steps per epoch')
  parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
            help='manual epoch number (useful on restarts)')
  parser.add_argument('--val_steps_per_epoch', default=-1, type=int, metavar='N',
            help='number of validation steps per epoch')
  parser.add_argument('-b', '--batch-size', default=200, type=int,
            metavar='N',
            help='mini-batch size (default: 200), this is the total '
               'batch size of all GPUs on the current node when '
               'using Data Parallel or Distributed Data Parallel')
  parser.add_argument('--val-batch-size', default=None, type=int)
  parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
            metavar='LR', help='initial learning rate', dest='lr')
  parser.add_argument('--lr-warmup-steps', default=2000, type=int,
            metavar='N', help='Number of steps to warm up lr.')
  parser.add_argument('--lr_schedule_step_size', default=5, type=int,
            metavar='N', help='Number of steps before decaying lr.')
  parser.add_argument('--lr_schedule_gamma', default=0.1, type=float,
            metavar='N', help='Decay parameter for learning rate scheduler.')
  parser.add_argument('--grad-accumulation-steps', default=1, type=int, metavar='N',
                      help='number of gradient accumulation steps')
  parser.add_argument('--grad-clip', default=1.0, type=float, help='gradient clipping amount')

  parser.add_argument('--precision', default='bf16', type=str, choices=['fp32', 'fp16', 'bf16'],
                      help="What precision to train in.")
  parser.add_argument('--cap-loss-scale', type=float, default=1.0, help="Scale on captioning loss.")
  parser.add_argument('--ret-loss-scale', type=float, default=1.0, help="Scale on retrieval loss.")
  parser.add_argument('--gen-loss-scale', type=float, default=1.0, help="Scale on retrieval loss.")

  parser.add_argument('--concat-captions-prob', type=float, default=0.5, help="Probability of concatenating two examples sequentially for captioning.")
  parser.add_argument('--input-prompt', default='', type=str, help="Input prompt for the language model, if any. Use empty string for Chinese models.")

  # 中文提示词数据增强（在 data.py 内给 caption 加前缀）
  parser.add_argument('--prompt-aug-mode', default='random', type=str,
                      choices=['none', 'fixed', 'random'],
                      help="Caption prefix augmentation mode for Chinese datasets. "
                           "'none' disables, 'fixed' uses the first template, 'random' samples from template pool.")
  parser.add_argument('--prompt-aug-prob', default=1.0, type=float,
                      help="Probability of applying caption prefix augmentation (train only). Default 1.0 keeps old behavior.")
  parser.add_argument('--prompt-aug-sep', default='', type=str,
                      help="Separator inserted between prefix and caption, e.g. '：' or ' '. Default '' keeps old behavior.")

  parser.add_argument('--image-size', default=224, type=int, metavar='N', help='Size of images.')
  parser.add_argument('--ret-emb-dim', default=256, type=int, metavar='N', help='Embedding dimension for retrieval.')
  parser.add_argument('--gen-emb-dim', default=768, type=int, metavar='N', help='Embedding dimension for generation. Use 2048 for Kolors.')
  
  # Kolors 适配参数
  parser.add_argument('--use-kolors-targets', action='store_true', 
            help='Use Kolors text encoder to generate target embeddings instead of CLIP.')
  parser.add_argument('--kolors-path', default='./model/Kolors', type=str,
            help='Path to Kolors model for target embedding generation.')
  
  text_fc_modes = ['linear', 'gill_mapper']
  parser.add_argument('--text-fc-mode', default='gill_mapper',
            choices=text_fc_modes, help='What kind of translation mapping to use.')
  parser.add_argument('--ret-text-fc-mode', default='linear',
            choices=text_fc_modes, help='What kind of translation mapping to use.')

  parser.add_argument('--max-len', default=32, type=int,
            metavar='N', help='Maximum length to truncate captions / generations to.')
  parser.add_argument('--n-visual-tokens', default=4, type=int,
            metavar='N', help='Number of visual tokens to use for the Frozen model.')

  parser.add_argument('--batch-timeout', default=300, type=int, metavar='N', help='Timeout for each batch in seconds (default: 300)')
  parser.add_argument('--max-retries', default=3, type=int, metavar='N', help='Max retries for problematic batches (default: 3)')

  parser.add_argument('--beta1', default=0.9, type=float, metavar='M',
            help='beta1 for Adam')
  parser.add_argument('--beta2', default=0.95, type=float, metavar='M',
            help='beta2 for Adam')
  parser.add_argument('--wd', '--weight-decay', default=0.01, type=float,
            metavar='W', help='weight decay (default: 0.01)',
            dest='weight_decay')
  parser.add_argument('-p', '--print-freq', default=10, type=int,
            metavar='N', help='print frequency (default: 10)')
  parser.add_argument('--resume', default='', type=str, metavar='PATH',
            help='path to latest checkpoint (default: none)')
  parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
            help='evaluate model on validation set')
  parser.add_argument('--world-size', default=-1, type=int,
            help='number of nodes for distributed training')
  parser.add_argument('--rank', default=-1, type=int,
            help='node rank for distributed training')
  parser.add_argument('--dist-url', default='tcp://127.0.0.1:1337', type=str,
            help='url used to set up distributed training')
  parser.add_argument('--dist-backend', default='nccl', type=str,
            help='distributed backend')
  parser.add_argument('--seed', default=None, type=int,
            help='seed for initializing training. ')
  parser.add_argument('--gpu', default=None, type=int,
            help='GPU id to use.')
  parser.add_argument('--multiprocessing-distributed', action='store_true',
            help='Use multi-processing distributed training to launch '
               'N processes per node, which has N GPUs. This is the '
               'fastest way to use PyTorch for either single node or '
               'multi node data parallel training')
  return parser.parse_args(args)


def safe_forward(model, images, tgt_tokens, token_len, mode, concat_captions, timeout_duration):
    """Safely perform forward pass with error handling"""
    try:
        # 移除了 timeout，因为 CUDA 操作无法被 Python 线程轻易中断，且容易导致死锁
        result = model(images, tgt_tokens, token_len, mode=mode,
                       concat_captions=concat_captions)
        return result
    except Exception as e:
        print(f"Error during forward pass for mode {mode}: {e}")
        return None


def safe_normalize_embeddings(model, args):
    """Safely normalize trainable embeddings"""
    try:
        # Use torch.no_grad() to ensure no gradients are tracked during normalization
        with torch.no_grad():  
            frozen_norm = torch.norm(model.module.model.input_embeddings.weight[:-args.num_tokens, :], dim=1).mean(0)
            for ret_idx in args.retrieval_token_idx:
                trainable_weight = model.module.model.input_embeddings.weight[ret_idx, :]
                weight_norm = trainable_weight.norm(dim=-1)
                if weight_norm != 0:
                    # Create new tensor instead of in-place operation
                    normalized_weight = trainable_weight / (weight_norm / frozen_norm)
                    model.module.model.input_embeddings.weight.data[ret_idx, :] = normalized_weight
        return True
    except Exception as e:
        print(f"Error during embedding normalization: {e}")
        return False


def sync_error_across_gpus(args, success):
    """Synchronize error status across all GPUs in distributed training"""
    if not args.distributed:
        return success
    
    try:
        success_tensor = torch.tensor([1 if success else 0], device=f'cuda:{args.gpu}')
        dist.all_reduce(success_tensor, op=dist.ReduceOp.MIN)
        return success_tensor.item() == 1
    except Exception as e:
        print(f"Error synchronizing across GPUs: {e}")
        return False


def main(args):
  args = parse_args(args)
  
  if args.resume and os.path.isfile(args.resume):
    args.log_dir = os.path.dirname(args.resume)
    print(f"Resuming training, using existing log directory: {args.log_dir}")
  else:
    i = 1
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    while os.path.exists(args.log_dir):
      args.log_dir = os.path.join(args.log_base_dir, f'{args.exp_name}_{i}')
      i += 1
    os.makedirs(args.log_dir)
    print(f"Starting new training, created log directory: {args.log_dir}")

  with open(os.path.join(args.log_dir, f'args.json'), 'w') as wf:
    json.dump(vars(args), wf, indent=4)

  # 简化 git dump，避免出错
  try:
      with open(os.path.join(args.log_dir, f'git_info.txt'), 'w') as wf:
        utils.dump_git_status(out_file=wf)
  except:
      pass

  print(f'Logging to {args.log_dir}.')

  if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training.')

  if args.gpu is not None:
    warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

  if args.dist_url == "env://" and args.world_size == -1:
    args.world_size = int(os.environ["WORLD_SIZE"])

  args.distributed = args.world_size > 1 or args.multiprocessing_distributed

  ngpus_per_node = torch.cuda.device_count()
  if args.multiprocessing_distributed:
    if args.world_size == -1:
      args.world_size = 1
    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
  else:
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
  global best_acc1
  args.gpu = gpu

  if args.gpu is not None:
    print("Use GPU: {} for training".format(args.gpu))

  if args.distributed:
    if args.dist_url == "env://" and args.rank == -1:
      args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
      if args.rank == -1:
        args.rank = 0
      args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                world_size=args.world_size, rank=args.rank,
                timeout=datetime.timedelta(seconds=7200))

  # 自动适配 Kolors 参数
  if hasattr(args, 'kolors_path') and ('kolors' in args.kolors_path.lower() or args.use_kolors_targets):
      if args.gen_emb_dim != 2048:
          if args.rank == 0 or args.rank == -1:
            print(f"⚠️ [Config Auto-Fix] 检测到 Kolors 配置，自动将 gen_emb_dim 从 {args.gen_emb_dim} 调整为 2048")
          args.gen_emb_dim = 2048
      if args.num_clip_tokens != 256:
          if args.rank == 0 or args.rank == -1:
            print(f"⚠️ [Config Auto-Fix] 检测到 Kolors 配置，自动将 num_clip_tokens 从 {args.num_clip_tokens} 调整为 256")
          args.num_clip_tokens = 256

  # Create model
  model_args = models.GILLArgs()
  model_args.opt_version = args.opt_version
  model_args.visual_encoder = args.visual_model
  model_args.text_emb_layers = [-1]
  model_args.freeze_lm = True
  model_args.freeze_vm = True
  model_args.n_visual_tokens = args.n_visual_tokens
  model_args.ret_emb_dim = args.ret_emb_dim
  model_args.gen_emb_dim = args.gen_emb_dim
  model_args.text_fc_mode = args.text_fc_mode
  model_args.ret_text_fc_mode = args.ret_text_fc_mode
  model_args.num_tokens = args.num_tokens
  model_args.num_clip_tokens = args.num_clip_tokens
  
  # Check if opt_version is a local path
  is_local_opt = os.path.exists(args.opt_version) and os.path.isdir(args.opt_version)
  if is_local_opt:
    print(f"Loading tokenizer from local path: {args.opt_version}")
    tokenizer = AutoTokenizer.from_pretrained(args.opt_version, use_fast=False, local_files_only=True, trust_remote_code=True)
  else:
    tokenizer = AutoTokenizer.from_pretrained(args.opt_version, use_fast=False, trust_remote_code=True)
  
  # 统一处理 pad_token
  if tokenizer.pad_token is None or tokenizer.pad_token_id is None:
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
      tokenizer.pad_token = tokenizer.eos_token
      tokenizer.pad_token_id = tokenizer.eos_token_id
      print(f"✓ Using eos_token as pad_token: pad_token_id={tokenizer.pad_token_id}")
    elif hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
      tokenizer.pad_token = tokenizer.unk_token
      tokenizer.pad_token_id = tokenizer.unk_token_id
      print(f"✓ Using unk_token as pad_token: pad_token_id={tokenizer.pad_token_id}")
    else:
      tokenizer.add_special_tokens({'pad_token': '[PAD]'})
      print(f"✓ Added new pad_token: pad_token_id={tokenizer.pad_token_id}")
  
  if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = 0
    print(f"⚠️ Warning: pad_token_id is None, using 0 as fallback")
  
  print(f"Final tokenizer config: pad_token='{tokenizer.pad_token}', pad_token_id={tokenizer.pad_token_id}")
  
  tokenizer.padding_side = 'right'
  print(f"Set tokenizer.padding_side = '{tokenizer.padding_side}'")
  tokenizer.add_special_tokens({"cls_token": "<|image|>"}) 

  model_args.retrieval_token_idx = []
  args.retrieval_token_idx = []
  for i in range(model_args.num_tokens):
    if i == 0: print(f'Adding [IMG] tokens to vocabulary...')
    tokenizer.add_tokens(f'[IMG{i}]')
    ret_token_idx = tokenizer(f'[IMG{i}]', add_special_tokens=False).input_ids
    assert len(ret_token_idx) == 1, ret_token_idx
    model_args.retrieval_token_idx.append(ret_token_idx[0])
    args.retrieval_token_idx.append(ret_token_idx[0])

  model_args.gen_token_idx = model_args.retrieval_token_idx
  args.gen_token_idx = args.retrieval_token_idx

  with open(os.path.join(args.log_dir, 'model_args.json'), 'w') as f:
    json.dump(vars(model_args), f, indent=4)

  model = models.GILL(tokenizer, model_args)
  if args.precision == 'fp16':
    model = model.half()
  elif args.precision == 'bf16':
    model = model.bfloat16()

  param_counts_text = utils.get_params_count_str(model)
  with open(os.path.join(args.log_dir, 'param_count.txt'), 'w') as f:
    f.write(param_counts_text)

  if not torch.cuda.is_available():
    print('WARNING: using CPU, this will be slow!')
    model = torch.nn.DataParallel(model)
  elif args.distributed:
    if args.gpu is not None:
      torch.cuda.set_device(args.gpu)
      model.cuda(args.gpu)
      args.batch_size = int(args.batch_size / ngpus_per_node)
      args.val_batch_size = int((args.val_batch_size or args.batch_size) / ngpus_per_node)
      args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
      # 关键: find_unused_parameters=True is needed for GILL's modular architecture
      model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
      # DDP Fix: 使用 _set_static_graph() 作为 workaround，因为计算图结构是固定的（只是不同 mode 使用不同分支）
      # 这可以避免参数被标记为 ready 多次的问题
      if hasattr(model, '_set_static_graph'):
        model._set_static_graph()
        print("✓ 启用 DDP static graph mode（避免参数被标记多次）")
    else:
      model.cuda()
      model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
      if hasattr(model, '_set_static_graph'):
        model._set_static_graph()
        print("✓ 启用 DDP static graph mode（避免参数被标记多次）")
  elif args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
  else:
    model = torch.nn.DataParallel(model).cuda()

  criterion = nn.CrossEntropyLoss().cuda(args.gpu)
  optimizer_cls = torch.optim.AdamW
  print('Using torch.optim.AdamW as the optimizer.')
  optimizer = optimizer_cls(model.parameters(), args.lr,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
                eps=1e-8)

  scheduler_steplr = StepLR(optimizer, step_size=args.lr_schedule_step_size * args.steps_per_epoch, gamma=args.lr_schedule_gamma)
  scheduler = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.lr_warmup_steps, after_scheduler=scheduler_steplr)
  
  if args.resume:
    if os.path.isfile(args.resume):
      print("=> loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(args.resume, map_location='cpu')
      args.start_epoch = checkpoint['epoch']
      best_acc1 = checkpoint['best_acc1']
      if args.gpu is not None and torch.is_tensor(best_acc1):
        best_acc1 = best_acc1.to(args.gpu)
      model.load_state_dict(checkpoint['state_dict'], strict=False)
      optimizer.load_state_dict(checkpoint['optimizer'])
      scheduler.load_state_dict(checkpoint['scheduler'])
      print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
      print("=> no checkpoint found at '{}'".format(args.resume))

  cudnn.benchmark = True

  train_dataset = data.get_dataset(args, 'train', tokenizer)
  val_dataset = data.get_dataset(args, 'val', tokenizer)
  print(f'Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples.')

  if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
  else:
    train_sampler = None
    val_sampler = None

  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

  val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=(args.val_batch_size or args.batch_size), shuffle=False,
    num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=True)

  if args.evaluate:
    epoch = args.start_epoch if args.resume else 0
    validate.validate(val_loader, model, tokenizer, criterion, epoch, args)
    # 避免 PyTorch 2.4+ 的 NCCL “process group not destroyed” 警告
    if args.distributed and dist.is_available() and dist.is_initialized():
      dist.destroy_process_group()
    return

  for epoch in range(args.start_epoch, args.epochs):
    if epoch == 0:
      validate.validate(val_loader, model, tokenizer, criterion, epoch-1, args)
    if args.distributed:
      train_sampler.set_epoch(epoch)

    train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args)
    acc1 = validate.validate(val_loader, model, tokenizer, criterion, epoch, args)

    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        and args.rank % ngpus_per_node == 0):

      stripped_state_dict = {
          k: v for k, v in model.state_dict().items() if 
          ('.lm' not in k and '.visual_model' not in k)
      }
      stripped_state_dict = OrderedDict(sorted(stripped_state_dict.items()))
      utils.save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': stripped_state_dict,
        'best_acc1': best_acc1,
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict()
      }, is_best, os.path.join(args.log_dir, 'ckpt'))

  # 避免 PyTorch 2.4+ 的 NCCL “process group not destroyed” 警告
  if args.distributed and dist.is_available() and dist.is_initialized():
    dist.destroy_process_group()


def train(train_loader, model, tokenizer, criterion, optimizer, epoch, scheduler, args):
  ngpus_per_node = torch.cuda.device_count()
  batch_time = utils.AverageMeter('Time', ':6.3f')
  cap_time = utils.AverageMeter('CaptioningTime', ':6.3f')
  ret_time = utils.AverageMeter('RetrievalTime', ':6.3f')
  data_time = utils.AverageMeter('Data', ':6.3f')
  losses = utils.AverageMeter('Loss', ':.4e')
  ce_losses = utils.AverageMeter('CeLoss', ':.4e')
  top1 = utils.AverageMeter('Acc@1', ':6.2f')
  top5 = utils.AverageMeter('Acc@5', ':6.2f')
  cont_losses = utils.AverageMeter('ContLoss', ':.4e')
  gen_losses = utils.AverageMeter('GenLoss', ':.4e')
  top1_caption = utils.AverageMeter('AccCaption@1', ':6.2f')
  top5_caption = utils.AverageMeter('AccCaption@5', ':6.2f')
  top1_image = utils.AverageMeter('AccImage@1', ':6.2f')
  top5_image = utils.AverageMeter('AccImage@5', ':6.2f')
  cap_vis_emb_norm = utils.AverageMeter('VisualEmbNormCap', ':.4e')
  ret_vis_emb_norm = utils.AverageMeter('VisualEmbNormRet', ':.4e')
  inp_emb_norm = utils.AverageMeter('TextEmbNorm', ':.4e')
  all_emb_norm = utils.AverageMeter('AllEmbNorm', ':.4e')
  ret_emb_norm = utils.AverageMeter('RetEmbNorm', ':.4e')

  writer = SummaryWriter(args.log_dir)

  progress = utils.ProgressMeter(
    args.steps_per_epoch,
    [batch_time, losses, ce_losses, cont_losses, gen_losses, top1, top5],
    prefix="Epoch: [{}]".format(epoch))

  model.train()
  end = time.time()
  
  # 兼容 PyTorch 新旧版本
  try:
      scaler = torch.amp.GradScaler('cuda', enabled=(args.precision == 'fp16'))
  except:
      scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == 'fp16'))
      
  successful_steps = 0
  
  optimizer.zero_grad()

  for i, (_, images, caption_images, ret_tokens, ret_caption_len, gen_tokens, gen_caption_len, clip_emb) in enumerate(train_loader):
    actual_step = epoch * args.steps_per_epoch + i + 1
    if i % 10 == 0:
      print(f'[Epoch {epoch}] Processing step {i+1}/{args.steps_per_epoch} (global step {actual_step})')
    
    data_time.update(time.time() - end)

    if torch.cuda.is_available():
      images = images.cuda(args.gpu, non_blocking=True)
      ret_tokens = ret_tokens.long().cuda(args.gpu, non_blocking=True)
      ret_caption_len = ret_caption_len.long().cuda(args.gpu, non_blocking=True)
      gen_tokens = gen_tokens.long().cuda(args.gpu, non_blocking=True)
      gen_caption_len = gen_caption_len.long().cuda(args.gpu, non_blocking=True)
      clip_emb = clip_emb.cuda(args.gpu, non_blocking=True)

    if args.precision == 'fp16':
      images = images.half()
    elif args.precision == 'bf16':
      images = images.bfloat16()

    model_modes = ['captioning', 'retrieval', 'generation']
    total_loss = 0  # 累积所有模式的 loss
    forward_success = True

    # -------------------------------------------------------------------------
    # DDP FIX 2: Single Backward
    # 我们不再对每个 mode 单独 backward。
    # 而是累积所有 modes 的 loss，最后只做一次 backward。
    # 只要我们保证了 dummy gradient (在 models.py 中)，所有参数都会在计算图中。
    # -------------------------------------------------------------------------
    
    # 在 DDP 模式下，只有最后一次 backward 才会触发梯度同步。
    # 这里我们只在 total_loss.backward() 时触发。
    # 但由于 forward 是分多次进行的，我们需要确保梯度不被覆盖（PyTorch 默认是累加，所以没问题）。
    # 问题在于 computational graph 是否能在多次 forward 之间保持。
    # 答案是肯定的，只要我们不清除中间变量。
    
    # 注意：为了避免显存爆炸（保存三份计算图），我们还是需要使用 no_sync 上下文
    # 或者，如果显存足够，一次 backward 是最安全的。
    # 考虑到之前的报错，最好的方式是：
    # 1. 禁用 DDP 自动同步 (使用 context manager)
    # 2. 依次 Forward 计算 Loss
    # 3. 累加 Loss
    # 4. 一次 Backward (自动触发同步)
    
    if args.distributed:
       # 这里实际上不需要显式的 no_sync，因为我们只在最后 backward 一次
       # DDP 会在 backward 时才进行 bucket reduction。
       # 但是，如果显存吃紧，我们可能希望分步 backward。
       # 鉴于之前的错误，我们尝试 "Accumulate Loss, Single Backward" 策略。
       pass

    for mode_idx, model_mode in enumerate(model_modes):
      if i == 0 or (i + 1) % 100 == 0:
        print(f'Step {i+1}: Running {model_mode}')
      mode_start = time.time()
      concat_captions = random.uniform(0, 1) < args.concat_captions_prob

      if model_mode == 'retrieval':
        tgt_tokens, token_len = ret_tokens, ret_caption_len
      elif model_mode == 'generation':
        tgt_tokens, token_len = gen_tokens, gen_caption_len
      else:
        tgt_tokens, token_len = ret_tokens, ret_caption_len

      # Mixed Precision Forward
      try:
          with torch.amp.autocast('cuda', enabled=(args.precision == 'fp16'), dtype=torch.float16):
              forward_result = safe_forward(model, images, tgt_tokens, token_len, model_mode, concat_captions, args.batch_timeout)
      except:
          with torch.cuda.amp.autocast(enabled=(args.precision == 'fp16'), dtype=torch.float16):
              forward_result = safe_forward(model, images, tgt_tokens, token_len, model_mode, concat_captions, args.batch_timeout)
      
      # Check forward success
      forward_ok = 1 if forward_result is not None else 0
      if args.distributed:
        forward_ok_tensor = torch.tensor([forward_ok], device=f'cuda:{args.gpu}')
        dist.all_reduce(forward_ok_tensor, op=dist.ReduceOp.MIN)
        forward_ok = forward_ok_tensor.item()
      
      if forward_ok == 0:
        print(f"Forward pass failed for mode {model_mode} in step {i+1}, skipping batch")
        forward_success = False
        break

      # Unpack results
      (model_output, full_labels, last_embedding, _, visual_embs, visual_embs_norm,
        input_embs_norm, _) = forward_result
      output = model_output.logits

      # Calculate loss for this mode
      ce_loss = model_output.loss
      if not torch.isfinite(ce_loss):
        print(f"[ERROR] ce_loss is NaN/Inf for mode={model_mode}, batch {i+1}")
        forward_success = False
        break
      
      mode_loss = 0
      
      if model_mode == 'captioning':
        ce_loss = ce_loss * args.cap_loss_scale
        mode_loss = mode_loss + ce_loss
        
        # Metrics
        acc_ok = True
        try:
          is_chinese = 'deepseek' in args.opt_version.lower() or 'chinese' in args.opt_version.lower() or 'qwen' in args.opt_version.lower()
          acc1, acc5 = utils.accuracy(output[:, :-1, :], full_labels[:, 1:], -100, topk=(1, 5), is_chinese=is_chinese)
          top1.update(acc1[0], images.size(0))
          top5.update(acc5[0], images.size(0))
        except Exception as e:
          print(f"Error computing accuracy: {e}")
          acc_ok = False
        if not sync_error_across_gpus(args, acc_ok):
            forward_success = False
            break

      elif model_mode == 'retrieval':
        ce_loss = ce_loss * args.ret_loss_scale * 0.5
        mode_loss = mode_loss + ce_loss
        
        # Retrieval specific loss (Contrastive)
        if args.distributed:
          gather_ok = True
          try:
            if visual_embs.numel() == 0 or last_embedding.numel() == 0: gather_ok = False
            if not sync_error_across_gpus(args, gather_ok):
              forward_success = False
              break
            all_visual_embs = [torch.zeros_like(visual_embs) for _ in range(dist.get_world_size())]
            all_last_embedding = [torch.zeros_like(last_embedding) for _ in range(dist.get_world_size())]
            dist.all_gather(all_visual_embs, visual_embs)
            dist.all_gather(all_last_embedding, last_embedding)
            all_visual_embs[dist.get_rank()] = visual_embs
            all_last_embedding[dist.get_rank()] = last_embedding
            visual_embs = torch.cat(all_visual_embs)
            last_embedding = torch.cat(all_last_embedding)
          except Exception as e:
            print(f"Error gathering: {e}")
            forward_success = False
            break

        cont_ok = True
        try:
          if not torch.isfinite(visual_embs).all() or not torch.isfinite(last_embedding).all(): raise ValueError("NaN Embeddings")
          logits_per_image = visual_embs @ last_embedding.t()
          logits_per_text = logits_per_image.t()
          caption_loss = losses_utils.contrastive_loss(logits_per_text)
          image_loss = losses_utils.contrastive_loss(logits_per_image)
          
          if not torch.isfinite(caption_loss) or not torch.isfinite(image_loss): raise ValueError("NaN Contrastive Loss")
          
          caption_acc1, caption_acc5 = losses_utils.contrastive_acc(logits_per_text, topk=(1, 5))
          image_acc1, image_acc5 = losses_utils.contrastive_acc(logits_per_image, topk=(1, 5))
          
          cont_loss = args.ret_loss_scale * (caption_loss + image_loss) / 2.0
          mode_loss = mode_loss + cont_loss
          cont_losses.update(cont_loss.item(), images.size(0))
          top1_caption.update(caption_acc1[0], images.size(0))
          top5_caption.update(caption_acc5[0], images.size(0))
          top1_image.update(image_acc1[0], images.size(0))
          top5_image.update(image_acc5[0], images.size(0))
        except Exception as e:
          print(f"Error computing contrastive: {e}")
          cont_ok = False
        if not sync_error_across_gpus(args, cont_ok):
          forward_success = False
          break

      elif model_mode == 'generation':
        ce_loss = ce_loss * args.gen_loss_scale * 0.5
        mode_loss = mode_loss + ce_loss
        
        gen_ok = True
        try:
          if not torch.isfinite(clip_emb).all() or not torch.isfinite(last_embedding).all(): raise ValueError("NaN Gen Embeddings")
          if args.num_tokens != 0 and args.num_clip_tokens != args.num_tokens:
            seq_len = clip_emb.shape[1]
            last_embedding = last_embedding.reshape((last_embedding.shape[0], seq_len, -1))
          
          image_loss = losses_utils.l2_loss(clip_emb, last_embedding)
          gen_loss = args.gen_loss_scale * image_loss.mean()
          
          if not torch.isfinite(gen_loss): raise ValueError("NaN Gen Loss")
          mode_loss = mode_loss + gen_loss
          gen_losses.update(gen_loss.item(), images.size(0))
        except Exception as e:
          print(f"Error computing l2 loss: {e}")
          gen_ok = False
        if not sync_error_across_gpus(args, gen_ok):
          forward_success = False
          break

      # Update metrics
      ce_losses.update(ce_loss.mean().item(), images.size(0))
      if model_mode == 'retrieval':
        ret_vis_emb_norm.update(visual_embs_norm.item(), images.size(0))
      elif model_mode == 'captioning':
        cap_vis_emb_norm.update(visual_embs_norm.item(), images.size(0))
      inp_emb_norm.update(input_embs_norm.item(), images.size(0))

      if model_mode in ['retrieval', 'generation']:
        ret_time.update(time.time() - mode_start)
      elif model_mode == 'captioning':
        cap_time.update(time.time() - mode_start)
      
      # ✅ 修复：立即 backward，释放计算图（避免显存爆炸）
      # 原代码累加 3 个 mode 的 loss 会保存 3 份计算图（24GB 激活值）
      # 修复后每次只保存 1 份计算图（8GB 激活值）
      loss_to_back = mode_loss / (len(model_modes) * args.grad_accumulation_steps)
      
      if scaler is not None:
          scaler.scale(loss_to_back).backward()
      else:
          loss_to_back.backward()
      
      # 累加 total_loss 用于日志记录（不用于 backward）
      total_loss = total_loss + mode_loss
      
      # 清理中间变量，释放显存
      del mode_loss
      if mode_idx < len(model_modes) - 1:
          torch.cuda.empty_cache()

    if not forward_success:
      optimizer.zero_grad()
      continue

    losses.update(total_loss.item(), images.size(0))
    
    # ---------------------------------------------------------------------
    
    # Optimizer step
    successful_steps += 1
    if (successful_steps % args.grad_accumulation_steps == 0) or (i == args.steps_per_epoch - 1):
        if successful_steps > 0:
            try:
                if args.grad_clip > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
            except (RuntimeError, ValueError) as e:
                print(f"Optimizer step failed: {e}")
                optimizer.zero_grad()
                successful_steps = 0
                continue

        if i % 10 == 0: print('=' * 80)

    if not safe_normalize_embeddings(model, args):
      print(f"Embedding normalization failed")
    
    # Norm logging (check if idx valid first)
    if args.retrieval_token_idx:
        ret_embedding_norm = torch.norm(model.module.model.input_embeddings.weight[args.retrieval_token_idx, :], dim=-1).mean()
        ret_emb_norm.update(ret_embedding_norm.item(), images.size(0))
    
    embedding_norm = torch.norm(model.module.model.input_embeddings.weight, dim=1).mean()
    all_emb_norm.update(embedding_norm.item(), images.size(0))

    batch_time.update(time.time() - end)
    end = time.time()

    if actual_step == 1 or (i + 1) % args.print_freq == 0:
      print('First 5 values of first 3 tokens:', model.module.model.input_embeddings.weight.data[:3, :5])
      if args.retrieval_token_idx:
          print('First 5 values of first [IMG0] token:', model.module.model.input_embeddings.weight.data[args.retrieval_token_idx[0], :5])

      ex_per_sec = args.batch_size / batch_time.avg
      if args.distributed:
        batch_time.all_reduce()
        data_time.all_reduce()
        ex_per_sec = (args.batch_size / batch_time.avg) * ngpus_per_node
        losses.all_reduce()
        ce_losses.all_reduce()
        top1.all_reduce()
        top5.all_reduce()
        cap_vis_emb_norm.all_reduce()
        ret_vis_emb_norm.all_reduce()
        inp_emb_norm.all_reduce()
        ret_time.all_reduce()
        all_emb_norm.all_reduce()
        ret_emb_norm.all_reduce()
        cont_losses.all_reduce()
        gen_losses.all_reduce()
        top1_caption.all_reduce()
        top5_caption.all_reduce()
        top1_image.all_reduce()
        top5_image.all_reduce()
        cap_time.all_reduce()

      progress.display(i + 1)

      writer.add_scalar('train/loss', losses.avg, actual_step)
      writer.add_scalar('train/ce_loss', ce_losses.avg, actual_step)
      # ... other scalars ...

      if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        image_bs = images.shape[0]
        normalized_images = images - images.min()
        normalized_images /= normalized_images.max()
        max_images_to_show = 16

        pred_tokens = output[:, args.n_visual_tokens-1:-1, :].argmax(dim=-1)
        generated_captions = tokenizer.batch_decode(pred_tokens, skip_special_tokens=False)

        if model_mode == 'captioning':
          try:
            # 中文可视化修复：直接传递字符串
            generated_cap_images = torch.stack([
              utils.create_image_of_text(
                generated_captions[i], 
                width=normalized_images.shape[3],
                color=(255, 255, 0))
              for i in range(len(generated_captions))], axis=0)

            if (args.concat_captions_prob > 0 and generated_cap_images.shape[0] != caption_images.shape[0]):
              generated_cap_images = torch.cat([generated_cap_images, generated_cap_images], axis=0)

            display_images = torch.cat([normalized_images.float().cpu(), caption_images, generated_cap_images], axis=2)[:max_images_to_show]
            grid = torchvision.utils.make_grid(display_images, nrow=int(max_images_to_show ** 0.5), padding=4)
            writer.add_image('train/images_gen_cap', grid, actual_step)
          except Exception as e:
            print(f"Error creating generated caption images: {e}")

      # Reset meters
      batch_time.reset()
      cap_time.reset()
      ret_time.reset()
      data_time.reset()
      losses.reset()
      ce_losses.reset()
      top1.reset()
      top5.reset()
      ret_vis_emb_norm.reset()
      cap_vis_emb_norm.reset()
      inp_emb_norm.reset()
      all_emb_norm.reset()
      ret_emb_norm.reset()
      cont_losses.reset()
      gen_losses.reset()
      top1_caption.reset()
      top5_caption.reset()
      top1_image.reset()
      top5_image.reset()

    if i == args.steps_per_epoch - 1:
      break

    scheduler.step()
    curr_lr = scheduler.get_last_lr()
    if (actual_step == 1) or (i + 1) % args.print_freq == 0:
      writer = SummaryWriter(args.log_dir)
      writer.add_scalar('train/lr', curr_lr[0], actual_step)
      writer.close()

  writer.close()

# Disable tokenizer parallelism.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
  main(sys.argv[1:])
