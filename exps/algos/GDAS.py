##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2020 #
###########################################################################
# Searching for A Robust Neural Architecture in Four GPU Hours, CVPR 2019 #
###########################################################################
import sys, time, random, argparse
from copy import deepcopy
import torch
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces
from nas_201_api  import NASBench201API as API

from datasets.get_dataset_with_transform import CUTOUT
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils.min_norm_solvers import MinNormSolver, gradient_normalizers

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def search_func(xloader, network, criterion, scheduler, w_optimizer, a_optimizer, epoch_str, xargs, logger, ood_loader=None):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  network.train()
  end = time.time()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    base_inputs = base_inputs.cuda(non_blocking=True)
    arch_inputs = arch_inputs.cuda(non_blocking=True)
    if xargs.adv_outer:
      arch_inputs.requires_grad = True
    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)

    if xargs.ood_inner or xargs.ood_outer:
      try:
        ood_input, _ = next(ood_loader_iter)
      except:
        ood_loader_iter = iter(ood_loader)
        ood_input, _ = next(ood_loader_iter)
      ood_input = ood_input.cuda(non_blocking=True)

    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    w_optimizer.zero_grad()
    _, logits, _, _ = network(base_inputs)
    base_loss = criterion(logits, base_targets)
    if xargs.ood_inner and ood_loader is not None:
      _, ood_logits, _, _ = network(ood_input)
      ood_loss = F.kl_div(input=F.log_softmax(ood_logits, dim=-1), target=torch.ones_like(ood_logits)/ood_logits.size()[-1])
      base_loss += ood_loss
    base_loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
    w_optimizer.step()
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(),  base_inputs.size(0))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))

    # update the architecture-weight
    a_optimizer.zero_grad()
    grads = {}
    loss_data = {}
    # ---- acc loss ----
    _, acc_logits, nop_loss, flp_loss = network(arch_inputs)
    acc_loss = criterion(acc_logits, arch_targets)
    loss_data['acc'] = acc_loss.item()
    grads['acc'] = list(torch.autograd.grad(acc_loss, network.get_alphas(), retain_graph=True))
    # del acc_logits
    # ---- end ----

    # ---- nop loss ----
    if xargs.nop_outer:
      if xargs.nop_constrain == 'abs':
        nop_loss = torch.abs(xargs.nop_constrain_min - nop_loss)
      loss_data['nop'] = nop_loss.item()
      grads['nop'] = list(torch.autograd.grad(nop_loss, network.get_alphas(), retain_graph=True))
    # ---- end ----

    # ---- flp loss ----
    if xargs.flp_outer:
      if xargs.flp_constrain == 'abs':
        flp_loss = torch.abs(xargs.flp_constrain_min - flp_loss)
      loss_data['flp'] = flp_loss.item()
      grads['flp'] = list(torch.autograd.grad(flp_loss, network.get_alphas(), retain_graph=True))
    # ---- end ----

    # ---- ood loss ----
    if xargs.ood_outer and ood_loader is not None:
      _, ood_logits, _, _ = network(ood_input)
      ood_loss = F.kl_div(input=F.log_softmax(ood_logits), target=torch.ones_like(ood_logits)/ood_logits.size()[-1])
      loss_data['ood'] = ood_loss.item()
      grads['ood'] = list(torch.autograd.grad(ood_loss, network.get_alphas(), retain_graph=True))
      del ood_logits
    # ---- end ----

    # ---- adv loss ----
    if xargs.adv_outer:
      if xargs.dataset == 'cifar10':
          mean = (0.4914, 0.4822, 0.4465)
          std = (0.2471, 0.2435, 0.2616)
      elif xargs.dataset == 'cifar100':
          mean = (0.5071, 0.4867, 0.4408)
          std = (0.2675, 0.2565, 0.2761)
      mean = torch.FloatTensor(mean).view(3,1,1)
      std = torch.FloatTensor(std).view(3,1,1)
      upper_limit = ((1 - mean)/ std).cuda()
      lower_limit = ((0 - mean)/ std).cuda()
      epsilon = ((xargs.epsilon / 255.) / std).cuda()
      step_size = epsilon * 1.25
      delta = ((torch.rand(arch_inputs.size())-0.5)*2).cuda() * epsilon
      adv_grad = torch.autograd.grad(acc_loss, arch_inputs, retain_graph=True, create_graph=False)[0]
      adv_grad = adv_grad.detach().data
      delta = clamp(delta + step_size * torch.sign(adv_grad), -epsilon, epsilon)
      delta = clamp(delta, lower_limit - arch_inputs.data, upper_limit - arch_inputs.data)
      adv_input = (arch_inputs.data + delta).cuda()
      _, adv_logits, _, _ = network(adv_input)
      adv_loss = criterion(adv_logits, arch_targets)
      loss_data['adv'] = adv_loss.item()
      grads['adv'] = list(torch.autograd.grad(adv_loss, network.get_alphas(), retain_graph=True))
      del mean, std, upper_limit, lower_limit, epsilon, step_size, delta, adv_grad, adv_input, adv_logits
    # ---- end ----

    # ---- MGDA ----
    gn = gradient_normalizers(grads, loss_data, normalization_type=xargs.grad_norm) # loss+, loss, l2

    for t in grads:
      for gr_i in range(len(grads[t])):
        grads[t][gr_i] = grads[t][gr_i] / (gn[t]+1e-7)
    
    if xargs.MGDA and (len(grads)>1):
      sol, _ = MinNormSolver.find_min_norm_element([grads[t] for t in grads])
      print(sol) # acc, adv, nop
    else:
      sol = [1] * len(grads)

    arch_loss = 0
    for kk, t in enumerate(grads):
      if t == 'acc':
        arch_loss += float(sol[kk]) * acc_loss
      elif t == 'adv':
        arch_loss += float(sol[kk]) * adv_loss
      elif t == 'nop':
        arch_loss += float(sol[kk]) * nop_loss
      elif t == 'ood':
        arch_loss += float(sol[kk]) * ood_loss
      elif t == 'flp':
        arch_loss += float(sol[kk]) * flp_loss
    # ---- end ----

    arch_loss.backward()
    a_optimizer.step()
    # record
    arch_prec1, arch_prec5 = obtain_accuracy(acc_logits.data, arch_targets.data, topk=(1, 5))
    arch_losses.update(arch_loss.item(),  arch_inputs.size(0))
    arch_top1.update  (arch_prec1.item(), arch_inputs.size(0))
    arch_top5.update  (arch_prec5.item(), arch_inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % xargs.print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      Astr = 'Arch [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=arch_losses, top1=arch_top1, top5=arch_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr + ' ' + Astr)
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg


def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  #config_path = 'configs/nas-benchmark/algos/GDAS.config'
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', config.batch_size, xargs.workers)
  
  if xargs.ood_inner or xargs.ood_outer:
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std  = [x / 255 for x in [63.0, 62.1, 66.7]]
    # lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(), transforms.Normalize(mean, std)]
    lists = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    # lists += [CUTOUT(-1)]
    ood_transform = transforms.Compose(lists)
    ood_data = dset.SVHN(root=args.data_path, split='train', download=True, transform=ood_transform)

    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=config.batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(list(range(len(ood_data)))[:len(train_data)]),
                pin_memory=True, num_workers=xargs.workers)
  else:
    ood_loader = None

  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  global search_space
  search_space = get_search_spaces('cell', xargs.search_space_name)
  if xargs.model_config is None:
    model_config = dict2config({'name': 'GDAS', 'C': xargs.channel, 'N': xargs.num_cells,
                                'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                                'space'    : search_space,
                                'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats),}, None)
  else:
    model_config = load_config(xargs.model_config, {'num_classes': class_num, 'space'    : search_space,
                                                    'affine'     : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  search_model = get_cell_based_tiny_net(model_config)
  # logger.log('search-model :\n{:}'.format(search_model))
  logger.log('model-config : {:}'.format(model_config))
  
  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  flop, param  = get_model_infos(search_model, xshape)
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search-space [{:} ops] : {:}'.format(len(search_space), search_space))
  if xargs.arch_nas_dataset is None:
    api = None
  else:
    api = API(xargs.arch_nas_dataset)
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  # network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
  network, criterion = search_model.cuda(), criterion.cuda()

  if last_info.exists(): # automatically resume from previous checkpoint
    logger.log("=> loading checkpoint of the last-info '{:}' start".format(last_info))
    last_info   = torch.load(last_info)
    start_epoch = last_info['epoch']
    checkpoint  = torch.load(last_info['last_checkpoint'])
    genotypes   = checkpoint['genotypes']
    valid_accuracies = checkpoint['valid_accuracies']
    search_model.load_state_dict( checkpoint['search_model'] )
    w_scheduler.load_state_dict ( checkpoint['w_scheduler'] )
    w_optimizer.load_state_dict ( checkpoint['w_optimizer'] )
    a_optimizer.load_state_dict ( checkpoint['a_optimizer'] )
    logger.log("=> loading checkpoint of the last-info '{:}' start with {:}-th epoch.".format(last_info, start_epoch))
  else:
    logger.log("=> do not find the last-info file : {:}".format(last_info))
    start_epoch, valid_accuracies, genotypes = 0, {'best': -1}, {-1: search_model.genotype()}

  # start training
  start_time, search_time, epoch_time, total_epoch = time.time(), AverageMeter(), AverageMeter(), config.epochs + config.warmup
  for epoch in range(start_epoch, total_epoch):
    w_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    search_model.set_tau( xargs.tau_max - (xargs.tau_max-xargs.tau_min) * epoch / (total_epoch-1) )
    logger.log('\n[Search the {:}-th epoch] {:}, tau={:}, LR={:}'.format(epoch_str, need_time, search_model.get_tau(), min(w_scheduler.get_lr())))

    search_w_loss, search_w_top1, search_w_top5, valid_a_loss , valid_a_top1 , valid_a_top5 \
              = search_func(search_loader, network, criterion, w_scheduler, w_optimizer, a_optimizer, epoch_str, xargs, logger, ood_loader)
    search_time.update(time.time() - start_time)
    logger.log('[{:}] searching : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))
    logger.log('[{:}] evaluate  : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%'.format(epoch_str, valid_a_loss , valid_a_top1 , valid_a_top5 ))
    # check the best accuracy
    valid_accuracies[epoch] = valid_a_top1
    if valid_a_top1 > valid_accuracies['best']:
      valid_accuracies['best'] = valid_a_top1
      genotypes['best']        = search_model.genotype()
      find_best = True
    else: find_best = False

    genotypes[epoch] = search_model.genotype()
    logger.log('<<<--->>> The {:}-th epoch : {:}'.format(epoch_str, genotypes[epoch]))
    # save checkpoint
    save_path = save_checkpoint({'epoch' : epoch + 1,
                'args'  : deepcopy(xargs),
                'search_model': search_model.state_dict(),
                'w_optimizer' : w_optimizer.state_dict(),
                'a_optimizer' : a_optimizer.state_dict(),
                'w_scheduler' : w_scheduler.state_dict(),
                'genotypes'   : genotypes,
                'valid_accuracies' : valid_accuracies},
                model_base_path, logger)
    last_info = save_checkpoint({
          'epoch': epoch + 1,
          'args' : deepcopy(args),
          'last_checkpoint': save_path,
          }, logger.path('info'), logger)
    if find_best:
      logger.log('<<<--->>> The {:}-th epoch : find the highest validation accuracy : {:.2f}%.'.format(epoch_str, valid_a_top1))
      copy_checkpoint(model_base_path, model_best_path, logger)
    with torch.no_grad():
      logger.log('{:}'.format(search_model.show_alphas()))
    if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[epoch], '200')))
    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()

  logger.log('\n' + '-'*100)
  # check the performance from the architecture dataset
  logger.log('GDAS : run {:} epochs, cost {:.1f} s, last-geno is {:}.'.format(total_epoch, search_time.sum, genotypes[total_epoch-1]))
  if api is not None: logger.log('{:}'.format(api.query_by_arch(genotypes[total_epoch-1], '200')))
  logger.close()
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser("GDAS")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')
  parser.add_argument('--model_config',       type=str,   help='The path of the model configuration. When this arg is set, it will cover max_nodes / channels / num_cells.')
  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
  parser.add_argument('--tau_min',            type=float,               help='The minimum tau for Gumbel')
  parser.add_argument('--tau_max',            type=float,               help='The maximum tau for Gumbel')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  # E2RNAS
  parser.add_argument('--nop_outer', default=False, action='store_true', help='use nop in outer loop')
  parser.add_argument('--flp_outer', default=False, action='store_true', help='use flp in outer loop')
  parser.add_argument('--adv_outer', default=False, action='store_true', help='use adv in outer loop')
  parser.add_argument('--ood_outer', default=False, action='store_true', help='use ood in outer loop')
  parser.add_argument('--ood_inner', default=False, action='store_true', help='use ood in inner loop')
  parser.add_argument('--MGDA', default=False, action='store_true', help='use MGDA')
  parser.add_argument('--grad_norm', type=str, default='none', choices=['none', 'lossplus', 'loss', 'l2'], help='use gradient normalization in MGDA')
  parser.add_argument('--nop_constrain', type=str, default='none', choices=['max', 'min', 'both', 'abs', 'none'], help='use constraint in model size')
  parser.add_argument('--nop_constrain_min', type=float, default=0, help='constrain the model size')
  parser.add_argument('--flp_constrain', type=str, default='none', choices=['max', 'min', 'both', 'abs', 'none'], help='use constraint in model size')
  parser.add_argument('--flp_constrain_min', type=float, default=0, help='constrain the model size')
  parser.add_argument('--epsilon', default=2, type=int)
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
