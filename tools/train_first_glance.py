# coding=utf-8
import argparse
import os, sys
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import torch.nn.functional as F
import gc

from tensorboardX import SummaryWriter
writer=SummaryWriter(log_dir='./experiments/logs/train_first_glance')

import _init_paths

from utils.metrics import AverageMeter, accuracy, multi_scores
from networks.person_pair import person_pair as First_Glance
from dataset.loader import get_test_loader, get_train_loader

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Relationship')

"""The dataset file arguments.
"""
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('objects', metavar='DIR', help='path to objects (bboxes and categories)')
parser.add_argument('trainlist', metavar='DIR', help='path to train list')
parser.add_argument('testlist', metavar='DIR', help='path to test list')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (defult: 4)')

"""The training arguments including optimizer's argument.
"""
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
					help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='N',
					help='optimizer\'s learning rate in training')
parser.add_argument('-m', '--momentum', default=0.9, type=float, metavar='N',
					help='optimizer\'s momentum in training')
parser.add_argument('--wd', '--weight-decay', default=0.0001, type=float, metavar='N',
					help='optimizer\'s weight-decay in training')
parser.add_argument('-e', '--epoch', default=100, type=int, metavar='N',
					help='training epoch number')

"""Other argument.
"""
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
					help='print frequency (default: 10)')
parser.add_argument('--weights', default='', type=str, metavar='PATH',
					help='path to weights (default: none)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
					help='path to checkpoint weight (default: none)')
parser.add_argument('--scale-size',default=256, type=int,
					help='input size')
parser.add_argument('--world-size', default=1, type=int,
					help='number of distributed processes')
parser.add_argument('-n', '--num-classes', default=3, type=int, metavar='N',
					help='number of classes / categories')
parser.add_argument('--crop-size',default=224, type=int,
					help='crop size')
parser.add_argument('--result-path', default='', type=str, metavar='PATH',
					help='path for saving result (default: none)')

best_prec1 = 0


def main():
	# global args, best_prec1
	args = parser.parse_args()
	print(args)

	# Create dataloader.
	print '====> Creating dataloader...'
	train_loader = get_train_loader(args)
	test_loader = get_test_loader(args)

	# Load First Glance network.
	print '====> Loading the network...'
	model = First_Glance(num_classes=args.num_classes, pretrained=True)
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

	"""Load checkpoint weight of network.
	"""
	if args.checkpoint:
		if os.path.isfile(args.checkpoint):
			print("====> loading model '{}'".format(args.checkpoint))
			checkpoint = torch.load(args.checkpoint)
			# checkpoint_dict = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}
			model.load_state_dict(checkpoint)
		else:
			print("====> no pretrain model at '{}'".format(args.checkpoint))
	
	
	model.cuda()
	criterion = nn.CrossEntropyLoss().cuda()
	cudnn.benchmark = True

	# Train first-glance model.
	for epoch in range(args.epoch):
		_, _, prec_tri, rec_tri, ap_tri = train_eval(train_loader, test_loader, model, criterion, optimizer, args, epoch)
		_, _, prec_val, rec_val, ap_val = validate_eval(test_loader, model, criterion, args, epoch)

		# Print result.
		writer.add_scalars('Prec@1 (per epoch)', {'train': prec_tri.mean()}, epoch)
		writer.add_scalars('Recall (per epoch)', {'train': rec_tri.mean()}, epoch)
		writer.add_scalars('mAP (per epoch)', {'train': ap_tri.mean()}, epoch)

		writer.add_scalars('Prec@1 (per epoch)', {'valid': prec_val.mean()}, epoch)
		writer.add_scalars('Recall (per epoch)', {'valid': rec_val.mean()}, epoch)
		writer.add_scalars('mAP (per epoch)', {'valid': ap_val.mean()}, epoch)
		
		print('=======> Train Scores')
		print('Train Prec', prec_tri)
		print('Train Recall', rec_tri)
		print('Train AP', ap_tri)

		print('=======> Valid Scores')
		print('Valid Prec', prec_val)
		print('Valid Recall', rec_val)
		print('Valid AP', ap_val)

		print('=======> Mean Scores')
		print('[Epoch {0}]:\n  '
			'Train:\n    '
			'Prec@1 {1:.3f}\n    '
			'Recall {2:.3f}\n    '
			'mAP {3:.3f}\n  '
			'Valid:\n    '
			'Prec@1 {4:.3f}\n    '
			'Recall {5:.3f}\n    '
			'mAP {6:.3f}\n'.format(epoch, 
				prec_tri.mean(), rec_tri.mean(), ap_tri.mean(), 
				prec_val.mean(), rec_val.mean(), ap_val.mean()))

def train_eval(train_loader, val_loader, model, criterion, optimizer, args, epoch, fnames=[]):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	model.eval()

	end = time.time()
	scores = np.zeros((len(train_loader.dataset), args.num_classes))
	labels = np.zeros((len(train_loader.dataset), ))
	for i, (union, obj1, obj2, bpos, target, _, _, _) in enumerate(train_loader):
		target = target.cuda(async=True)
		union = union.cuda()
		obj1 = obj1.cuda()
		obj2 = obj2.cuda()
		bpos = bpos.cuda()
		
		output, _ = model(union, obj1, obj2, bpos)
		
		loss = criterion(output, target)
		losses.update(loss.item(), union.size(0))

		prec1 = accuracy(output.data, target)
		top1.update(prec1[0], union.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()
		
		if i % args.print_freq == 0:
			"""Every 10 batches, print on screen and print train information on tensorboard
			"""
			niter = epoch * len(train_loader) + i
			print('Train [Batch {0}/{1}|Epoch {2}/{3}]:  '
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
					'Loss {loss.val:.4f} ({loss.avg:.4f})  '
					'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
						i, len(train_loader), epoch, args.epoch, batch_time=batch_time,
						loss=losses, top1=top1))
			
		 	writer.add_scalars('Loss (per batch)', {'train-10b': loss.item()}, niter)
			writer.add_scalars('Prec@1 (per batch)', {'train-10b': prec1[0]}, niter)

		
		if i % (args.print_freq*10) == 0 :
			# Every 100 batches, print on screen and print validation information on tensorboard
			
			top1_avg_val, loss_avg_val, prec, recall, ap = validate_eval(val_loader, model, criterion, args, epoch)
			writer.add_scalars('Loss (per batch)', {'valid': loss_avg_val}, niter)
			writer.add_scalars('Prec@1 (per batch)', {'valid': top1_avg_val}, niter)

			# Save model every 100 batches.
			torch.save(model.state_dict(), args.weights)
			print('...Model saved')
		
		# Record scores.
		output_f = F.softmax(output, dim=1)  # To [0, 1]
		output_np = output_f.data.cpu().numpy()
		labels_np = target.data.cpu().numpy()
		b_ind = i*args.batch_size
		e_ind = b_ind + min(args.batch_size, output_np.shape[0])
		scores[b_ind:e_ind, :] = output_np
		labels[b_ind:e_ind] = labels_np

	
	res_scores = multi_scores(scores, labels, ['precision', 'recall', 'average_precision'])
	print('Train [Epoch {0}/{1}]:  '
		'*Time {2:.2f}mins ({batch_time.avg:.2f}s)  '
		'*Loss {loss.avg:.4f}  '
		'*Prec@1 {top1.avg:.3f}'.format(epoch, args.epoch, batch_time.sum/60,
			batch_time=batch_time, loss=losses, top1=top1))
	
	return top1.avg, losses.avg, res_scores['precision'], res_scores['recall'], res_scores['average_precision']

def validate_eval(val_loader, model, criterion, args, epoch=None, fnames=[]):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	model.eval()

	end = time.time()
	scores = np.zeros((len(val_loader.dataset), args.num_classes))
	labels = np.zeros((len(val_loader.dataset), ))
	for i, (union, obj1, obj2, bpos, target, _, _, _) in enumerate(val_loader):
		with torch.no_grad():
			target = target.cuda(async=True)
			union = union.cuda()
			obj1 = obj1.cuda()
			obj2 = obj2.cuda()
			bpos = bpos.cuda()

			output, _ = model(union, obj1, obj2, bpos)
			
			loss = criterion(output, target)
			losses.update(loss.item(), union.size(0))
			prec1 = accuracy(output.data, target)
			top1.update(prec1[0], union.size(0))

			batch_time.update(time.time() - end)
			end = time.time()

			# Record scores.
			output_f = F.softmax(output, dim=1)  # To [0, 1]
			output_np = output_f.data.cpu().numpy()
			labels_np = target.data.cpu().numpy()
			b_ind = i*args.batch_size
			e_ind = b_ind + min(args.batch_size, output_np.shape[0])
			scores[b_ind:e_ind, :] = output_np
			labels[b_ind:e_ind] = labels_np
	
	print('Test [Epoch {0}/{1}]:  '
		'*Time {2:.2f}mins ({batch_time.avg:.2f}s)  '
		'*Loss {loss.avg:.4f}  '
		'*Prec@1 {top1.avg:.3f}'.format(epoch, args.epoch, batch_time.sum/60,
			batch_time=batch_time, top1=top1, loss=losses))

	res_scores = multi_scores(scores, labels, ['precision', 'recall', 'average_precision'])
	return top1.avg, losses.avg, res_scores['precision'], res_scores['recall'], res_scores['average_precision']


if __name__=='__main__':
	main()
