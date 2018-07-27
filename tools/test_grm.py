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

import _init_paths

from utils.metrics import AverageMeter, accuracy
from networks.GRM import GRM
from dataset.loader import get_test_loader

model_names = sorted(name for name in models.__dict__
	if name.islower() and not name.startswith("__")
	and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Relationship')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('objects', metavar='DIR', help='path to objects (bboxes and categories)')
parser.add_argument('testlist', metavar='DIR', help='path to test list')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (defult: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
					help='mini-batch size (default: 1)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
					help='print frequency (default: 10)')
parser.add_argument('--weights', default='', type=str, metavar='PATH',
					help='path to weights (default: none)')
parser.add_argument('--scale-size',default=256, type=int,
					help='input size')
parser.add_argument('--world-size', default=1, type=int,
					help='number of distributed processes')
parser.add_argument('-n', '--num-classes', default=3, type=int, metavar='N',
					help='number of classes / categories')
parser.add_argument('--write-out', dest='write_out', action='store_true',
					help='write scores')
parser.add_argument('--adjacency-matrix', default='', type=str, metavar='PATH',
					help='path to adjacency-matrix of graph')
parser.add_argument('--crop-size',default=224, type=int,
					help='crop size')
parser.add_argument('--result-path', default='', type=str, metavar='PATH',
					help='path for saving result (default: none)')

best_prec1 = 0


def main():
	global args, best_prec1
	args = parser.parse_args()
	print(args)

	# Create dataloader.
	print '====> Creating dataloader...'
	test_loader = get_test_loader(args)

	# Load GRM network.
	print '====> Loading the network...'
	model = GRM(num_classes=args.num_classes, adjacency_matrix=args.adjacency_matrix)
	# print model

	# Load fine-tuned weight of network.
	if args.weights:
		if os.path.isfile(args.weights):
			print("====> loading model '{}'".format(args.weights))
			checkpoint = torch.load(args.weights)
			checkpoint_dict = {k.replace('module.',''):v for k,v in checkpoint['state_dict'].items()}
			model.load_state_dict(checkpoint_dict)
		else:
			print("====> no pretrain model at '{}'".format(args.weights))
	
	model.fg = torch.nn.DataParallel(model.fg)
	model.full_im_net = torch.nn.DataParallel(model.full_im_net)
	model.cuda()

	criterion = nn.CrossEntropyLoss().cuda()
	
	cudnn.benchmark = True

	fnames = []
	if args.write_out:
		print '------Write out result---'
		for i in range(args.num_classes):
			fnames.append(open(args.result_path + str(i) + '.txt', 'w'))
	
	validate(test_loader, model, criterion, fnames)

	if args.write_out:
		for i in range(args.num_classes):
			fnames[i].close()
	return

def validate(val_loader, model, criterion, fnames=[]):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()

	model.eval()

	end = time.time()
	tp = {} # precision
	p = {}  # prediction
	r = {}  # recall
	for i, (union, obj1, obj2, bpos, target, full_im, bboxes_14, categories) in enumerate(val_loader):
		batch_size = bboxes_14.size(0)
		cur_rois_sum = categories[0,0].item()
		bboxes = bboxes_14[0, 0:categories[0,0].item(), :]
		for b in range(1, batch_size):
			bboxes = torch.cat((bboxes, bboxes_14[b, 0:categories[b,0], :]), 0)
			cur_rois_sum += categories[b,0].item()
		assert(bboxes.size(0) == cur_rois_sum), 'Bboxes num must equal to categories num'
		target = target.cuda(async=True)
		union_var = torch.autograd.Variable(union, volatile=True).cuda()
		obj1_var = torch.autograd.Variable(obj1, volatile=True).cuda()
		obj2_var = torch.autograd.Variable(obj2, volatile=True).cuda()
		bpos_var = torch.autograd.Variable(bpos, volatile=True).cuda()
		full_im_var = torch.autograd.Variable(full_im, volatile=True).cuda()
		bboxes_var = torch.autograd.Variable(bboxes, volatile=True).cuda()
		categories_var = torch.autograd.Variable(categories, volatile=True).cuda()
		
		target_var = torch.autograd.Variable(target, volatile=True)

		output = model(union_var, obj1_var, obj2_var, bpos_var, full_im_var, bboxes_var, categories_var)
		
		loss = criterion(output, target_var)
		losses.update(loss.data[0], union.size(0))
		prec1 = accuracy(output.data, target)
		top1.update(prec1[0], union.size(0))

		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.print_freq == 0:
			print('Test: [{0}/{1}]\t'
					'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					'Prec@1 {top1.val[0]:.3f} ({top1.avg[0]:.3f})\t'.format(
						i, len(val_loader), batch_time=batch_time,
						loss=losses, top1=top1))

		#####################################
		## write scores
		if args.write_out:
			output_f = F.softmax(output, dim=1)
			output_np = output_f.data.cpu().numpy()
			pre = np.argmax(output_np[0])
			t = target.data.cpu().numpy()[0]
			if r.has_key(t):
				r[t] += 1
			else:
				r[t] = 1
			if p.has_key(pre):
				p[pre] += 1
			else:
				p[pre] = 1
			if pre == t:
				if tp.has_key(t):
					tp[t] += 1
				else:
					tp[t] = 1
			
			for j in range(args.num_classes):
				fnames[j].write(str(output_np[0][j]) + '\n')
		#####################################

	print 'tp: ', tp
	print 'p: ', p
	print 'r: ', r
	precision = {}
	recall = {}
	for k in tp.keys():
		precision[k] = float(tp[k]) / float(p[k])
		recall[k] = float(tp[k]) / float(r[k])
	print 'precision: ', precision
	print 'recall: ', recall

	print(' * Prec@1 {top1.avg[0]:.3f}\t * Loss {loss.avg:.4f}'.format(top1=top1, loss=losses))
	return top1.avg[0]


if __name__=='__main__':
	main()
