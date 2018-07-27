# coding=utf-8
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


import torch.utils.model_zoo as model_zoo
import torchvision.models as models

class person_pair(nn.Module):
	def __init__(self, num_classes = 3, pretrained=False):
		super(person_pair, self).__init__()
		# Resnet models.
		self.resnet101_union = models.resnet101(pretrained=pretrained)
		self.resnet101_union = nn.Sequential(*list(self.resnet101_union.children())[:-1])

		self.resnet101_a = models.resnet101(pretrained=pretrained)
		self.resnet101_a = nn.Sequential(*list(self.resnet101_a.children())[:-1])

		self.resnet101_b = self.resnet101_a

		# Fc classifier.
		self.bboxes = nn.Linear(10, 256)
		self.fc6 = nn.Linear(2048+2048+2048+256, 4096)
		self.fc7 = nn.Linear(4096, num_classes)
		self.ReLU = nn.ReLU(False)
		self.Dropout = nn.Dropout()
		
	   	if not pretrained:
			self._initialize_weights()
	else:
		self.init_linear_weight(self.bboxes)
			self.init_linear_weight(self.fc6)
			self.init_linear_weight(self.fc7)
		
		

	# x1 = pu, x2 = p1, x3 = p2, x4 = bbox geometric info
	def forward(self, x1, x2, x3, x4): 
		x1 = self.resnet101_union(x1).view(-1, 2048)
		x2 = self.resnet101_a(x2).view(-1, 2048)
		x3 = self.resnet101_b(x3).view(-1, 2048)
		x4 = self.bboxes(x4)
	
		x = torch.cat((x4, x1, x2, x3), 1)
		x = self.Dropout(x)
		fc6 = self.fc6(x)
		x = self.ReLU(fc6)
		x = self.Dropout(x)
		x = self.fc7(x)

		return x, fc6

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2./n))
				if m.bias is not None:
					m.bias.data.zero_()

			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()

	def init_linear_weight(self, m):
		if type(m) == nn.Linear:
			m.weight.data.normal_(0, 0.01)
			m.bias.data.zero_()

