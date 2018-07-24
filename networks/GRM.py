# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from person_pair import person_pair
from ggnn import GGNN
from torch.distributions import Bernoulli
from vgg_v1 import vgg16_rois_v1
import math

class GRM(nn.Module):
	"""Graph Reasoning Model.

	Args:
		num_classes:
		ggnn_hidden_channel:
		ggnn_output_channel:
		time_step:
		attr_num:
		adjacency_matrix:
	
	Attributes:
		_num_classes:
		_ggnn_hidden_channel:
		_ggnn_output_channel:
		_time_step:
		_adjacency_matrix:
		_attr_num:
		_graph_num: Node number of ggnn graph.

		fg:
		full_im_net:
		ggnn:
		classifiler:
		ReLU:

	"""
	def __init__(self, num_classes = 3,
				ggnn_hidden_channel = 4098,
				ggnn_output_channel = 512, time_step = 3,
				attr_num = 80, adjacency_matrix=''):
		super(GRM, self).__init__()
		self._num_classes = num_classes
		self._ggnn_hidden_channel = ggnn_hidden_channel
		self._ggnn_output_channel = ggnn_output_channel
		self._time_step = time_step
		self._adjacency_matrix = adjacency_matrix
		self._attr_num = attr_num
		self._graph_num = attr_num + num_classes
		

		self.fg = person_pair(num_classes)  # First glance.

		self.full_im_net = vgg16_rois_v1(pretrained=False)

		self.ggnn = GGNN( hidden_state_channel = self._ggnn_hidden_channel,
			output_channel = self._ggnn_output_channel,
			time_step = self._time_step,
			adjacency_matrix=self._adjacency_matrix,
			num_classes = self._num_classes)

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(self._ggnn_output_channel * (self._attr_num + 1) , 4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, 1)
		)

		self.ReLU = nn.ReLU(True)

		self._initialize_weights()

	def forward(self, union, b1, b2, b_geometric, full_im, rois, categories):
		"""GRM forward.

		Args:
			union:
			b1:
			b2:
			b_geometric:
			full_im:
			rois:
			categories:

		Returns:
			final_scores:

		"""
		batch_size = union.size()[0]
		rois_feature = self.full_im_net(full_im, rois, categories)  # roi_feature has size of [object_num x box_size].
		contextual = Variable(torch.zeros(batch_size, self._graph_num, self._ggnn_hidden_channel), requires_grad=False).cuda()
		contextual[:, 0:self._num_classes, 0] = 1.  # [1, 0], relationship nodes.
		contextual[:, self._num_classes:, 1] = 1.  # [0, 1], object nodes.

		start_idx = 0
		end_idx = 0


		# Initial GGNN graph hidden status, fill the roi features to object nodes according to categories.
		for b in range(batch_size):
			cur_rois_num = categories[b, 0].data[0]  # Rois object number. It is not fixed.
			end_idx += cur_rois_num
			idxs = categories[b, 1:(cur_rois_num+1)].data.tolist()
			for i in range(cur_rois_num):
				contextual[b, int(idxs[i])+self._num_classes, 2:] = rois_feature[start_idx+i, :]  # Fill the roi features to object nodes according to categories.
			start_idx = end_idx

		# First glance scores.
		scores, fc7_feature = self.fg(union, b1, b2, b_geometric)
		
		# GGNN input, fill the pair feature to all the relationship nodes.
		fc7_feature_norm_enlarge = fc7_feature.view(batch_size, 1, -1).repeat(1, self._num_classes, 1)
		contextual[:, 0: self._num_classes, 2:] = fc7_feature_norm_enlarge  # Fill the pair feature to all the relationship nodes.
		ggnn_input = contextual.view(batch_size, -1)

		# GGNN forward.
		ggnn_feature = self.ggnn(ggnn_input)
		ggnn_feature_norm = ggnn_feature.view(batch_size * self._num_classes, -1)  # With size of [batch_size * num_classes * output_channel].

		# Classifier.
		final_scores = self.classifier(ggnn_feature_norm).view(batch_size, -1)
		
		return final_scores

	def _initialize_weights(self):
		for m in self.classifier.modules():
			cnt = 0
			if isinstance(m, nn.Linear):
				if cnt == 0:
					m.weight.data.normal_(0, 0.001)
				else :
					m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()
				cnt += 1
