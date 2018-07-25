# coding=utf-8
class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

import sklearn.metrics as metrics


def multi_scores(pre_scores, labels, options=['precision', 'recall', 'average_precision']):
	"""Make use of metrics.precision_score, metrics.recall_score, metrics.average_precision_score
	"""
	result = {}
	for op in options:
		if op == 'precision':
			metrics_tool = metrics.precision_score
		elif op == 'recall':
			metrics_tool = metrics.recall_score
		elif op == 'average_precision':
			metrics_tool = metrics.average_precision_score
		else:
			result.append({})
			continue

		scores = np.zeros((pre_scores.shape[1], ))
		for l in range(pre_scores.shape[1]):
			scores[l] = metrics_tool((labels == l).astype(int), pre_scores[:, l])
		
		result[op] = scores
	
	return result
