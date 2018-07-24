import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset import SRDataset

def get_test_loader(data_dir, objects_dir, test_list, scale_size, crop_size, workers, batch_size):
	"""Create dataset and return dataset loader of test Dataset.

	Args:
		data_dir: The directory of the dataset image.
		objects_dir: The directory of the ROI object extracted from image by Faster RCNN.
		test_list: The file path of annotation list with the unit of content: image_id, box1, box2, label.
		scale_size: Scale size of transform.
		crop_size: Crop size of trnasform.

	Returns:
		test_laoder: [torch.utils.data.Loader] loader data in batch size.

	"""

	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	test_data_transform = transforms.Compose([
			transforms.Scale((crop_size, crop_size)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip

	test_full_transform = transforms.Compose([
			transforms.Scale((448, 448)),
			transforms.ToTensor(),
			normalize])  # what about horizontal flip

	test_set = SRDataset(data_dir, objects_dir, test_list, test_data_transform, test_full_transform )
	test_loader = DataLoader(dataset=test_set, num_workers=workers,
							batch_size=batch_size, shuffle=False)
	return test_loader
