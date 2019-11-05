import torch.nn as nn
from torchvision import models

class CorA_model(nn.Module):
	def __init__(self):
		super(CorA_model, self).__init__()
		self.restored = False

		# self.feature_src = nn.Sequential(
		# 	nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5)),  #64 28 28
		# 	nn.BatchNorm2d(64),
		# 	nn.ReLU(inplace=True),
		# 	nn.MaxPool2d(kernel_size=(2,2)),   #64 14 14
		# 	nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5)),  #64 10 10
		# 	nn.BatchNorm2d(64),
		# 	nn.Dropout2d(),
		# 	nn.ReLU(inplace=True),
		# 	nn.MaxPool2d(kernel_size=(2,2)),   #64 5 5
		# 	nn.ReLU(inplace=True),
		# )

		# self.feature_tgt = nn.Sequential(
		# 	nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5,5)),  #64 24 24
		# 	nn.BatchNorm2d(64),
		# 	nn.ReLU(inplace=True),
		# 	nn.MaxPool2d(kernel_size=(2,2)),   #64 12 12
		# 	nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5)),  #64 8 8
		# 	nn.BatchNorm2d(64),
		# 	nn.Dropout2d(),
		# 	nn.ReLU(inplace=True),
		# 	nn.MaxPool2d(kernel_size=(2,2)),   #64 4 4
		# 	nn.ReLU(inplace=True),
		# )

		# self.feature_change = nn.Sequential(
		# 	nn.Linear(64*5*5, 64*4*4),
		# 	nn.BatchNorm1d(1024),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(64*4*4, 64*4*4)
		# 	)

		# self.classifier = nn.Sequential(
		# 	nn.Linear(64*4*4, 1024),
		# 	nn.BatchNorm1d(1024),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(1024,256),
		# 	nn.BatchNorm1d(256),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(256,10),
			# nn.Sigmoid(),
		# )
		self.feature = nn.Sequential(
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 32 32
			nn.BatchNorm2d(64),
			# nn.InstanceNorm2d(64),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 32 32
			nn.BatchNorm2d(64),
			# nn.InstanceNorm2d(64),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 32 32
			nn.BatchNorm2d(64),
			# nn.InstanceNorm2d(64),
			nn.LeakyReLU(0.1, True),
			nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),  #64 16 16
			nn.Dropout2d(),

			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 16 16
			nn.BatchNorm2d(64),
			# nn.InstanceNorm2d(64),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 16 16
			nn.BatchNorm2d(64),
			# nn.InstanceNorm2d(64),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 16 16
			nn.BatchNorm2d(64),
			# nn.InstanceNorm2d(64),
			nn.LeakyReLU(0.1, True),
			nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),  #64 8 8 
			nn.Dropout2d(),
		)

		self.classifier = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 8 8 
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 8 8
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 8 8
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.1, True),
			nn.AvgPool2d((8, 8)),
		)

		self.classifier_out = nn.Sequential(
			nn.Linear(64, 10)
		)

		# self.ReconNet = nn.Sequential(
		# 	nn.Upsample(scale_factor=2),
			
		# 	nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 16 16
		# 	nn.BatchNorm2d(64),
		# 	nn.LeakyReLU(0.1, True),
		# 	nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 16 16
		# 	nn.BatchNorm2d(64),
		# 	nn.LeakyReLU(0.1, True),
		# 	nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 16 16
		# 	nn.BatchNorm2d(64),
		# 	nn.LeakyReLU(0.1, True),
			
		# 	nn.Upsample(scale_factor=2),
		# 	nn.Dropout2d(),

		# 	nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 32 32
		# 	nn.BatchNorm2d(64),
		# 	nn.LeakyReLU(0.1, True),
		# 	nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 32 32
		# 	nn.BatchNorm2d(64),
		# 	nn.LeakyReLU(0.1, True),
		# 	nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=(3,3), padding=(1,1)),  #3 32 32
		# )

	def forward(self, input_data, flag):
		# input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
		# if flag=="src":
		# 	feature = self.feature_src(input_data)
		# else:
		# 	feature = self.feature_src(input_data)
		# feature = feature.view(-1, 64*5*5)
		# feature = self.feature_change(feature)
		# class_output = self.classifier(feature)

		input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
		feature = self.feature(input_data)
		
		# image_ = self.ReconNet(feature)
		class_feature = self.classifier(feature)
		class_feature = class_feature.view(-1, 64)
		class_output = self.classifier_out(class_feature)

		return class_output, feature


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.restored = False

		# self.discriminator = nn.Sequential(
		# 	nn.Linear(64*4*4, 1024),
		# 	nn.BatchNorm1d(1024),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(1024, 256),
		# 	nn.BatchNorm1d(256),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(256, 2),
		# )		
		self.discriminator = nn.Sequential(
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 8 8 
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.1, True),
			nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 8 8
			nn.BatchNorm2d(64),
			nn.LeakyReLU(0.1, True),
			# nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=(1,1)),  #64 8 8
			# nn.BatchNorm2d(64),
			# nn.LeakyReLU(0.1, True),
			nn.AvgPool2d((8, 8)),
		)
		self.output = nn.Sequential(
			nn.Linear(64, 2),
			# nn.Sigmoid(),
		)

	def forward(self, feature):
		feature = self.discriminator(feature)
		feature = feature.view(-1, 64)
		domain_output = self.output(feature)

		return domain_output