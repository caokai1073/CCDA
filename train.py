import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from test import eval
from utils import save_model

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

def train_model(CorA, Discriminator, params, src_data_loader, src_data_loader_eval, \
	tgt_data_loader, tgt_data_loader_eval):
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	optimizer_CorA = optim.Adam(CorA.parameters(), lr=0.001)
	optimizer_D = optim.Adam(Discriminator.parameters(), lr=0.001)
	criterion = nn.CrossEntropyLoss()
	criterion_mse = nn.MSELoss()

# 10 samples with labels
	# flag = 0 
	# images_tgt_labeled = torch.zeros(10,3,32,32).to(device)
	# class_tgt =torch.zeros(10).to(device)
	# for img, class_0 in tgt_data_loader:

	# 	if flag==1:
	# 		break;
	# 	flag=1;
	# 	for i in range(10):
	# 		for j in range(params.batch_size):
	# 			if(class_0[j]==i):
	# 				images_tgt_labeled[i] = img[j].to(device)
	# 				class_tgt[i] = class_0[j].to(device)
	# 				break;
	# class_tgt = class_tgt.long()

	#aver_all_src = torch.zeros(10,1024).to(device)
	for epoch in range(params.num_epochs):
		# pre_believed_data = torch.zeros(1000,3,32,32).to(device)
		# pre_believed_label = torch.zeros(1000,1).to(device)
		# number = 0
		# number1 = 0
		CorA.train() #pre-train BN and dropout
		Discriminator.train()

		len_dataloader = min(len(src_data_loader), len(tgt_data_loader))
		data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
		
		for step, ((images_src, class_src), (images_tgt, _)) in data_zip:
			# print(torch.mean(images_src[0]))
			#print(step)
			# if flag==0:
			# 	Sdata = images_tgt[0]
			# 	flag = 1
			# 	print(Sdata.size())
			# 	for i in range(params.batch_size):
			# 		if images_tgt.eq(Sdata.data).sum == 32*32:
			# 			print("**********")

			# p = float(step + epoch * len_dataloader) / \
			# 	params.num_epochs / len_dataloader
			# alpha = 2. / (1. + np.exp(-10 * p)) - 1

			# adjust_learning_rate(optimizer_CorA, p)
			# adjust_learning_rate(optimizer_D, p)

			# CorA = nn.DataParallel(CorA)
			# CorA.to(device)

			images_src = images_src.to(device)
			class_src = class_src.to(device)
			images_tgt = images_tgt.to(device)
			# class_tgt = class_tgt.to(device)

			class_src_pred, feature_src = CorA(images_src, "src")
			class_tgt_pred, feature_tgt = CorA(images_tgt, "tgt")
			loss_feature_mse = torch.abs(criterion_mse(feature_src, torch.zeros(feature_src.size()).to(device))-criterion_mse(feature_tgt, torch.zeros(feature_src.size()).to(device)))
			
			d_src = torch.randn(images_src.size()).to(device)
			d_src /= torch.max(torch.abs(d_src))
			d_src /= torch.sqrt(1e-6+torch.sum(torch.pow(d_src,2)))
			d_src = 1e-6 * d_src

			d_tgt = torch.randn(images_src.size()).to(device)
			d_tgt /= torch.max(torch.abs(d_tgt))
			d_tgt /= torch.sqrt(1e-6+torch.sum(torch.pow(d_tgt,2)))
			d_tgt = 1e-6 * d_tgt


			logit_src, logit_feature_src = CorA(images_src + d_src, "src")
			logit_tgt, logit_feature_tgt = CorA(images_tgt + d_tgt, "tgt")

			# feature_dis_src = criterion_mse(feature_src, logit_feature_src)
			# feature_dis_tgt = criterion_mse(feature_tgt, logit_feature_tgt)
			# print(feature_dis_src, feature_dis_tgt)

			# print(class_tgt_pred)
			# print(logit_tgt)
			loss_KL_tgt = torch.sum(F.softmax(class_tgt_pred, dim=1)*torch.log(F.softmax(class_tgt_pred, dim=1)/ \
				F.softmax(logit_tgt, dim=1))) / params.batch_size 
			loss_KL_src = torch.sum(F.softmax(class_src_pred, dim=1)*torch.log(F.softmax(class_src_pred, dim=1)/ \
				F.softmax(logit_src, dim=1))) / params.batch_size 
			# loss_KL_tgt = torch.sum(torch.pow(class_tgt_pred-logit_tgt, 2)) / params.batch_size
			# loss_KL_src = torch.sum(torch.pow(class_src_pred-logit_src, 2)) / params.batch_size
			# print("KL:", loss_KL)
			# class_tgt_labeled_pred, class_tgt_feature, _ = CorA(images_tgt_labeled, "tgt")
			#if step % 5 == 0:
			#	print(step, class_tgt_labeled_pred.max(1)[1])
			#print(images_tgt.size())
			label_tgt = class_tgt_pred.data.max(1)[1]

				# count = 0.0
				# count1 = 0.0
				# for k in range(len(label_tgt)):
				# 	if(max(class_tgt_pred[k])>=0.999):
				# 		count1 += 1
				# 		if label_tgt[k]==class_tgt[k]:
				# 			count += 1		
				# if(count1>0):
				# 	print(count/count1)
				
			

			# look for probabilty bigger than 0.999
			# for index, label in enumerate(label_tgt):
			# 	if max(class_tgt_pred[index])>0.999 and number<1000:
			# 		pre_believed_data[number] = images_tgt[index]
			# 		pre_believed_label[number] = label
			# 		number += 1
			# 		if label==class_tgt[index]:
			# 			number1 += 1
			# if step==467:
			# 	tmp = pre_believed_data[0:number]
			# 	print(number1/number, number, tmp)
			# pre_believed_label = pre_believed_label.long()
			#print(feature_src[0].reshape(64*4*4, 1).mm(feature_src[0].reshape(1, 64*4*4)))
			prob_domain_src = Discriminator(feature_src)
			prob_domain_tgt = Discriminator(feature_tgt)
			# acc_domain = 0.0
			# size = 128
			# domain = torch.ones(size).long().to(device)
			# pred_domain_src = prob_domain_src.data.max(1)[1]
			# acc_domain_src = pred_domain_src.eq(domain.data).sum().item()
			# acc_domain_src /= 128
			# pred_domain_tgt = prob_domain_tgt.data.max(1)[1]
			# acc_domain_tgt = pred_domain_tgt.eq(domain.data).sum().item()
			# acc_domain_tgt /= 128
			# print("Source = {:2%}, target = {:2%}".format(acc_domain_src, acc_domain_tgt))

			logit_domain_src = Discriminator(logit_feature_src)
			logit_domain_tgt = Discriminator(logit_feature_tgt)

			# loss_KL_domain_src = torch.sum(torch.pow(prob_domain_src-logit_domain_src, 2)) / params.batch_size
			# loss_KL_domain_tgt = torch.sum(torch.pow(prob_domain_tgt-logit_domain_tgt, 2)) / params.batch_size
			# print(loss_KL_domain_src, loss_KL_domain_tgt)
			loss_KL_domain_src = torch.sum(F.softmax(prob_domain_src, dim=1)*torch.log(F.softmax(prob_domain_src, dim=1)/ \
				F.softmax(logit_domain_src, dim=1))) / params.batch_size
			loss_KL_domain_tgt = torch.sum(F.softmax(prob_domain_tgt, dim=1)*torch.log(F.softmax(prob_domain_tgt, dim=1)/ \
				F.softmax(logit_domain_tgt, dim=1))) / params.batch_size

			feature_src = feature_src.view(-1,64*8*8)
			feature_tgt = feature_tgt.view(-1,64*8*8)

			sum_tgt = torch.zeros(10, 64*8*8).to(device)
			sum_total_tgt = torch.zeros(1, 64*8*8).to(device)
			
			count = torch.zeros(10)
			for index , label in enumerate(label_tgt):
				sum_tgt[label] += feature_tgt[index]     
				sum_total_tgt += feature_tgt[index]     
				count[label] += 1

			aver_tgt = torch.zeros(10, 64*8*8).to(device)
			aver_total_tgt = torch.zeros(1,64*8*8).to(device)
			none_data_tgt = []
			for label in range(10):
				if count[label]==0:
					none_data_tgt.append(label)
				else:
					aver_tgt[label] = sum_tgt[label] / count[label]

			aver_total_tgt = sum_total_tgt / params.batch_size

			# St_tgt = torch.zeros(64*8*8, 64*8*8).to(device)
			Sb_tgt = torch.zeros(64*8*8, 64*8*8).to(device)
			Sw_tgt = torch.zeros(64*8*8, 64*8*8).to(device)

			# for i in range(params.batch_size):
			# 	St_tgt += (feature_tgt[i]-aver_total_tgt).reshape(64*8*8, 1).mm((feature_tgt[i]-aver_total_tgt).reshape(1, 64*8*8))
			# #print(St)

			for label in range(10):
				if label not in none_data_tgt:
					Sb_tgt += count[label]*(aver_tgt[label]-aver_total_tgt).reshape(64*8*8, 1).mm((aver_tgt[label]-aver_total_tgt).reshape(1, 64*8*8))

			for index, label in enumerate(label_tgt):
					Sw_tgt += (feature_tgt[index] - aver_tgt[label]).reshape(64*8*8, 1).mm((feature_tgt[index] - aver_tgt[label]).reshape(1, 64*8*8))
			# print(none_data_tgt)

			# # norm_sum = torch.zeros(1).to(device)
			# # for label in range(10):
			# # 	norm_sum += torch.norm(aver_tgt[label], p=2)


			
			sum_src = torch.zeros(10, 64*8*8).to(device)
			sum_total_src = torch.zeros(1, 64*8*8).to(device)
			count = torch.zeros(10)
			for index , label in enumerate(class_src):
				sum_src[label] += feature_src[index]     
				sum_total_src += feature_src[index]     
				count[label] += 1
			

			aver_src = torch.zeros(10, 64*8*8).to(device)
			aver_total_src = torch.zeros(1,64*8*8).to(device)
			none_data_src = []
			for label in range(10):
				if count[label]==0:
					none_data_src.append(label)
				else:
					aver_src[label] = sum_src[label] / count[label]

			aver_total_src = sum_total_src / params.batch_size
			#print(aver_total_src)

			St_src = torch.zeros(64*8*8, 64*8*8).to(device)
			Sb_src = torch.zeros(64*8*8, 64*8*8).to(device)
			Sw_src = torch.zeros(64*8*8, 64*8*8).to(device)

			# for i in range(params.batch_size):
			# 	St_src += (feature_src[i]-aver_total_src).reshape(64*8*8, 1).mm((feature_src[i]-aver_total_src).reshape(1, 64*8*8))
			# #print(St)

			for label in range(10):
				if label not in none_data_src:
					Sb_src += count[label]*(aver_src[label]-aver_total_src).reshape(64*8*8, 1).mm((aver_src[label]-aver_total_src).reshape(1, 64*8*8))

			for index, label in enumerate(class_src):
					Sw_src += (feature_src[index] - aver_src[label]).reshape(64*8*8, 1).mm((feature_src[index] - aver_src[label]).reshape(1, 64*8*8))

			
			# norm_sum = torch.zeros(1).to(device)
			# for label in range(10):
			# 	norm_sum += torch.norm(aver_src[label], p=2)

			# # print("aver_total_norm: {:4f}".format(torch.norm(aver_total, p=2)))
			
			# # print(norm_sum)
			# # pre_preds, _ = CorA(pre_believed_data)
			# # loss_pre = criterion(pre_preds, pre_believed_label)
			# distance_loss = torch.zeros(1).to(device)
			# for i in range(10):
			# 	for j in range(10):
			# 		if j is not i:
			# 			distance_loss += torch.norm(class_tgt_feature[i]-class_tgt_feature[j], p=2)
			# # print(distance_loss)

			for i in none_data_src:
				aver_tgt[i] = torch.zeros(64*8*8).to(device)
			for i in none_data_tgt:
				aver_src[i] = torch.zeros(64*8*8).to(device)
			
			center_loss = criterion_mse(aver_src, aver_tgt)
			
			size = len(class_src)
			loss_cls = criterion(class_src_pred, class_src)
			loss_domain = criterion(prob_domain_tgt, torch.ones(size).long().to(device))
			# loss_domain = torch.mean(torch.pow(1.0 - prob_domain_tgt, 2))

			if len(none_data_tgt)==9:
				loss_lda_tgt = 0
			else:
				loss_lda_tgt = torch.trace(Sw_tgt) / torch.trace(Sb_tgt)

			if len(none_data_src)==9:
				loss_lda_src = 0
			else:
				loss_lda_src = torch.trace(Sw_src) / torch.trace(Sb_src)

			tgt_loss_crossentropy = torch.mean(-torch.sum(F.softmax(class_tgt_pred, dim=1)*torch.log(F.softmax(class_tgt_pred, dim=1)), dim=1))
			# tgt_loss_crossentropy = torch.mean(1.0 - F.softmax(class_tgt_pred, dim=1).max(1)[0])
			# labeled_loss = criterion(class_tgt_labeled_pred, class_tgt)
			# print(labeled_loss)
			# cluster_loss = torch.zeros(1).to(device)
			# for i in range(10):
			# 	if i not in none_data_src:
			# 		cluster_loss += torch.norm(aver_src[i]-class_tgt_feature, p=2)
			# 	if i not in none_data_tgt:
			# 		cluster_loss += torch.norm(aver_tgt[i]-class_tgt_feature, p=2)
			# print(cluster_loss)
			# Recon_loss = criterion_mse(images_tgt, images_tgt_)
			# print(Recon_loss)
			# print(Sw_tgt, Sb_tgt, Sw_src, Sb_src, loss_lda_tgt, loss_lda_src)
			CorA_loss =  loss_cls + 0.1*loss_domain + loss_KL_src + 0.1*(loss_KL_tgt + tgt_loss_crossentropy) \
			+ 0.02*loss_lda_tgt + 0.02*loss_lda_src + 0.05*center_loss#+ loss_feature_mse #+ labeled_loss#+ 0.1*Recon_loss
			# print(loss_cls, loss_domain, loss_KL_src, loss_KL_tgt, tgt_loss_crossentropy)
			#+ alpha * labeled_loss + 0.05*cluster_loss - 0.5 * distance_loss + 0.5 * tgt_loss_crossentropy
			#+ 0.1*loss_lda_tgt  + alpha * labeled_loss + + 0.1*loss_lda_src + 2 * loss_pre
			#+ 0.01*torch.norm(aver_total, p=2) - 0.005*norm_sum

			# D_loss_src = - torch.mean(torch.pow(prob_domain_src, 2))
			# D_loss_tgt = - torch.mean(torch.pow(1.0 - prob_domain_tgt, 2))
			D_loss_src = criterion(prob_domain_src, torch.ones(size).long().to(device))
			D_loss_tgt = criterion(prob_domain_tgt, torch.zeros(size).long().to(device))
			D_loss = D_loss_src + D_loss_tgt + max(0.1, 1-0.1*epoch)*(loss_KL_domain_tgt + loss_KL_domain_src)
			# print(D_loss_src, D_loss_tgt, loss_KL_domain_src)

			# nn.utils.clip_grad_norm_(CorA.parameters(), 5)
			# nn.utils.clip_grad_norm_(Discriminator.parameters(), 5)

			optimizer_D.zero_grad()
			D_loss.backward(retain_graph=True)
			optimizer_D.step()

			optimizer_CorA.zero_grad()
			CorA_loss.backward()
			optimizer_CorA.step()

			if ((step+1) % params.log_step == 0):
				print("[{:4d}/{}] [{:2d}/{}]: loss_cls={:4f}, loss_domain={:4f}, crossentropy={:4f}, loss_KL_src={:4f}, loss_KL_tgt={:4f}, loss_CorA={:4f}, loss_D={:4f}".format(epoch + 1, params.num_epochs, step+1, \
					len_dataloader, loss_cls.data.item(), loss_domain.data.item(), tgt_loss_crossentropy.data.item(), loss_KL_src.data.item(), loss_KL_tgt.data.item(), CorA_loss.data.item(), D_loss.data.item()))

		# print("Sw: {:4f}".format(torch.trace(Sw_tgt)))
		# print("Sb: {:4f}".format(torch.trace(Sb_tgt)))

		if ((epoch+1) % params.eval_step == 0):
			print("eval on source domain")
			eval(CorA, Discriminator, src_data_loader_eval, device, "src")
			print("eval on target domain")
			eval(CorA, Discriminator, tgt_data_loader_eval, device, "tgt")

		if ((epoch +1) % params.save_step == 0):
			save_model(CorA, params.model_root, "CorA-{}.pt".format(epoch+1))
			save_model(Discriminator, params.model_root, "Discriminator-{}.pt".format(epoch+1))
		

	save_model(CorA, params.model_root, "CorA.pt")
	save_model(Discriminator, params.model_root, "Discriminator.pt")

	return CorA, Discriminator

def adjust_learning_rate(optimizer, p):
	lr_0 = 0.01
	alpha = 10
	beta = 0.75
	lr = lr_0 / (1 + alpha * p)**beta
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr



