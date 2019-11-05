import torch.utils.data
import torch.nn as nn 
import torch.optim as optim

def eval(CorA, Discriminator, data_loader, device, flag):
	CorA.eval()
	Discriminator.eval()

	loss = 0.0
	acc = 0.0
	acc_domain = 0.0
	cirterion = nn.CrossEntropyLoss()

	for (images, labels) in  data_loader:
		images = images.to(device)
		labels = labels.to(device)
		size = len(labels)

		domain = torch.ones(size).long().to(device)

		preds, feature = CorA(images,flag)
		domain_preds = Discriminator(feature)

		loss += cirterion(preds, labels).data.item()

		pred_cls = preds.data.max(1)[1]
		pred_domain = domain_preds.data.max(1)[1]
		acc += pred_cls.eq(labels.data).sum().item()

		# if(flag=="tgt"):
		# 	count = 0.0
		# 	count1 = 0.0
		# 	for k in range(len(pred_cls)):
		# 		if(max(preds[k])>=0.99):
		# 			count1 += 1
		# 			if pred_cls[k]==labels[k]:
		# 				count += 1		
		# 	if(count1>0):
		# 		print(count/count1)

		acc_domain += pred_domain.eq(domain.data).sum().item()

	loss /= len(data_loader)
	acc /= len(data_loader.dataset)
	acc_domain /= len(data_loader.dataset)

	print("Avg Loss = {:.6f}, Avg Accuracy = {:.2%}, Avg Domain Accuracy = {:2%}".format(loss, acc, acc_domain))

# def train_tgt(CorA, tgt_net, data_loader, data_loader_eval, device):
# 	CorA.eval()
# 	tgt_net.train()

	
# 	optimizer = optim.Adam(tgt_net.parameters(), lr=0.001)
# 	cirterion = nn.CrossEntropyLoss()
# 	for epoch in range(300):
# 		len_dataloader = len(data_loader)
# 		for step, (images, labels) in enumerate(data_loader):

# 			images = images.to(device)
# 			labels = labels.to(device)

# 			pred_labels, _ = CorA(images, 'tgt')
			
# 			pred_cls = pred_labels.data.max(1)[1]

# 			class_out = tgt_net(images)

# 			loss = cirterion(class_out, pred_cls)

# 			optimizer.zero_grad()
# 			loss.backward()
# 			optimizer.step()

# 			if(step % 20 == 0):
# 				print("[{:4d}/{}] [{:2d}/{}]: loss_D={:4f}".format(epoch + 1, 300, step+1, \
# 					len_dataloader, loss.data.item()))
# 		print(epoch)
# 		if( epoch%2== 0):
# 			acc = 0.0
# 			for (images, labels) in  data_loader_eval:
# 				images = images.to(device)
# 				labels = labels.to(device)
# 				preds = tgt_net(images)

# 				count = 0.0
# 				count1 = 0.0
# 				for k in range(len(labels)):
# 					if(max(preds[k])>=0.9):
# 						count1 += 1
# 						if pred_cls[k]==labels[k]:
# 							count += 1		
# 				if(count1>0):
# 					print(count, count1, count/count1)

# 				preds_CorA, _ = CorA(images,"tgt")
# 				pred_cls = preds.data.max(1)[1]
# 				pred_tmp = preds_CorA.data.max(1)[1]
# 				acc += pred_cls.eq(labels.data).sum().item()
# 				acc_tmp = pred_tmp.eq(labels.data).sum().item()
# 			acc /= len(data_loader_eval.dataset)
# 			acc_tmp /= len(data_loader_eval.dataset)
# 			print("Accuracy = {:.2%}, Accuracy_CorA = {:.2%}".format(acc, acc_tmp))
