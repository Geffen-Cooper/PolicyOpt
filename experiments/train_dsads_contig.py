import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from utils.setup_funcs import *
from datasets.dsads_contig.dsads import *
from train import *
from models import *

activities = np.load("../datasets/dsads_contig/activity_list.npy")

if __name__ == '__main__':

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	logging_prefix = "dsads_contig"

	# hyperparameters
	EPOCHS = 50
	LR = 0.01
	MOMENTUM = 0.9
	WD = 1e-4
	ESE = 20
	BATCH_SIZE = 128

	seeds = [123,456,789]

	for seed_i,seed in enumerate(seeds):

		# setup the session
		logger = init_logger(f"{logging_prefix}/train_seed{seed}_activities_{activities}")
		init_seeds(seed)
		
		logger.info(f"Seed: {seed}")

		# create the datasets
		train_loader,val_loader,test_loader = load_dsads_person_dataset(BATCH_SIZE)

		# init models
		model = SimpleNet(3,len(np.unique(train_loader.dataset.raw_labels)))
		opt = torch.optim.SGD(model.parameters(),lr=LR,momentum=MOMENTUM,weight_decay=WD)

		# load the model if already trained
		ckpt_path = os.path.join(PROJECT_ROOT,f"saved_data/checkpoints/{logging_prefix}/seed{seed}_activities_{activities}.pth")
		if os.path.exists(ckpt_path):
			model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
		else:
			# otherwise train the model
			train(model,nn.CrossEntropyLoss(label_smoothing=0.1),opt,f"{logging_prefix}/seed{seed}_activities_{activities}",EPOCHS,ESE,device,
				train_loader,val_loader,logger,torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS),20)
			
		# load the one with the best validation accuracy and evaluate on test set
		model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])
		val_acc,val_f1,val_loss = validate(model, test_loader, device, nn.CrossEntropyLoss())
		logger.info(f"Test F1: {val_f1}")