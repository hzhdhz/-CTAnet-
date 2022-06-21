from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import scipy.io as sio
import torchvision.transforms.functional as F
class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta

        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, train_dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get train data loader
        # train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                       batch_size=self.batch_size,
                                       pin_memory=True,
                                       shuffle=True)

        # Set device for network
        net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()

        # Set loss
        criterion = nn.MSELoss(reduction='none')
        # Set device
        criterion = criterion.to(self.device)


        for epoch in range(self.n_epochs):
            self.c = self.init_center_c(train_loader, net)
            # print('epoch===========================', epoch)

            scheduler.step()
            if epoch != 0 and epoch // 50 == 0:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                # print('data===========================')
                inputs, _, semi_targets, _ , _, _= data



                inputs, semi_targets = inputs.to(self.device), semi_targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()
                outputs, inputs_rec = net(inputs)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                rec_loss = criterion(inputs_rec, inputs)
                losses_rec = torch.mean(rec_loss)
                loss = losses_rec + 0.1 * torch.mean(losses) #* 10


                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    def test(self, test_dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Get test data loader
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size = self.batch_size)
        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data, labels, semi_targets, idx, x, y in test_loader:
                # print('n_batches===========================', n_batches)
                inputs = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                semi_targets = semi_targets.to(self.device)
                idx = idx.to(self.device)
                x = x.to(self.device)
                y = y.to(self.device)

                outputs, inputs_rec = net(inputs)

                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                # dist = torch.sum(torch.sum((outputs - self.c) ** 2, dim=1), dim=1)
                losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist(),
                                            x.cpu().data.numpy().tolist(),
                                            y.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores, x, y = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        x = np.array(x)
        y = np.array(y)
        self.test_auc = roc_auc_score(labels, scores)
        #############################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #############################################%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')




    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0

        #c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        i = 0
        with torch.no_grad():
            for data, _, _, _, _, _ in train_loader:
                # get the inputs of the batch
                inputs = data
                # inputs = F.resize(inputs, size=(40, 40))
                inputs = inputs.to(self.device)
                outputs, inputs_rec = net(inputs)
                # outputs = net(inputs)
                n_samples = n_samples + outputs.shape[0]
                if i == 0:
                    c = torch.sum(outputs, dim=0)
                else:
                    c = c + torch.sum(outputs, dim=0)
                i = i + 1

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
