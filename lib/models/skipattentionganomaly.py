""" GANomaly """
# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915

##
from collections import OrderedDict
import os
import time
import numpy as np

import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from lib.models.networks import define_G, define_D, get_scheduler
from lib.loss import l2_loss
from lib.evaluate import roc
from lib.models.basemodel import BaseModel

import scipy.io as io

from thop import profile
from thop import clever_format
import torch


class Skipattentionganomaly(BaseModel):
    """Skip-Attention Ganomaly Class"""

    @property
    def name(self):
        return 'skip-attention-ganomaly'

    def __init__(self, opt, data=None):
        super(Skipattentionganomaly, self).__init__(opt, data)
        ##

        # -- Misc attributes
        self.add_noise = True
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = define_G(self.opt, norm='batch', use_dropout=False, init_type='normal')
        self.netd = define_D(self.opt, norm='batch', use_sigmoid=False, init_type='normal')

        # input = torch.randn(1, 3, 32, 32)
        # macs, params = profile(self.netg, inputs=(input,))
        # # macs, params = clever_format([macs, params], "%.3f")
        # print(macs)
        # print(params)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        if self.opt.verbose:
            print(self.netg)
            print(self.netd)

        ##
        # Loss Functions
        self.l_adv = nn.BCELoss()
        self.l_con = nn.L1Loss()
        self.l_lat = l2_loss

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        self.noise = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize, self.opt.isize), dtype=torch.float32,
                                 device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize, self.opt.isize),
                                       dtype=torch.float32, device=self.device)
        self.real_label = torch.ones(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)

        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizers = []
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_d)
            self.optimizers.append(self.optimizer_g)
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def forward(self):
        self.forward_g()
        self.forward_d()

    def forward_g(self):
        """ Forward propagate through netG
        """
        # lgl
        self.fake = self.netg(self.input + self.noise)

    def forward_d(self):
        """ Forward propagate through netD
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    def backward_g(self):
        """ Backpropagate netg
        """
        self.err_g_adv = self.opt.w_adv * self.l_adv(self.pred_fake, self.real_label)
        self.err_g_con = self.opt.w_con * self.l_con(self.fake, self.input)
        self.err_g_lat = self.opt.w_lat * self.l_lat(self.feat_fake, self.feat_real)
        # lgl
        # self.err_consistency = self.opt.w_lat * self.l_lat(self.feat_fake, self.latent)
        # print(self.err_consistency)

        self.err_g = self.err_g_adv + self.err_g_con + self.err_g_lat

        self.err_g.backward(retain_graph=True)

    def backward_d(self):
        # Fake
        pred_fake, _ = self.netd(self.fake.detach())
        self.err_d_fake = self.l_adv(pred_fake, self.fake_label)

        # Real
        # pred_real, feat_real = self.netd(self.input)
        self.err_d_real = self.l_adv(self.pred_real, self.real_label)

        # Combine losses.
        self.err_d = self.err_d_real + self.err_d_fake + self.err_g_lat
        self.err_d.backward(retain_graph=True)

    def update_netg(self):
        """ Update Generator Network.
        """
        self.optimizer_g.zero_grad()
        self.backward_g()
        self.optimizer_g.step()

    def update_netd(self):
        """ Update Discriminator Network.
        """
        self.optimizer_d.zero_grad()
        self.backward_d()
        self.optimizer_d.step()
        if self.err_d < 1e-5: self.reinit_d()

    ##
    def optimize_params(self):
        """ Optimize netD and netG  networks.
        """
        self.forward()
        self.update_netg()
        self.update_netd()

    ##
    def test(self, plot_hist=True):
        """ Test GANomaly model.

        Args:
            data ([type]): Dataloader for the test set

        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                if os.path.exists(self.opt.load_weights):
                    self.load_weights(path=self.opt.load_weights)
                else:
                    self.load_weights(is_best=True)

            self.opt.phase = 'test'

            scores = {}
            file_names = []
            bboxx1 = []
            bboxx2 = []
            bboxy1 = []
            bboxy2 = []
            label_txts = []
            patch_ids = []
            global_x1y1x2y2s = []
            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.data.valid.dataset),), dtype=torch.long, device=self.device)
            self.features = torch.zeros(size=(len(self.data.valid.dataset), self.opt.nz), dtype=torch.float32,
                                        device=self.device)

            print("   Testing %s" % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(tqdm(self.data.valid, leave=False, total=len(self.data.valid)), 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()

                if self.opt.dataset == "atz":
                    meta = data[1]
                    file_names.extend(meta['current_file'])
                    bboxx1.extend(meta['x1'])
                    bboxx2.extend(meta['x2'])
                    bboxy1.extend(meta['y1'])
                    bboxy2.extend(meta['y2'])
                    label_txts.extend(meta['label_txt'])
                    patch_ids.extend(meta['patch_id'])
                    global_x1y1x2y2s.extend(meta['global_x1y1x2y2'])
                    data = data[0]

                # Forward - Pass
                self.set_input(data)
                self.fake = self.netg(self.input)

                _, self.feat_real = self.netd(self.input)
                _, self.feat_fake = self.netd(self.fake)

                # Calculate the anomaly score.
                si = self.input.size()
                sz = self.feat_real.size()
                rec = (self.input - self.fake).view(si[0], si[1] * si[2] * si[3])
                # save_rec = np.array(rec.cpu().detach())
                # io.savemat('save_rec.mat', {'save_rec': save_rec})
                lat = (self.feat_real - self.feat_fake).view(sz[0], sz[1] * sz[2] * sz[3])
                rec = torch.mean(torch.pow(rec, 2), dim=1)
                lat = torch.mean(torch.pow(lat, 2), dim=1)
                error = 0.9 * rec + 0.1 * lat

                time_o = time.time()

                self.an_scores[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                self.gt_labels[i * self.opt.batchsize: i * self.opt.batchsize + error.size(0)] = self.gt.reshape(
                    error.size(0))

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst): os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(self.input.data, '%s/real_%03d.eps' % (dst, i + 1), normalize=True)
                    vutils.save_image(self.fake.data, '%s/fake_%03d.eps' % (dst, i + 1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (
                    torch.max(self.an_scores) - torch.min(self.an_scores)
                    + torch.tensor(0.000000001, dtype=torch.float32, device=self.device))
            saveto = os.path.join(self.opt.outf, self.opt.name, self.opt.phase)
            # auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            auc = roc(self.gt_labels, self.an_scores)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), ('AUC', auc)])

            ##
            # PLOT HISTOGRAM
            if plot_hist:
                plt.ion()
                # Create data frame for scores and labels.
                scores['scores'] = self.an_scores.cpu()
                scores['labels'] = self.gt_labels.cpu()
                scores['x1'] = bboxx1
                scores['x2'] = bboxx2
                scores['y1'] = bboxy1
                scores['y2'] = bboxy2
                scores['label_txt'] = label_txts
                scores['file_name'] = file_names
                scores['patch_id'] = patch_ids
                scores['global_x1y1x2y2'] = global_x1y1x2y2s
                hist = pd.DataFrame.from_dict(scores)
                file = "%s/%s_%s__epoch%d_histogram.csv" % (saveto, self.opt.name, self.opt.phase, self.epoch)
                print("Saving histogram @: ", file)
                hist.to_csv(file)

            ##
            # PLOT PERFORMANCE
            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.data.valid.dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)

            ##
            # RETURN
            return performance
