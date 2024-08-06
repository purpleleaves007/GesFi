from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist

from model import*
from network import Adver_network, common_network
from base import Algorithm
from loss.common_loss import Entropylogits
import argparse

class GeneFi(Algorithm):

    def __init__(self, args):

        super(GeneFi, self).__init__(args)
        self.featurizer = FeatureNet()
        in_features = 512
        self.dbottleneck = common_network.feat_bottleneck(
            in_features, args.bottleneck, args.layer)
        self.ddiscriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.num_classes)

        self.bottleneck = common_network.feat_bottleneck(
            in_features, args.bottleneck, args.layer)
        self.classifier = common_network.feat_classifier(
            args.num_classes, args.bottleneck, args.classifier)

        self.abottleneck = common_network.feat_bottleneck(
            in_features, args.bottleneck, args.layer)

        self.aclassifier = common_network.feat_classifier(
            args.num_classes*args.latent_domain_num, args.bottleneck, args.classifier)
        self.dclassifier = common_network.feat_classifier(
            args.latent_domain_num, args.bottleneck, args.classifier)
        self.discriminator = Adver_network.Discriminator(
            args.bottleneck, args.dis_hidden, args.latent_domain_num)
        self.args = args

    def update_d(self, inputs, labels, pdlabels, opt):
        all_x1 = inputs.cuda().float()
        all_d1 = labels.cuda().long()
        all_c1 = pdlabels.cuda().long()
        z1 = self.dbottleneck(self.featurizer(all_x1))
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)
        disc_out1 = self.ddiscriminator(disc_in1)
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')
        cd1 = self.dclassifier(z1)
        ent_loss = Entropylogits(cd1)*self.args.lam + \
            F.cross_entropy(cd1, all_c1)
        loss = ent_loss+disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'dis': disc_loss.item(), 'ent': ent_loss.item()}

    def set_dlabel(self, loader):
        self.dbottleneck.eval()
        self.dclassifier.eval()
        self.featurizer.eval()

        start_test = True
        with torch.no_grad():
            iter_test = iter(loader)
            for _ in range(len(loader)):
                inputs, labels, pdlables, item = next(iter_test)
                inputs = inputs.cuda().float()
                index = item
                feas = self.dbottleneck(self.featurizer(inputs))
                outputs = self.dclassifier(feas)
                if start_test:
                    all_fea = feas.float().cpu()
                    all_output = outputs.float().cpu()
                    all_index = index
                    start_test = False
                else:
                    all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_index = np.hstack((all_index, index))
        all_output = nn.Softmax(dim=1)(all_output)

        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
        all_fea = all_fea.float().cpu().numpy()

        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)

        for _ in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_fea, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
        
        loader.dataset.set_labels_by_index(pred_label, all_index, 'pdlabel')
        print(Counter(pred_label))
        self.dbottleneck.train()
        self.dclassifier.train()
        self.featurizer.train()
        return Counter(pred_label)

    def update(self, inputs, labels, pdlables, opt, Cpd):
        all_x = inputs.cuda().float()
        all_y = labels.cuda().long()
        all_z = self.bottleneck(self.featurizer(all_x))

        disc_input = all_z
        disc_loss = 0
        cpsum = sum(Cpd.values())
        disc_input = Adver_network.ReverseLayerF.apply(
            disc_input, self.args.alpha)
        disc_out = self.discriminator(disc_input)
        disc_labels = pdlables.cuda().long()
        for i in range(len(disc_labels)):
            k=pdlables[i]
            disc_loss = disc_loss + (cpsum/(len(Cpd)*Cpd[k.item()]))*F.cross_entropy(disc_out[i], disc_labels[i])
        disc_loss = disc_loss/len(disc_labels)
        #disc_loss = F.cross_entropy(disc_out, disc_labels)
        all_preds = self.classifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        loss = classifier_loss+disc_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'total': loss.item(), 'class': classifier_loss.item(), 'dis': disc_loss.item()}

    def update_a(self, inputs, labels, pdlables, opt):
        all_x = inputs.cuda().float()
        all_c = labels.cuda().long()
        all_y = all_c
        all_z = self.abottleneck(self.featurizer(all_x))
        all_preds = self.aclassifier(all_z)
        classifier_loss = F.cross_entropy(all_preds, all_y)
        loss = classifier_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        return {'class': classifier_loss.item()}

    def predict(self, x):
        return self.classifier(self.bottleneck(self.featurizer(x)))

    def predict1(self, x):
        return self.ddiscriminator(self.dbottleneck(self.featurizer(x)))
