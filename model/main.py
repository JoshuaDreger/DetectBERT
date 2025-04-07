import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from dataloader_original import ApkEmbDataset

from DetectBERT import DetectBERT
from lookahead import create_optimizer
from utils import read_yaml, get_device


ApkEmbDataset = {'apk': ApkEmbDataset, 'text': None, 'code': None}

class Trainer():
    def __init__(self, data_type, emb_dir, train_list, valid_list, test_list, classifier, device, log_dir, save_dir, cfg):
        
        assert data_type in {'apk', 'text', 'code'}
        
        self.data_type = data_type
        self.emb_dir = emb_dir
        self.train_list = train_list
        self.valid_list = valid_list
        self.test_list = test_list
        self.classifier = classifier
        self.device = device
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.cfg = cfg

    def save(self, i, model):
        """ save current model """
        print(f"Saving model at step {i} to {self.save_dir}")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        torch.save(model.state_dict(), # save model object before nn.DataParallel
            os.path.join(self.save_dir, 'model_steps_'+str(i)+'.pt'))

    def pad_collate_fn(self, batch):
        input_embs = [torch.tensor(x[0]) for x in batch]
        bag_labels = torch.tensor([x[1] for x in batch])
        
        # Pad input_embs to match the length of the longest sequence in the batch
        input_embs_padded = pad_sequence(input_embs, batch_first=True)
        
        return input_embs_padded, bag_labels
    
    def train(self):
        print("Starting training process...")
        print(f"hi Josh \n {self.emb_dir} \n\n {self.train_list}")
        Dataset = ApkEmbDataset[self.data_type]
        dataset = Dataset(self.emb_dir, self.train_list)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)#, collate_fn=self.pad_collate_fn) # for batchsize > 1

        criterion = nn.CrossEntropyLoss()
        optimizer = create_optimizer(self.cfg.Optimizer, self.classifier)

        global_step = 1 # global iteration steps regardless of epochs
        writer = SummaryWriter(log_dir=self.log_dir)

        metric_file = open(os.path.join(self.log_dir, 'validation_log.txt'), 'w')

        for e in range(self.cfg.Train.n_epochs):
            print(f"Epoch {e+1}/{self.cfg.Train.n_epochs}")

            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            iter_bar = tqdm(dataloader, desc='Iter (loss=X.XXX)')
            
            for i, batch in enumerate(iter_bar):
                print(f"Processing batch {i+1}")
                input_embs, bag_label = batch
                input_embs, bag_label = input_embs.to(self.device), bag_label.to(self.device)

                optimizer.zero_grad()

                results_dict = self.classifier(data=input_embs)
                logits = results_dict['logits']

                loss = criterion(logits, bag_label)
                print(f"Loss for batch {i+1}: {loss.item()}")
                
                loss_sum += loss.item()

                loss.backward()
                optimizer.step()

                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item())
                writer.add_scalars('data',
                                {'loss': loss.item(),
                                 'learning_rate': optimizer.param_groups[0]['lr'],
                                 'accuracy': (logits.argmax(dim=1) == bag_label).float().mean().item()
                                },
                                global_step)
                
                if global_step % self.cfg.Train.save_steps == 0: # save
                    print(f"Saving model at global step {global_step}")
                    self.save(global_step, self.classifier)
                    self.validation(global_step, metric_file)

                if self.cfg.Train.total_steps and self.cfg.Train.total_steps < global_step:
                    print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.Train.n_epochs, loss_sum/(i+1)))
                    print('The Total Steps have been reached.')
                    self.save(global_step, self.classifier) # save and finish when global_steps reach total_steps
                    return

                global_step += 1

            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.Train.n_epochs, loss_sum/(i+1)))
        self.save(global_step, self.classifier)
        metric_file.close()
        print("Training process completed.")
    
    def validation(self, global_step, metric_log):
        print(f"Starting validation at global step {global_step}")

        Dataset = ApkEmbDataset[self.data_type]
        dataset = Dataset(self.emb_dir, self.valid_list)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 

        pre_list = []
        gt_list  = []
        for _, batch in enumerate(tqdm(dataloader)):
            input_embs, bag_label = batch
            input_embs, bag_label = input_embs.to(self.device), bag_label.to(self.device)

            pre = self.classifier(data=input_embs)['Y_hat']
            pre_list.extend(pre.tolist())
            gt_list.extend(bag_label.tolist())

        precision, recall, fbeta_score, _ = precision_recall_fscore_support(gt_list, pre_list, beta=1.0, average='weighted')
        accuracy = accuracy_score(gt_list, pre_list)
        print(f"Validation results at global step {global_step}: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={fbeta_score:.4f}")
        metric_log.write('global_step: '+str(global_step)+'\n')
        metric_log.write('Acc	Pre	Rec	F1	Samp_num\n') 
        metric_log.write('{:.4f}	{:.4f}	{:.4f}	{:.4f}	{}\n'.format(accuracy, precision, recall, fbeta_score, len(gt_list)))

    def evaluation(self, weights=None):
        print("Starting evaluation...")

        if weights is not None:
            print(f"Loading weights from {weights}")
            self.classifier.load_state_dict(torch.load(weights), strict=True)
    
        Dataset = ApkEmbDataset[self.data_type]
        dataset = Dataset(self.emb_dir, self.test_list)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 

        pre_list = []
        gt_list  = []
        for _, batch in enumerate(tqdm(dataloader)):
            input_embs, bag_label = batch
            input_embs = input_embs.to(self.device)
            bag_label = bag_label.to(self.device)

            pre = self.classifier(data=input_embs)['Y_hat']
            pre_list.extend(pre.tolist())
            gt_list.extend(bag_label.tolist())

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        f = open(os.path.join(self.log_dir, 'evaluation.txt'), 'w')
        precision, recall, fbeta_score, _ = precision_recall_fscore_support(gt_list, pre_list, beta=1.0, average='weighted')
        accuracy = accuracy_score(gt_list, pre_list)
        print('Weighted Average:')
        print('Acc	Pre	Rec	F1	Samp_num')
        print('{:.4f}	{:.4f}	{:.4f}	{:.4f}	{}'.format(accuracy, precision, recall, fbeta_score, len(gt_list)))
        f.write('Weighted Average:\n')
        f.write('Acc	Pre	Rec	F1	Samp_num\n') 
        f.write('{:.4f}	{:.4f}	{:.4f}	{:.4f}	{}\n'.format(accuracy, precision, recall, fbeta_score, len(gt_list)))

        precision, recall, fbeta_score, support = precision_recall_fscore_support(gt_list, pre_list, beta=1.0)
        print('Metrics on Each Category:')
        print('  Category	Pre	Rec	F1	Samp_num')
        print('Benignware	{:.4f}	{:.4f}	{:.4f}	{}'.format(precision[0], recall[0], fbeta_score[0], support[0]))
        print('   Malware	{:.4f}	{:.4f}	{:.4f}	{}'.format(precision[1], recall[1], fbeta_score[1], support[1]))
        f.write('Metrics on Each Category:\n')
        f.write('  Category	Pre	Rec	F1	Samp_num\n')
        f.write('Benignware	{:.4f}	{:.4f}	{:.4f}	{}\n'.format(precision[0], recall[0], fbeta_score[0], support[0]))
        f.write('   Malware	{:.4f}	{:.4f}	{:.4f}	{}\n'.format(precision[1], recall[1], fbeta_score[1], support[1]))
        f.close()
        print("Evaluation completed.")

if __name__ == "__main__":

    print("Initializing training script...")
    os.nice(19)
    os.sched_setaffinity(0, set(range(64)))
    
    cfg        = './config.yaml'
    
    cfg = read_yaml(cfg)

    split_idx  = "random"
    dataset = 'dexray'
    print(f'running on split {split_idx}')

    data_type  = 'apk'  # choose one forom ('apk', 'text', 'code')
    emb_dir   = f'/shares/no-backup/j.dreger/{dataset}/emb/'
    train_list = f'../data/apk_splits/{cfg.Master.subset}_txt/{dataset}/train_{split_idx}.txt'
    valid_list = f'../data/apk_splits/{cfg.Master.subset}_txt/{dataset}/test_{split_idx}.txt'
    test_list  = f'../data/apk_splits/{cfg.Master.subset}_txt/{dataset}/test_{split_idx}.txt'
    log_dir    = f'./log/{cfg.Master.subset}/{dataset}/split_{split_idx}/{cfg.Model.aggregation}'
    save_dir   = f'./save/{cfg.Master.subset}/{dataset}/split_{split_idx}'
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= str(cfg.Train.device)
    
    device = get_device()
    classifier = DetectBERT(cfg= cfg, n_classes=cfg.Model.catg_num, input_size=cfg.Model.input_len, hidden_size=cfg.Model.hidden_len)
    classifier = classifier.to(device)

    trainer = Trainer(data_type, emb_dir, train_list, valid_list, test_list, classifier, device, log_dir, save_dir, cfg)
    print("Starting training...")
    trainer.train()
    print("Starting evaluation...")
    trainer.evaluation()
    # trainer.evaluation(weights='./save/malware_detectin/split_{}/model_steps_2000000.pt'.format(split_idx))
