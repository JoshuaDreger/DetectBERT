import os
import os.path as osp
import numpy as np
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from torch.utils.data import Dataset, DataLoader
from utils import DownloadApk, Disassemble, get_device
from SmaliPreprocess import Smalis2Txt
import tokenization
from models import DexBERT, Config
from dataloader import PreprocessEmbedding
from tqdm import tqdm  # Import tqdm for progress bars

class SmaliSeqDataset(Dataset):
    def __init__(self, file, tokenize, max_len, pipeline=[]):
        super().__init__()
        print(f"Initializing SmaliSeqDataset with file: {file}")
        self.file_path = file
        self.tokenize = tokenize # tokenize function
        self.max_len = max_len # maximum length of tokens
        self.pipeline = pipeline
        self.current_class_id = 0
        self.instance_list = self.instance_generator()
        print(f"Finished initializing SmaliSeqDataset with {len(self.instance_list)} instances")

    def __getstate__(self):
        # Custom method to ensure the dataset can be pickled
        state = self.__dict__.copy()
        # Remove file handler if it exists
        if 'file' in state:
            del state['file']
        return state

    def __setstate__(self, state):
        # Custom method to restore the state
        self.__dict__.update(state)
        # Reinitialize the file handler
        self.file = open(self.file_path, "r", encoding='utf-8', errors='ignore')

    def read_tokens(self, f, length, discard_last_and_restart=False, keep_method_name=True):
        """ Read tokens from file pointer with limited length """
        tokens   = []
        ClassEnd = False
        while len(tokens) < length:
            line = f.readline()
            if not line: # end of file
                return None, ClassEnd
            if not line.strip(): # blank line (delimiter of documents)
                if discard_last_and_restart:
                    tokens = [] # throw all and restart
                    continue
                else:
                    return tokens, ClassEnd # return last tokens in the document
            if line.strip().startswith('ClassName:'):
                continue  # skip the smali class name
            if line.strip().startswith('MethodName:') and not keep_method_name:
                continue # skip the smali method name
            if line.strip().startswith('ClassEnd'):
                ClassEnd = True
                return tokens, ClassEnd
            tokens.extend(self.tokenize(line.strip()))
        return tokens, ClassEnd

    def instance_generator(self): # iterator to load data
        print("Starting instance generation")
        instance_list = []
        close_file = False
        with open(self.file_path, "r", encoding='utf-8', errors='ignore') as f:
            while True and not close_file:
                len_tokens = self.max_len

                tokens, ClassEnd = self.read_tokens(f, len_tokens, discard_last_and_restart=False, keep_method_name=True)
                
                if ClassEnd:  # end of current class -> end of current batch
                    self.current_class_id += 1
                    #print(f"Current class ID updated to {self.current_class_id}")
                
                if tokens is None:  # end of file
                    close_file = True
                    #print("File closed after reaching the end")
                    break
                if len(tokens) == 0:
                    continue 

                class_id = self.current_class_id
                instance = (tokens, class_id)
                for proc in self.pipeline:
                    instance = proc(instance)
                
                instance_list.append(instance)
        #print(f"Instance generation complete with {len(instance_list)} instances")
        return instance_list

    def save_dataset(self, save_path):
        smali_dir = os.path.dirname(save_path)
        if not os.path.exists(smali_dir):
            os.makedirs(smali_dir)
        print(f"Saving dataset to {save_path}")
        self.pipeline = None
        self.tokenize = None
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Dataset saved to {save_path}")

    def __len__(self):
        return len(self.instance_list)
    
    def __getitem__(self, index):
        input_ids, segment_ids, input_mask, class_id = self.instance_list[index]
        return np.array(input_ids), np.array(segment_ids), np.array(input_mask), np.array(class_id)


def set_cpu_limit():
    # Set CPU affinity for the current process to a limited number of cores
    os.sched_setaffinity(0, set(range(32)))  # Limit to CPU cores 0-15 (can be modified)
    # Lower the priority of the current process
    os.nice(19)

def Hash2ApkEmb(hash, tmp_dir, smali_dir, pipeline, hash_dir):
    set_cpu_limit()  # Limit CPU usage before starting task
    save_path = osp.join(smali_dir, hash.upper() + '.pkl')
    if osp.exists(save_path):
        print(f"Dataset for hash {hash} already exists at {save_path}, skipping generation.")
        return
    if not osp.exists(osp.join(hash_dir, hash.upper() + '.apk')):
        print(f"APK for hash {hash} not downloaded at {hash_dir}, skipping generation.")
        return

    print(f"Processing hash: {hash}")
    apk_path = osp.join(hash_dir, hash.upper()+'.apk')
    smali_dir = osp.join(tmp_dir, hash)
    print(f"Disassembling APK at {apk_path}")
    Disassemble(apk_path, smali_dir)
    print(f"Disassembly complete, converting smali to txt at {smali_dir}")
    Smalis2Txt(tmp_dir, smali_dir, only_keep_func_name=False)
    ApkName = smali_dir.split('/')[-1] if smali_dir.split('/')[-1] else smali_dir.split('/')[-2]
    txt_file = osp.join(tmp_dir, ApkName+'.txt')
    print(f"Creating dataset from {txt_file}")
    dataset = SmaliSeqDataset(txt_file, tokenize, 512, pipeline)
    dataset.save_dataset(save_path)
    print(f"Finished processing hash: {hash}")
    os.system('rm -r {}'.format(osp.join(tmp_dir, ApkName+'.txt')))
    os.system('rm -r {}'.format(osp.join(tmp_dir, ApkName)))


def process_hashes_in_parallel(hash_list, tmp_dir, smali_dir, pipeline, hash_dir, max_concurrent_tasks=16):
    print(f"Starting parallel processing of {len(hash_list)} hashes with a maximum of {max_concurrent_tasks} concurrent tasks")

    with ProcessPoolExecutor(max_workers=max_concurrent_tasks) as executor:
        futures = []

        for hash in hash_list:
            future = executor.submit(Hash2ApkEmb, hash.strip(), tmp_dir, smali_dir, pipeline, hash_dir)
            futures.append(future)
        
        with tqdm(total=len(futures), desc="Processing hashes") as pbar:
            for completed_future in as_completed(futures):
                try:
                    completed_future.result()  # Wait for the completed task
                    pbar.update(1)  # Update progress bar
                except Exception as e:
                    print(f"Error processing hash: {e}")

    print("Finished parallel processing of hashes")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with only one iteration')
    parser.add_argument('--max_workers', type=int, default=1, help='Number of worker threads for processing')
    parser.add_argument('--apk_type', type=str, default='goodware', help='Type of data to process (e.g., goodware, malware)')
    parser.add_argument('--dataset', type=str, default='dexray', help='Dataset that should be used')
    args = parser.parse_args()

    root_dir = f'/work/j.dreger/data/{args.dataset}/'
    dataset_root = f'/mnt/{args.dataset}'
    apk_list = [[f'{args.dataset}_{args.apk_type}_hashes.txt', args.apk_type]]
    vocab = './vocab.txt'

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))  

    pipeline = [PreprocessEmbedding(tokenizer.convert_tokens_to_ids)]

    for pair in apk_list:
        src_path, data_dir = pair[0], pair[1]
        hash_list = open(osp.join(root_dir, src_path), 'r').readlines()
        if args.debug:
            hash_list = hash_list[:1]  # Only process one hash in debug mode
        hash_dir = dataset_root
        smali_dir = osp.join(root_dir, 'smali_datasets', args.dataset, data_dir)
        tmp_dir = osp.join(root_dir, 'tmp', data_dir)
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)
        if not os.path.exists(smali_dir):
            os.makedirs(smali_dir)

        print(f"Starting processing for data directory: {data_dir}")
        process_hashes_in_parallel(hash_list, tmp_dir, smali_dir, pipeline, hash_dir, args.max_workers)
        print(f"Finished processing for data directory: {data_dir}")
