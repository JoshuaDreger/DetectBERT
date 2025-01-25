import os
import os.path as osp
import numpy as np
import pickle
import debugpy
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from utils import DownloadApk, Disassemble, get_device
from SmaliPreprocess import Smalis2Txt

import tokenization
from models import DexBERT, Config
from dataloader import PreprocessEmbedding

# Importing additional required packages
import logging
from concurrent.futures import ThreadPoolExecutor

# Set up logging to capture errors or warnings
logging.basicConfig(level=logging.INFO)

# Dataset class that reads and tokenizes smali files for model inference
class SmaliSeqDataset(Dataset):
    def __init__(self, file, tokenize, max_len, pipeline=[]):
        super().__init__()
        self.file = open(file, "r", encoding='utf-8', errors='ignore')  # Open the smali text file
        self.tokenize = tokenize  # Tokenize function used for tokenizing the file content
        self.max_len = max_len  # Maximum length of tokens allowed per instance
        self.pipeline = pipeline  # Additional preprocessing steps to apply to the data
        self.current_class_id = 0  # To track class id per smali file
        
        # Generate instances from the opened file
        self.instance_list = self.instance_generator()

    # Method to read tokens from a file pointer with limited length
    def read_tokens(self, f, length, discard_last_and_restart=False, keep_method_name=True):
        tokens = []
        ClassEnd = False
        while len(tokens) < length:
            line = f.readline()
            if not line:  # End of file reached
                return None, ClassEnd
            if not line.strip():  # Blank line, indicating end of a document
                if discard_last_and_restart:
                    tokens = []  # Restart the tokenization if needed
                    continue
                else:
                    return tokens, ClassEnd  # Return collected tokens
            if line.strip().startswith('ClassName:'):
                continue  # Skip the smali class name
            if line.strip().startswith('MethodName:') and not keep_method_name:
                continue  # Skip the smali method name if specified
            if line.strip().startswith('ClassEnd'):
                ClassEnd = True  # Mark the end of the current class
                return tokens, ClassEnd
            tokens.extend(self.tokenize(line.strip()))  # Tokenize and extend the tokens list
        return tokens, ClassEnd

    # Generator that creates instances by reading from the smali file
    def instance_generator(self):
        instance_list = []
        close_file = False
        while True and not close_file:
            len_tokens = self.max_len
            tokens, ClassEnd = self.read_tokens(self.file, len_tokens, discard_last_and_restart=False, keep_method_name=True)
            
            if ClassEnd:  # End of current class -> assign a new class id
                self.current_class_id += 1
            
            if tokens is None:  # End of file
                self.file.close()
                close_file = True
                break
            if len(tokens) == 0:
                continue 

            class_id = self.current_class_id
            instance = (tokens, class_id)  # Create instance tuple
            for proc in self.pipeline:  # Apply the processing pipeline
                instance = proc(instance)
            
            instance_list.append(instance)  # Add instance to the list
        return instance_list
    
    def __len__(self):
        return len(self.instance_list)  # Return number of instances in the dataset
    
    def __getitem__(self, index):
        input_ids, segment_ids, input_mask, class_id = self.instance_list[index]
        return np.array(input_ids), np.array(segment_ids), np.array(input_mask), np.array(class_id)

# Function for inference using Bert model, processing data batch-wise
# Returns a list of class-level embeddings

def BertInfer(BertAEmodel, dataloader, device):
    class_vector_list = []
    last_class_id  = 0

    seq_iter_bar = tqdm(dataloader)  # Display progress bar for iteration
    with torch.no_grad():  # Disable gradient calculation for inference
        for _, batch in enumerate(seq_iter_bar):
            batch = [t.to(device) for t in batch]  # Move batch to the specified device
            input_ids, segment_ids, input_mask, class_id = batch

            r2 = BertAEmodel(input_ids, segment_ids, input_mask)  # Model inference
            batch_vec = r2.cpu().detach().numpy()  # Move result to CPU and convert to NumPy array
            
            # Process embeddings based on class ids
            for i, emb in enumerate(batch_vec):
                if len(class_vector_list) == 0:
                    class_vector_list.append(np.expand_dims(emb, axis=0))
                    continue
                if int(class_id.cpu()[i]) == last_class_id:
                    class_vector_list[-1] = np.concatenate([class_vector_list[-1], np.expand_dims(emb, axis=0)])
                    continue
                class_vector_list.append(np.expand_dims(emb, axis=0))
                last_class_id = int(class_id.cpu()[i])
    
    return class_vector_list

# Function to generate APK embeddings from hashes
# Handles the whole process of tokenizing, processing and storing embeddings

def Hash2ApkEmb(hash, emb_dir, smali_dir, BertAE, batch_size, pipeline, hash_dir):
    load_path = osp.join(smali_dir, hash.upper()+'.pkl')
    try:
        with open(load_path, 'rb') as f:
            loaded_dataset = pickle.load(f)  # Load dataset from serialized pickle file
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        logging.error(f"Failed to load dataset for hash {hash}: {e}")
        return

    # Reinitialize the tokenizer and pipeline
    vocab = './vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize_function = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))
    pipeline = [PreprocessEmbedding(tokenizer.convert_tokens_to_ids)]

    # Update the dataset with the tokenizer and pipeline
    loaded_dataset.tokenize = tokenize_function
    loaded_dataset.pipeline = pipeline

    dataloader = DataLoader(loaded_dataset, batch_size=batch_size, shuffle=False)  # Prepare data loader

    # Perform inference to get class-level vectors
    class_vec_list = BertInfer(BertAE, dataloader, device)
    class_vec_list = np.vstack(class_vec_list)  # Stack all class vectors vertically

    # Save the embeddings to disk
    try:
        with open(osp.join(emb_dir, hash.upper()+'.pkl'), 'wb') as f:
            pickle.dump(class_vec_list, f)
    except Exception as e:
        logging.error(f"Failed to save embeddings for hash {hash}: {e}")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run APK Embedding Generation')
    parser.add_argument('--gpu_ids', type=str, default='4', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--max_workers', type=int, default=1, help='Number of worker threads for processing')
    parser.add_argument('--src_data', type=str, default='goodware', help='Type of data to process (e.g., goodware, malware)')
    parser.add_argument('--dataset', type=str, default='dexray', help='Dataset that should be used')
    parser.add_argument('--reverse', action='store_true', default=False, help='Process the src_data_list in reverse order if set to True')

    args = parser.parse_args()

    # Setting up environment variables for CUDA
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", args.gpu_ids)
    
    root_dir = f'/shares/no-backup/j.dreger/{args.dataset}/'
    Bert_model_cfg = './bert_base.json'
    DexBERT_file = './pretrained_dexbert_model_steps_604364.pt'
    src_data_list = [[f'metadata/{args.dataset}_smooth_{args.src_data}_hashes.txt', args.src_data]]
    vocab = './vocab.txt'

    # Model initialization
    batch_size = 32
    device = get_device()  # Obtain available device (CPU or GPU)
    Bert_model_cfg = Config.from_json(Bert_model_cfg)  # Load BERT model configuration
    BertAE = DexBERT(Bert_model_cfg)  # Instantiate the model
    BertAE.load_state_dict(torch.load(DexBERT_file), strict=False)  # Load pre-trained weights
    BertAE.to(device)  # Move model to device
    BertAE.eval()  # Set model to evaluation mode

    # Tokenizer and preprocessing pipeline setup
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))  
    pipeline = [PreprocessEmbedding(tokenizer.convert_tokens_to_ids)]

    # Function to process each hash in the list (added ThreadPoolExecutor for faster processing)
    def process_hash(hash):
        hash = hash.strip()
        if os.path.exists(os.path.join(emb_dir, hash.upper() + '.pkl')):
            print(f"Embedding for hash {hash} already exists at {emb_dir}, skipping generation.")
            return
        try:
            Hash2ApkEmb(hash, emb_dir, smali_dir, BertAE, batch_size, pipeline, hash_dir)
        except Exception as e:
            logging.error(f"Error processing hash {hash}: {e}")

    # Process all hashes using threading to improve speed
    for pair in src_data_list:
        src_path, data_dir = pair[0], pair[1]
        hash_list = open(osp.join(root_dir, src_path), 'r').readlines()
        if args.reverse:
            hash_list = hash_list[::-1]
        hash_dir = osp.join(root_dir, data_dir)
        #smali_dir = osp.join("/work/j.dreger/data/detect_bert", 'smali_datasets', data_dir)
        smali_dir = osp.join(root_dir, 'smali', data_dir)
        tmp_dir = osp.join("tmp/josh/", 'emb', data_dir)
        #emb_dir = osp.join("/work/j.dreger/data/detect_bert", 'emb_dir', data_dir)
        emb_dir = osp.join(root_dir, 'emb', data_dir)

        # Create embedding directory if it doesn't exist
        if not os.path.exists(emb_dir):
            os.makedirs(emb_dir)
        
        # Use threading for processing
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            list(tqdm(executor.map(process_hash, hash_list), total=len(hash_list)))
    
    # Optional cleanup after all processing is done
    # os.system('rm -r {}'.format(tmp_dir))
