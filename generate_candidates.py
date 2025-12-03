import os
import torch
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from config import *
from model import *
from dataloader import *
from trainer import *
from pytorch_lightning import seed_everything


if __name__ == "__main__":
    # Setup args
    args.model_code = 'lru'
    set_template(args)
    
    seed_everything(args.seed)
    
    # Load data
    train_loader, val_loader, test_loader = dataloader_factory(args)
    
    # Load model
    model = LRURec(args)
    export_root = EXPERIMENT_ROOT + '/' + args.model_code + '/' + args.dataset_code
    
    # Load best model checkpoint
    model_path = os.path.join(export_root, 'models', 'best_acc_model.pth')
    print(f'Loading model from {model_path}')
    model.load_state_dict(torch.load(model_path, map_location=args.device))
    
    # Create trainer
    trainer = LRUTrainer(args, model, train_loader, val_loader, test_loader, export_root, False)
    
    # Generate candidates
    retrieved_path = os.path.join(export_root, 'retrieved.pkl')
    print(f'Generating candidates to {retrieved_path}')
    trainer.generate_candidates(retrieved_path)
    print('Done!')
