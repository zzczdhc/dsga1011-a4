import os
import argparse
import json
from time import monotonic
from tqdm import tqdm

import torch
from torch.nn.functional import cross_entropy
import wandb

from t5_utils import checkpoint_directory, initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from load_data import get_t5_tokenizer, load_t5_data
from utils import compute_metrics, save_queries_and_records, set_random_seeds

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-1)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="cosine", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=0,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=0,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=0,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--gradient_clip', type=float, default=0)
    parser.add_argument('--strict_sql_selection', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--max_train_minutes', type=float, default=0)

    args = parser.parse_args()
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_score = (-1, 0, 0)
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = checkpoint_directory(args)
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    deadline = monotonic() + args.max_train_minutes * 60 if args.max_train_minutes else float('inf')
    previous_epoch_seconds = 0.0
    for epoch in range(args.max_n_epochs):
        epoch_start = monotonic()
        if epoch and epoch_start + previous_epoch_seconds >= deadline:
            print('Ending training to leave time to evaluate the best checkpoint.')
            break
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        if args.use_wandb:
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        score = (record_f1, -error_rate, -eval_loss) if args.strict_sql_selection else (record_f1, 0, 0)
        if score > best_score:
            best_score = score
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        save_model(checkpoint_dir, model, best=False)
        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)
        previous_epoch_seconds = monotonic() - epoch_start

        if epochs_since_improvement >= args.patience_epochs:
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        loss = sequence_loss(args, model, encoder_input, encoder_mask, decoder_input, decoder_targets)
        loss.backward()
        if args.gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = decoder_targets.ne(PAD_IDX).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens


def sequence_loss(args, model, encoder_ids, encoder_mask, decoder_ids, targets):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
        logits = model(
            input_ids=encoder_ids,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_ids,
            use_cache=False,
        ).logits
        return cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=PAD_IDX)


def generate_queries(args, model, encoder_ids, encoder_mask, initial_decoder_ids):
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16):
        generated = model.generate(
            input_ids=encoder_ids,
            attention_mask=encoder_mask,
            decoder_input_ids=initial_decoder_ids,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
    queries = get_t5_tokenizer().batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [query.strip().replace('\n', ' ').replace('\r', ' ') for query in queries]
        
@torch.inference_mode()
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    queries = []
    for batch in tqdm(dev_loader):
        encoder_ids, encoder_mask, decoder_ids, targets, initial_ids = (tensor.to(DEVICE) for tensor in batch)
        loss = sequence_loss(args, model, encoder_ids, encoder_mask, decoder_ids, targets)
        tokens = targets.ne(PAD_IDX).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        queries.extend(generate_queries(args, model, encoder_ids, encoder_mask, initial_ids))

    save_queries_and_records(queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, errors = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path,
        penalize_sql_errors=args.strict_sql_selection,
    )
    return total_loss / total_tokens, record_f1, record_em, sql_em, sum(bool(error) for error in errors) / len(errors)
        
@torch.inference_mode()
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    queries = []
    for batch in tqdm(test_loader):
        encoder_ids, encoder_mask, initial_ids = (tensor.to(DEVICE) for tensor in batch)
        queries.extend(generate_queries(args, model, encoder_ids, encoder_mask, initial_ids))
    save_queries_and_records(queries, model_sql_path, model_record_path)

def main():
    # Get key arguments
    args = get_args()
    if args.max_n_epochs < 1 or args.patience_epochs < 1:
        raise ValueError('Set positive --max_n_epochs and --patience_epochs values')
    if args.bf16 and (DEVICE.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('--bf16 requires a CUDA GPU with bfloat16 support')
    set_random_seeds()
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model(args)
    optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))

    # Train 
    train(args, model, train_loader, dev_loader, optimizer, scheduler)

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    dev_selection_record_f1 = dev_record_f1
    if args.strict_sql_selection:
        _, _, dev_record_f1, _ = compute_metrics(gt_sql_path, model_sql_path, gt_record_path, model_record_path)
        print(f'Dev set selection Record F1 with SQL errors scored as zero: {dev_selection_record_f1:.6f}')
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    output_name = f't5_ft_{experiment_name}' if args.finetune else f't5_ft_{experiment_name}_ec'
    model_sql_path = os.path.join('results', f'{output_name}_test.sql')
    model_record_path = os.path.join('records', f'{output_name}_test.pkl')
    test_inference(args, model, test_loader, model_sql_path, model_record_path)

    metrics = {
        'training': vars(args),
        'dev_loss': float(dev_loss),
        'dev_record_f1': float(dev_record_f1),
        'dev_selection_record_f1': float(dev_selection_record_f1),
        'dev_record_em': float(dev_record_em),
        'dev_sql_em': float(dev_sql_em),
        'dev_error_rate': float(dev_error_rate),
        'dev_examples': len(dev_loader.dataset),
        'test_examples': len(test_loader.dataset),
    }
    with open(os.path.join('results', f'{output_name}_metrics.json'), 'w') as stream:
        json.dump(metrics, stream, indent=2)

if __name__ == "__main__":
    main()
