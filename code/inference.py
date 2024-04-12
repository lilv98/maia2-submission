from utils import *
from main import *


class TestDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, all_moves_dict, elo_dict):
        
        self.all_moves_dict = all_moves_dict
        self.data = data.values.tolist()
        self.elo_dict = elo_dict
    
    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        fen, move, elo_self, elo_oppo = self.data[idx]

        if fen.split(' ')[1] == 'w':
            board = chess.Board(fen)
        elif fen.split(' ')[1] == 'b':
            board = chess.Board(fen).mirror()
            move = mirror_move(move)
        else:
            raise ValueError(f"Invalid fen: {fen}")
            
        board_input = board_to_tensor(board)
        
        elo_self = map_to_category(elo_self, self.elo_dict)
        elo_oppo = map_to_category(elo_oppo, self.elo_dict)
        
        legal_moves, _ = get_side_info(board, move, self.all_moves_dict)
        
        return fen, board_input, elo_self, elo_oppo, legal_moves


def get_preds(model, dataloader, all_moves_dict_reversed, cfg_inference):
    
    all_probs = []
    predicted_move_probs = []
    predicted_moves = []
    predicted_win_probs = []
    
    model.eval()
    with torch.no_grad():
        
        for fens, boards, elos_self, elos_oppo, legal_moves in dataloader:
            
            if cfg_inference.gpu:
                boards = boards.cuda()
                elos_self = elos_self.cuda()
                elos_oppo = elos_oppo.cuda()
                legal_moves = legal_moves.cuda()

            logits_maia, _, logits_value = model(boards, elos_self, elos_oppo)
            logits_maia_legal = logits_maia * legal_moves
            probs = logits_maia_legal.softmax(dim=-1)

            all_probs.append(probs.cpu())
            predicted_move_probs.append(probs.max(dim=-1).values.cpu())
            predicted_move_indices = probs.argmax(dim=-1)
            for i in range(len(fens)):
                fen = fens[i]
                predicted_move = all_moves_dict_reversed[predicted_move_indices[i].item()]
                if fen.split(' ')[1] == 'b':
                    predicted_move = mirror_move(predicted_move)
                predicted_moves.append(predicted_move)

            predicted_win_probs.append((logits_value / 2 + 0.5).cpu())
    
    all_probs = torch.cat(all_probs).cpu().numpy()
    predicted_move_probs = torch.cat(predicted_move_probs).numpy()
    predicted_win_probs = torch.cat(predicted_win_probs).numpy()
    
    return all_probs, predicted_move_probs, predicted_moves, predicted_win_probs


def inference_batch(data):
    
    cfg = parse_args()
    cfg_inference = parse_inference_args()
    if cfg_inference.verbose:
        show_cfg(cfg_inference)

    all_moves = get_all_possible_moves()
    all_moves_dict = {move: i for i, move in enumerate(all_moves)}
    elo_dict = create_elo_dict()

    model = MAIA2Model(len(all_moves), elo_dict, cfg)
    model = nn.DataParallel(model)
    
    checkpoint = torch.load(cfg_inference.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.module
    
    if cfg_inference.gpu:
        model = model.cuda()
    
    all_moves_dict_reversed = {v: k for k, v in all_moves_dict.items()}
    dataset = TestDataset(data, all_moves_dict, elo_dict)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=cfg_inference.batch_size, 
                                            shuffle=False, 
                                            drop_last=False,
                                            num_workers=cfg_inference.num_workers)
    if cfg_inference.verbose:
        dataloader = tqdm.tqdm(dataloader)
    all_probs, predicted_move_probs, predicted_moves, predicted_win_probs = get_preds(model, dataloader, all_moves_dict_reversed, cfg_inference)
    
    data['predicted_move'] = predicted_moves
    data['predicted_move_prob'] = predicted_move_probs
    data['predicted_win_prob'] = predicted_win_probs
    data['all_probs'] = all_probs.tolist()
    
    return data

def parse_inference_args(args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--batch_size', default=2048, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--model_path', default='your_model_path', type=str)
    parser.add_argument('--gpu', default=True, type=bool)

    return parser.parse_args(args)

def show_cfg(cfg):
    print('Configurations:', flush=True)
    for arg in vars(cfg):
        print(f'\t{arg}: {getattr(cfg, arg)}', flush=True)


if __name__ == '__main__':
    
    data = pd.read_csv('../data/all_reduced_rapid.csv')
    data = data[data.move_ply > 10]
    
    data_novice = data[data['rounded_elo'] <= 1500][['board', 'move', 'active_elo', 'opponent_elo']]
    data_intermediate = data[(data['rounded_elo'] > 1500) & (data['rounded_elo'] < 2000)][['board', 'move', 'active_elo', 'opponent_elo']]
    data_advanced = data[data['rounded_elo'] >= 2000][['board', 'move', 'active_elo', 'opponent_elo']]
    print(f'lens: {len(data_novice)}, {len(data_intermediate)}, {len(data_advanced)}', flush=True)
    
    results = []
    for split in [data_novice, data_intermediate, data_advanced]:
        inference_batch(split)
        print(round(len(split[split['predicted_move'] == split['move']]) / len(split), 4))
    
    
    
    
