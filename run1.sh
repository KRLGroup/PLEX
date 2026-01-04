python train_baseline.py --only_eval --dataset Ba2Motifs --batch_size 128 --dropout 0.5 --epochs 3000 --hidden_dim 32 --l2 0.0001 --lr 0.001 --num_layers 3
python train_baseline_node.py --only_eval --dataset BaCommunity --batch_size 128 --dropout 0.5 --epochs 3000 --hidden_dim 16 --l2 0.0001 --layer_double --lr 0.01 --num_layers 5
python train_baseline.py --only_eval --dataset BaMultiShapes --batch_size 128 --dropout 0 --epochs 3000 --hidden_dim 32 --l2 0.0001 --layer_double --lr 0.001 --num_layers 3
python train_baseline_node.py --only_eval --dataset BaShapes --batch_size 32 --dropout 0.5 --epochs 3000 --hidden_dim 64 --l2 0.0001 --layer_double --lr 0.01 --num_layers 5
python train_baseline.py --only_eval --dataset BBBP --batch_size 32 --dropout 0.5 --epochs 3000 --hidden_dim 32 --l2 0.0001 --layer_double --lr 0.001 --num_layers 5 --edge_once
python train_baseline.py --only_eval --dataset MUTAG --batch_size 128 --dropout 0.5 --epochs 3000 --hidden_dim 32 --l2 0.0001 --layer_double --lr 0.01 --num_layers 5 --edge_once
python train_baseline.py --only_eval --dataset Mutagenicity --batch_size 32 --dropout 0 --epochs 3000 --hidden_dim 64 --l2 0.0001 --layer_double --lr 0.001 --num_layers 3
python train_baseline.py --only_eval --dataset NCI1 --batch_size 128 --dropout 0.5 --epochs 3000 --hidden_dim 64 --l2 0.0001 --layer_double --lr 0.001 --num_layers 3
python train_baseline.py --only_eval --dataset PROTEINS --batch_size 32 --dropout 0 --epochs 3000 --hidden_dim 32 --l2 0.0001 --layer_double --lr 0.01 --num_layers 3
python train_baseline.py --only_eval --dataset REDDIT-BINARY --batch_size 32 --dropout 0 --epochs 3000 --hidden_dim 64 --l2 0.0001 --layer_double --lr 0.001 --num_layers 5
python train_baseline_node.py --only_eval --dataset TreeGrid --batch_size 128 --dropout 0.5 --epochs 3000 --hidden_dim 32 --l2 0.0001 --layer_double --lr 0.001 --num_layers 5


batch_size=32|dropout=0|edge_once=False|epochs=3000|hidden_dim=32|l2=0.0001|layer_double=True|lr=0.01|nogumbel=False|num_layers=3