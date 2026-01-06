import torch
from torch_geometric.data import Dataset, Data
import os
import pandas as pd
from rdkit import Chem

class Aromatic(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(Aromatic, self).__init__(root, transform, pre_transform, pre_filter)
        self.data_df = pd.read_csv(root)
        self.data_df['mol'] = self.data_df['smiles'].apply(Chem.MolFromSmiles)

        # Global atom types
        atom_type_set = set()
        # Global bond types
        bond_type_set = set()

        for mol in self.data_df['mol']:
            if mol is None:
                continue
            for atom in mol.GetAtoms():
                atom_type_set.add(atom.GetAtomicNum())
            for bond in mol.GetBonds():
                bond_type_set.add(bond.GetBondType())

        self.atom_types = sorted(atom_type_set)
        self.bond_types = sorted(bond_type_set, key=lambda bt: int(bt))
        self.bond_type_to_idx = {bt: i for i, bt in enumerate(self.bond_types)}
        self.num_classes = self.data_df['aromatic_label'].nunique()
        self._num_node_features = len(self.atom_types)
        self._num_edge_features = len(self.bond_types)
    
    @property
    def data(self):
        """Create a pseudo-data object for compatibility"""
        class PseudoData:
            def __init__(self, labels):
                self.y = torch.tensor(labels, dtype=torch.long)
        
        return PseudoData(self.data_df['aromatic_label'].values)
    
    @property
    def num_features(self):
        return self._num_node_features
    
    @property
    def num_edge_features(self):
        return self._num_edge_features

    @property
    def raw_file_names(self):
        return ['aromatic_data.csv']

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(len(self.data_df))]

    def download(self):
        pass

    def len(self):
        return len(self.data_df)

    def get(self, idx):
        mol = self.data_df.iloc[idx]['mol']
        label = self.data_df.iloc[idx]['aromatic_label']

        # Node features: one-hot over global atom types
        x = []
        for atom in mol.GetAtoms():
            one_hot = [0] * len(self.atom_types)
            atomic_num = atom.GetAtomicNum()
            one_hot[self.atom_types.index(atomic_num)] = 1
            x.append(one_hot)
        x = torch.tensor(x, dtype=torch.float)

        # Edge index and one-hot bond features
        edge_index = []
        edge_attr = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            bt = bond.GetBondType()
            bt_idx = self.bond_type_to_idx.get(bt, 0)

            one_hot_bt = [0] * len(self.bond_types)
            one_hot_bt[bt_idx] = 1

            # Undirected edges
            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(one_hot_bt)
            edge_attr.append(one_hot_bt)

        edge_index = (
            torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            if edge_index else torch.zeros((2, 0), dtype=torch.long)
        )
        edge_attr = (
            torch.tensor(edge_attr, dtype=torch.float)
            if edge_attr else torch.zeros((0, len(self.bond_types)), dtype=torch.float)
        )

        y = torch.tensor([label], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

