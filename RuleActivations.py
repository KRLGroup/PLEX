"""
Script per analizzare le attivazioni dalle regole salvate per multipli dataset.

Genera 2 immagini totali:
1. Istogrammi delle attivazioni per tutti i dataset
2. Conteggio regole per tutti i dataset
"""

import pickle
import pandas as pd
import numpy as np
import torch
from pathlib import Path


def analyze_activations_from_all_rules(all_rules_path, dataset_name=None):
    """
    Analizza le attivazioni usando all_nodes_mask da all_rules.
    
    Args:
        all_rules_path: Path al file all_rules_{dataset}.pkl
        dataset_name: Nome del dataset (estratto dal filename se None)
    
    Returns:
        DataFrame con le statistiche di attivazione
    """
    
    # Carica all_rules
    print(f"Loading all_rules from: {all_rules_path}")
    with open(all_rules_path, 'rb') as f:
        all_rules = pickle.load(f)
    
    if dataset_name is None:
        dataset_name = Path(all_rules_path).stem.replace('all_rules_', '')
    
    print(f"Dataset: {dataset_name}")
    
    # Verifica struttura
    if 'conv' not in all_rules:
        raise ValueError("all_rules doesn't contain 'conv' key")
    
    conv_rules = all_rules['conv']
    print(f"Number of conv rule sets: {len(conv_rules)}")
    
    # Lista per raccogliere i risultati
    results = []
    
    # Per ogni set di regole (layer, feat, sublayer)
    for key, value in conv_rules.items():
        layer, feat, sublayer = key
        
        # Ottieni all_nodes_mask (o all_masks per edges)
        if 'all_nodes_mask' in value:
            mask = value['all_nodes_mask']
        elif 'all_masks' in value:
            mask = value['all_masks']
        else:
            print(f"Warning: No mask data for {key}, available keys: {list(value.keys())}")
            continue
        
        # Converti in tensor se necessario
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        
        print(f"\nLayer {layer}, Feature {feat}, Sublayer {sublayer}:")
        print(f"  Mask shape: {mask.shape}")
        print(f"  Mask type: {mask.dtype}")
        
        # Calcola la percentuale di True
        num_total = mask.shape[0]
        num_active = mask.sum().item()
        pct_activated = (num_active / num_total) * 100.0
        
        results.append({
            'dataset': dataset_name,
            'layer': layer,
            'feature': feat,
            'sublayer': sublayer,
            'pct_activated_nodes': pct_activated,
            'num_total': num_total,
            'num_active': num_active
        })
        
        print(f"  Activation: {pct_activated:.1f}% ({num_active}/{num_total})")
    
    # Crea DataFrame
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("\nWARNING: No data processed!")
        return df
    
    # Ordina per layer e sublayer
    df = df.sort_values(['layer', 'sublayer', 'feature'])
    
    print(f"\n{'='*80}")
    print(f"Processed {len(df)} rules for {dataset_name}")
    print(f"{'='*80}")
    
    return df


def count_rules_per_layer(all_rules_path, dataset_name, num_layers):
    """
    Conta il numero di regole per ogni layer e sublayer.
    
    Args:
        all_rules_path: Path al file all_rules.pkl
        dataset_name: Nome del dataset
        num_layers: Numero totale di layer
    
    Returns:
        DataFrame con colonne: dataset, layer, sublayer, num_rules
    """
    print(f"Loading all_rules from: {all_rules_path}")
    with open(all_rules_path, 'rb') as f:
        all_rules = pickle.load(f)
    
    conv_rules = all_rules.get('conv', {})
    
    # Dictionary: {layer: {'nn_0': count, 'nn_1': count}}
    layer_counts = {}
    
    # Inizializza tutti i layer con zero
    for layer in range(num_layers):
        layer_counts[layer] = {'nn_0': 0, 'nn_1': 0}
    
    # Conta le regole per ogni (layer, feat, sublayer)
    for key, value in conv_rules.items():
        layer, feat, sublayer = key
        rules = value.get('rules', [])
        
        # Conta solo regole non vuote
        num_rules = len([r for r in rules])
        if num_rules == 0:
            num_rules = 1
        
        # Aggiungi al conteggio
        layer_counts[layer][sublayer] += num_rules
    
    # Crea DataFrame
    records = []
    for layer in sorted(layer_counts.keys()):
        records.append({
            'dataset': dataset_name,
            'layer': layer,
            'sublayer': 'lambda_0',
            'num_rules': layer_counts[layer]['nn_0']
        })
        records.append({
            'dataset': dataset_name,
            'layer': layer,
            'sublayer': 'lambda_1',
            'num_rules': layer_counts[layer]['nn_1']
        })
    
    df = pd.DataFrame(records)
    
    print(f"\nRule counts for {dataset_name}:")
    print(df)
    
    return df


def create_combined_activation_histogram(all_dfs, save_path='combined_activation_histogram.png'):
    """
    Crea un istogramma combinato delle attivazioni per tutti i dataset.
    Ogni dataset ha la sua riga con 2 sublayer x num_layers colonne.
    
    Args:
        all_dfs: Lista di tuple (df, dataset_name, num_layers)
        save_path: Path dove salvare il plot
    """
    import matplotlib.pyplot as plt
    
    num_datasets = len(all_dfs)
    
    # Trova il max numero di layer tra tutti i dataset
    max_layers = max(num_layers for _, _, num_layers in all_dfs)
    
    # Crea figura: num_datasets*2 righe (2 sublayer per dataset), max_layers colonne
    fig, axes = plt.subplots(num_datasets * 2, max_layers, 
                            figsize=(4*max_layers, 3*num_datasets*2))
    
    # Assicurati che axes sia sempre 2D
    if num_datasets == 1 and max_layers == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    elif num_datasets == 1:
        axes = axes.reshape(2, max_layers)
    elif max_layers == 1:
        axes = axes.reshape(num_datasets * 2, 1)
    
    sublayer_names = ['nn_0', 'nn_1']
    
    for dataset_idx, (df, dataset_name, num_layers) in enumerate(all_dfs):
        for sublayer_idx, sublayer in enumerate(sublayer_names):
            row_idx = dataset_idx * 2 + sublayer_idx
            
            for col_idx in range(max_layers):
                ax = axes[row_idx, col_idx]
                ax.set_xlabel('% Activated Nodes', fontsize=10)
                
                if col_idx < num_layers:
                    # Filtra per layer e sublayer
                    layer_data = df[(df['layer'] == col_idx) & 
                                  (df['sublayer'] == sublayer)]['pct_activated_nodes'].values
                    
                    if len(layer_data) > 0:
                        ax.hist(layer_data, bins=20, range=(0, 100), 
                               edgecolor='black', linewidth=0.8, alpha=0.7)
                    
                    ax.set_xlim(0, 100)
                    ax.grid(True, alpha=0.3)
                    
                    # Titolo solo sulla prima riga
                    if row_idx == 0:
                        ax.set_title(f'Layer {col_idx + 1}', fontsize=10)
                    
                    # X label solo sull'ultima riga
                    if row_idx == num_datasets * 2 - 1:
                        ax.set_xlabel('% Activated', fontsize=9)
                    
                    # Y label solo sulla prima colonna
                    if col_idx == 0:
                        ax.set_ylabel('# Rules', fontsize=9)
                    
                else:
                    # Nascondi gli assi per layer non presenti
                    ax.axis('off')
                
                # Label sublayer a sinistra
                if col_idx == 0:
                    ax.text(-0.3, 0.5, f'λ{sublayer_idx}', 
                           transform=ax.transAxes, rotation=90, 
                           va='center', ha='center', fontsize=10, fontweight='bold')
                
                # Nome dataset CENTRATO sopra la prima riga di ogni dataset
                if sublayer_idx == 0 and col_idx == max_layers // 2:
                    # Posiziona il nome del dataset centrato sopra i subplot
                    ax.text(0.5, 1.25, dataset_name, 
                           transform=ax.transAxes,
                           va='bottom', ha='center', fontsize=12, fontweight='bold')
    
    # Rimuovi il titolo generale (suptitle)
    # plt.suptitle('Rule Activations - All Datasets', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined activation histogram: {save_path}")
    plt.close()


def plot_combined_rule_counts(all_counts_dfs, save_path='combined_rule_counts.png'):
    """
    Crea un grafico a barre del numero di regole per tutti i dataset.
    Layout: 5 dataset per riga.
    
    Args:
        all_counts_dfs: Lista di DataFrame con i conteggi
        save_path: Path dove salvare il plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Combina tutti i DataFrame
    combined_df = pd.concat(all_counts_dfs, ignore_index=True)
    
    # Ottieni lista dataset
    datasets = combined_df['dataset'].unique()
    num_datasets = len(datasets)
    
    # Layout: 5 colonne per riga
    ncols = 4
    nrows = int(np.ceil(num_datasets / ncols))
    
    # Crea subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    
    # Assicurati che axes sia sempre 2D
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)
    
    # Appiattisci per iterare facilmente
    axes_flat = axes.flatten()
    
    for idx, dataset_name in enumerate(datasets):
        ax = axes_flat[idx]
        
        # Filtra per dataset
        df_dataset = combined_df[combined_df['dataset'] == dataset_name]
        
        # Prepara i dati
        layers = sorted(df_dataset['layer'].unique())
        lambda_0_counts = df_dataset[df_dataset['sublayer'] == 'lambda_0'].sort_values('layer')['num_rules'].values
        lambda_1_counts = df_dataset[df_dataset['sublayer'] == 'lambda_1'].sort_values('layer')['num_rules'].values
        
        # Posizioni delle barre
        x = np.arange(len(layers))
        width = 0.35
        
        # Barre affiancate
        bars1 = ax.bar(x - width/2, lambda_0_counts, width, label='λ₀', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, lambda_1_counts, width, label='λ₁', alpha=0.8, edgecolor='black')
        
        # Labels e titolo
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('# Rules', fontsize=10)
        ax.set_title(f'{dataset_name}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{i+1}' for i in layers])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    # Nascondi gli assi inutilizzati
    for idx in range(num_datasets, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined rule count plot: {save_path}")
    plt.close()


# ============================================================================
# CONFIGURAZIONE DATASET
# ============================================================================

datasets_config = [
    {
        'name': 'Ba2Motifs',
        'path': '/home/palu001/LogiX-Me/explanations/Ba2Motifs/0/classe_1/all_rules.pkl',
        'num_layers': 3
    },
    {
        'name': 'BaCommunity',
        'path': '/home/palu001/LogiX-Me/explanations/BaCommunity/0/classe_0/all_rules.pkl',
        'num_layers': 5
    },
    {
        'name': 'BaMultiShapes',
        'path': '/home/palu001/LogiX-Me/explanations/BaMultiShapes/0/classe_0/all_rules.pkl',
        'num_layers': 3
    },
    {
        'name': 'BaShapes',
        'path': '/home/palu001/LogiX-Me/explanations/BaShapes/0/classe_0/all_rules.pkl',
        'num_layers': 5
    },
    # {
    #     'name': 'BBBP',
    #     'path': '/home/palu001/LogiX-Me/explanations/BBBP/0/classe_1/all_rules.pkl',
    #     'num_layers': 5
    # },
    # {
    #     'name': 'CiteSeer',
    #     'path': '/home/palu001/LogiX-Me/explanations/CiteSeer/0/classe_0/all_rules.pkl',
    #     'num_layers': 5
    # },
    # {
    #     'name': 'Cora',
    #     'path': '/home/palu001/LogiX-Me/explanations/Cora/0/classe_0/all_rules.pkl',
    #     'num_layers': 5
    # },
    # {
    #     'name': 'MUTAG',
    #     'path': '/home/palu001/LogiX-Me/explanations/MUTAG/0/classe_1/all_rules.pkl',
    #     'num_layers': 5
    # },
    # {
    #     'name': 'Mutagenicity',
    #     'path': '/home/palu001/LogiX-Me/explanations/Mutagenicity/0/classe_1/all_rules.pkl',
    #     'num_layers': 3
    # },
    # {
    #     'name': 'NCI1',
    #     'path': '/home/palu001/LogiX-Me/explanations/NCI1/0/classe_1/all_rules.pkl',
    #     'num_layers': 3
    # },
    # {
    #     'name': 'PROTEINS',
    #     'path': '/home/palu001/LogiX-Me/explanations/PROTEINS/0/classe_1/all_rules.pkl',
    #     'num_layers': 5
    # },
    # {
    #     'name': 'REDDIT-BINARY',
    #     'path': '/home/palu001/LogiX-Me/explanations/REDDIT-BINARY/0/classe_1/all_rules.pkl',
    #     'num_layers': 5
    # },
    # {
    #     'name': 'TreeGrid',
    #     'path': '/home/palu001/LogiX-Me/explanations/TreeGrid/0/classe_1/all_rules.pkl',
    #     'num_layers': 5
    # },
]

# ============================================================================
# PROCESSING
# ============================================================================

all_activation_dfs = []
all_count_dfs = []

for config in datasets_config:
    print(f"\n{'='*80}")
    print(f"Processing {config['name']}")
    print(f"{'='*80}")
    
    # Analizza attivazioni
    df = analyze_activations_from_all_rules(config['path'], config['name'])
    all_activation_dfs.append((df, config['name'], config['num_layers']))
    
    # Conta regole
    counts_df = count_rules_per_layer(config['path'], config['name'], config['num_layers'])
    all_count_dfs.append(counts_df)

# ============================================================================
# GENERA IMMAGINI COMBINATE
# ============================================================================

print(f"\n{'='*80}")
print("GENERATING COMBINED PLOTS")
print(f"{'='*80}")

# Istogramma attivazioni combinato
create_combined_activation_histogram(all_activation_dfs, 
                                     save_path='combined_activation_histogram.png')

# Grafico conteggio regole combinato
plot_combined_rule_counts(all_count_dfs, 
                          save_path='combined_rule_counts.png')

# Salva anche CSV combinati
combined_activations = pd.concat([df for df, _, _ in all_activation_dfs], ignore_index=True)
combined_activations.to_csv('combined_rule_activations.csv', index=False)
print("\nSaved: combined_rule_activations.csv")

combined_counts = pd.concat(all_count_dfs, ignore_index=True)
combined_counts.to_csv('combined_rule_counts.csv', index=False)
print("Saved: combined_rule_counts.csv")

print(f"\n{'='*80}")
print("DONE!")
print(f"{'='*80}")