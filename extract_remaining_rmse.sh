#!/bin/bash
# Quick script to extract RMSE values for iterations 7-10
# Run this FIRST THING tomorrow morning!

LOG="/home/developer/Desktop/nt_project/MatlabProject/matlab_training.log"
OUTPUT="/home/developer/Desktop/nt_project/MatlabProject/Results/ALL_ITERATIONS_RMSE.txt"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        EXTRACTING RMSE VALUES - ALL ITERATIONS                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

python3 << 'PYEOF'
import re

log_file = "/home/developer/Desktop/nt_project/MatlabProject/matlab_training.log"

with open(log_file, 'r') as f:
    content = f.read()

iterations = re.split(r'--- Random Search Iteration (\d+)/10 ---', content)

print("ITERATION | VAL RMSE | HIPERPARAMETRELER")
print("─────────────────────────────────────────────────────────────────────────────")

results = []

for i in range(1, len(iterations), 2):
    iter_num = int(iterations[i])
    iter_content = iterations[i+1]

    # Hyperparameters
    lr = re.search(r'Learning rate: ([\d.e-]+)', iter_content)
    emb = re.search(r'Embedding dim: (\d+)', iter_content)
    heads = re.search(r'Attention heads: (\d+)', iter_content)
    enc = re.search(r'Encoder layers: (\d+)', iter_content)
    dec = re.search(r'Decoder layers: (\d+)', iter_content)
    drop = re.search(r'Dropout rate: ([\d.]+)', iter_content)
    batch = re.search(r'Batch size: (\d+)', iter_content)

    # Final RMSE
    training_lines = re.findall(r'\|\s+\d+\s+\|\s+\d+\s+\|\s+[\d:]+\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|', iter_content)
    final_val_rmse = float(training_lines[-1][1]) if training_lines else None

    # Status
    if 'Training finished: Met validation criterion' in iter_content:
        status = "Early Stop"
    elif 'Training finished: Max epochs completed' in iter_content:
        status = "Max Epochs"
    else:
        status = "In Progress / Not Finished"

    if lr and final_val_rmse:
        results.append({
            'iter': iter_num,
            'rmse': final_val_rmse,
            'lr': float(lr.group(1)),
            'emb': int(emb.group(1)),
            'heads': int(heads.group(1)),
            'enc': int(enc.group(1)),
            'dec': int(dec.group(1)),
            'drop': float(drop.group(1)),
            'batch': int(batch.group(1)),
            'status': status
        })

        emoji = "🏆" if final_val_rmse == min([r['rmse'] for r in results]) else "  "

        print(f"{emoji} {iter_num:2d}      | {final_val_rmse:8.2f} | LR={lr.group(1)}, Emb={emb.group(1)}, Heads={heads.group(1)}, Enc={enc.group(1)}, Dec={dec.group(1)}, Drop={drop.group(1)}, Batch={batch.group(1)}")
        print(f"         |          | Status: {status}")
        print()

print("─────────────────────────────────────────────────────────────────────────────")
print()

if results:
    best = min(results, key=lambda x: x['rmse'])
    print("EN İYİ SONUÇ:")
    print(f"  Iteration: {best['iter']}")
    print(f"  Val RMSE: {best['rmse']:.2f}")
    print(f"  Learning Rate: {best['lr']:.6f}")
    print(f"  Embedding Dim: {best['emb']}")
    print(f"  Attention Heads: {best['heads']}")
    print(f"  Encoder Layers: {best['enc']}")
    print(f"  Decoder Layers: {best['dec']}")
    print(f"  Dropout: {best['drop']:.2f}")
    print(f"  Batch Size: {best['batch']}")
    print()
    print("BU HİPERPARAMETRELERİ ASIL EĞİTİMDE KULLAN!")
    print()
    print("config.m'e yaz:")
    print(f"  cfg.model.embedding_dim = {best['emb']};")
    print(f"  cfg.model.num_heads = {best['heads']};")
    print(f"  cfg.model.num_encoder_layers = {best['enc']};")
    print(f"  cfg.model.num_decoder_layers = {best['dec']};")
    print(f"  cfg.model.dropout_rate = {best['drop']:.2f};")
    print(f"  cfg.train.initial_lr = {best['lr']:.6f};")
    print(f"  cfg.train.min_batch_size = {best['batch']};")

# Save to file
with open('/home/developer/Desktop/nt_project/MatlabProject/Results/ALL_ITERATIONS_RMSE.txt', 'w') as f:
    f.write("ALL ITERATIONS RMSE VALUES\n")
    f.write("=" * 80 + "\n\n")
    for r in sorted(results, key=lambda x: x['rmse']):
        f.write(f"Iteration {r['iter']}: Val RMSE = {r['rmse']:.2f}\n")
        f.write(f"  LR={r['lr']:.6f}, Emb={r['emb']}, Heads={r['heads']}, Enc={r['enc']}, Dec={r['dec']}, Drop={r['drop']:.2f}, Batch={r['batch']}\n")
        f.write(f"  Status: {r['status']}\n\n")

    if results:
        best = min(results, key=lambda x: x['rmse'])
        f.write("\n" + "=" * 80 + "\n")
        f.write("BEST HYPERPARAMETERS (lowest RMSE):\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Iteration: {best['iter']}\n")
        f.write(f"Val RMSE: {best['rmse']:.2f}\n\n")
        f.write(f"cfg.model.embedding_dim = {best['emb']};\n")
        f.write(f"cfg.model.num_heads = {best['heads']};\n")
        f.write(f"cfg.model.num_encoder_layers = {best['enc']};\n")
        f.write(f"cfg.model.num_decoder_layers = {best['dec']};\n")
        f.write(f"cfg.model.dropout_rate = {best['drop']:.2f};\n")
        f.write(f"cfg.train.initial_lr = {best['lr']:.6f};\n")
        f.write(f"cfg.train.min_batch_size = {best['batch']};\n")

print()
print(f"✅ Results saved to: {OUTPUT}")

PYEOF
