#!/usr/bin/env python3
"""
Generate final report for Bayesian Optimization
Extracts all 10 iterations' hyperparameters and RMSE values
"""

import re
from datetime import datetime

log_file = "/home/developer/Desktop/nt_project/MatlabProject/matlab_training.log"
report_file = "/home/developer/Desktop/nt_project/MatlabProject/Results/BAYESIAN_OPT_FINAL_REPORT.txt"

def extract_results():
    with open(log_file, 'r') as f:
        content = f.read()

    # Split by iterations
    iterations = re.split(r'--- Random Search Iteration (\d+)/10 ---', content)

    results = []

    for i in range(1, len(iterations), 2):
        iter_num = iterations[i]
        iter_content = iterations[i+1]

        # Find hyperparameters
        lr = re.search(r'Learning rate: ([\d.e-]+)', iter_content)
        emb = re.search(r'Embedding dim: (\d+)', iter_content)
        heads = re.search(r'Attention heads: (\d+)', iter_content)
        enc = re.search(r'Encoder layers: (\d+)', iter_content)
        dec = re.search(r'Decoder layers: (\d+)', iter_content)
        drop = re.search(r'Dropout rate: ([\d.]+)', iter_content)
        batch = re.search(r'Batch size: (\d+)', iter_content)

        # Find final validation RMSE (last line before Training finished)
        training_lines = re.findall(r'\|\s+\d+\s+\|\s+\d+\s+\|\s+[\d:]+\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|', iter_content)

        final_val_rmse = float(training_lines[-1][1]) if training_lines else None
        final_train_rmse = float(training_lines[-1][0]) if training_lines else None

        # Find MSE
        mse_match = re.search(r'Validation MSE: ([\d.]+|Inf)', iter_content)
        if mse_match:
            mse_val = mse_match.group(1)
            mse = float(mse_val) if mse_val != 'Inf' else float('inf')
        else:
            mse = None

        # Find status
        if 'Training finished: Met validation criterion' in iter_content:
            status = "Early Stop"
        elif 'Training finished: Max epochs completed' in iter_content:
            status = "Max Epochs"
        else:
            status = "In Progress"

        results.append({
            'iter': int(iter_num),
            'lr': float(lr.group(1)) if lr else None,
            'emb': int(emb.group(1)) if emb else None,
            'heads': int(heads.group(1)) if heads else None,
            'enc': int(enc.group(1)) if enc else None,
            'dec': int(dec.group(1)) if dec else None,
            'drop': float(drop.group(1)) if drop else None,
            'batch': int(batch.group(1)) if batch else None,
            'train_rmse': final_train_rmse,
            'val_rmse': final_val_rmse,
            'mse': mse,
            'status': status
        })

    return results

def generate_report(results):
    # Sort by validation RMSE (lower is better)
    valid_results = [r for r in results if r['val_rmse'] is not None]
    sorted_by_rmse = sorted(valid_results, key=lambda x: x['val_rmse'])

    # Sort by MSE (what code will use - WRONG)
    valid_mse = [r for r in results if r['mse'] is not None and r['mse'] != float('inf')]
    sorted_by_mse = sorted(valid_mse, key=lambda x: x['mse'])

    best_by_rmse = sorted_by_rmse[0] if sorted_by_rmse else None
    best_by_mse = sorted_by_mse[0] if sorted_by_mse else None

    report = []
    report.append("╔════════════════════════════════════════════════════════════════════════════╗")
    report.append("║     BAYESIAN OPTIMIZATION FINAL REPORT - " + datetime.now().strftime("%Y-%m-%d %H:%M") + "           ║")
    report.append("╚════════════════════════════════════════════════════════════════════════════╝")
    report.append("")
    report.append("TÜM 10 İTERASYON TAMAMLANDI")
    report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    report.append("")

    # Table of all results
    report.append("DETAYLI SONUÇLAR (Tüm İterasyonlar)")
    report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    report.append("")
    report.append(f"{'Iter':<5} {'LR':<11} {'Emb':<5} {'Heads':<6} {'Enc':<4} {'Dec':<4} {'Drop':<6} {'Batch':<6} {'ValRMSE':<10} {'MSE':<10} {'Status':<12}")
    report.append("-" * 100)

    for r in results:
        lr_str = f"{r['lr']:.2e}" if r['lr'] else "N/A"
        emb_str = str(r['emb']) if r['emb'] else "N/A"
        heads_str = str(r['heads']) if r['heads'] else "N/A"
        enc_str = str(r['enc']) if r['enc'] else "N/A"
        dec_str = str(r['dec']) if r['dec'] else "N/A"
        drop_str = f"{r['drop']:.2f}" if r['drop'] else "N/A"
        batch_str = str(r['batch']) if r['batch'] else "N/A"
        val_rmse_str = f"{r['val_rmse']:.2f}" if r['val_rmse'] else "N/A"
        mse_str = f"{r['mse']:.2f}" if r['mse'] and r['mse'] != float('inf') else "Inf" if r['mse'] == float('inf') else "N/A"

        report.append(f"{r['iter']:<5} {lr_str:<11} {emb_str:<5} {heads_str:<6} {enc_str:<4} {dec_str:<4} {drop_str:<6} {batch_str:<6} {val_rmse_str:<10} {mse_str:<10} {r['status']:<12}")

    report.append("")
    report.append("")
    report.append("SIRALAMA (Validation RMSE - DOĞRU METRIK)")
    report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    report.append("")

    for rank, r in enumerate(sorted_by_rmse, 1):
        emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
        report.append(f"{emoji} {rank}. İterasyon {r['iter']}: Val RMSE = {r['val_rmse']:.2f}")
        report.append(f"      LR={r['lr']:.2e}, Emb={r['emb']}, Heads={r['heads']}, Enc={r['enc']}, Dec={r['dec']}, Drop={r['drop']:.2f}, Batch={r['batch']}")
        report.append("")

    report.append("")
    report.append("SIRALAMA (MSE - KOD BU METRİĞİ KULLANACAK - YANLIŞ!)")
    report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    report.append("")

    for rank, r in enumerate(sorted_by_mse, 1):
        emoji = "⭐" if rank == 1 else "  "
        report.append(f"{emoji} {rank}. İterasyon {r['iter']}: MSE = {r['mse']:.2f} (Val RMSE = {r['val_rmse']:.2f})")
        report.append(f"      LR={r['lr']:.2e}, Emb={r['emb']}, Heads={r['heads']}, Enc={r['enc']}, Dec={r['dec']}, Drop={r['drop']:.2f}, Batch={r['batch']}")
        report.append("")

    report.append("")
    report.append("⚠️  DİKKAT!")
    report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if best_by_rmse and best_by_mse and best_by_rmse['iter'] != best_by_mse['iter']:
        report.append(f"Kod otomatik olarak İterasyon {best_by_mse['iter']}'i seçecek (MSE={best_by_mse['mse']:.2f})")
        report.append(f"AMA gerçekte İterasyon {best_by_rmse['iter']} en iyi! (RMSE={best_by_rmse['val_rmse']:.2f})")
        report.append("")
        report.append("ÖNERİ: Asıl eğitimde İterasyon {}'nın parametrelerini MANUEL olarak kullan!".format(best_by_rmse['iter']))

    report.append("")
    report.append("")
    report.append("KULLANILACAK HİPERPARAMETRELER (RMSE bazında en iyi)")
    report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if best_by_rmse:
        report.append("")
        report.append(f"cfg.model.embedding_dim = {best_by_rmse['emb']};")
        report.append(f"cfg.model.num_heads = {best_by_rmse['heads']};")
        report.append(f"cfg.model.num_encoder_layers = {best_by_rmse['enc']};")
        report.append(f"cfg.model.num_decoder_layers = {best_by_rmse['dec']};")
        report.append(f"cfg.model.dropout_rate = {best_by_rmse['drop']:.2f};")
        report.append(f"cfg.train.initial_lr = {best_by_rmse['lr']:.6f};")
        report.append(f"cfg.train.min_batch_size = {best_by_rmse['batch']};")
        report.append("")
        report.append(f"Beklenen performans: Validation RMSE ≈ {best_by_rmse['val_rmse']:.2f}")

    report.append("")
    report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    report.append(f"Rapor oluşturulma: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(report)

if __name__ == "__main__":
    print("Extracting results from log file...")
    results = extract_results()

    print(f"Found {len(results)} iterations")

    print("Generating final report...")
    report_text = generate_report(results)

    with open(report_file, 'w') as f:
        f.write(report_text)

    print(f"✅ Final report saved to: {report_file}")
    print("")
    print("Summary:")
    print(report_text)
