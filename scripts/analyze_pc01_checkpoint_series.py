"""Read-only CPU audit of saved PC-01 epochs and their aggregate diagnostics.

Does not instantiate a model, load dataset rows/images, or run inference.
Writes a separate report; never edits or selects a checkpoint.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

from web_agent.train.selection import controlled_quality_gates


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().float().contiguous().numpy().tobytes()).hexdigest()


def audit(report_path, diagnostics_path):
    torch.set_num_threads(1)
    report = json.loads(report_path.read_text())
    diagnostics = json.loads(diagnostics_path.read_text())
    epochs = {x['epoch']: x for x in diagnostics['epochs']}
    selection = controlled_quality_gates(report, report['selection_rule'])
    assert selection['selected_epoch'] == report['selected_epoch']
    rows = []
    previous = None
    for epoch, name in sorted(report['epoch_checkpoints'].items(), key=lambda x: int(x[0])):
        epoch = int(epoch)
        path = Path(name)
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        assert checkpoint['epoch'] == epoch and checkpoint['epoch_complete']
        recorded = next(x for x in checkpoint['diagnostics_history'] if x['epoch'] == epoch)
        assert recorded == epochs[epoch], f'Checkpoint/report diagnostics mismatch: {epoch}'
        action = checkpoint['action']
        weight, bias = action['action_type.weight'], action['action_type.bias']
        assert weight.shape == (6, 256) and bias.shape == (6,)
        assert all(bool(torch.isfinite(t).all()) for t in action.values())
        diag = recorded['action_type']
        cm = diag['confusion_matrix']
        labels = diag['labels']
        assert labels == ['CLICK', 'TYPE', 'SELECT', 'SCROLL', 'NAVIGATE', 'PRESS_KEY']
        assert [sum(row) for row in cm] == [diag['true_distribution'][x] for x in labels]
        assert [sum(row[j] for row in cm) for j in range(6)] == [diag['predicted_distribution'][x] for x in labels]
        accuracy = sum(cm[i][i] for i in range(6)) / diag['rows']
        macro_f1 = sum(diag['per_class'][x]['f1'] for x in labels) / 6
        history = next(h for h in report['history'] if h['epoch'] == epoch)
        assert abs(accuracy - history['action_acc']) < 1e-10
        assert abs(macro_f1 - history['action_macro_f1']) < 1e-10
        row = {
            'epoch': epoch, 'path': str(path), 'bytes': path.stat().st_size,
            'sha256': sha(path), 'step': checkpoint['step'],
            'diagnostics_match_export': True, 'finite_six_class_head': True,
            'classifier_weight_sha256': tensor_sha(weight),
            'classifier_row_norms': weight.float().norm(dim=1).tolist(),
            'classifier_bias': bias.float().tolist(),
            'classifier_delta_from_previous_l2': None if previous is None else float((weight.float() - previous).norm()),
            'action': diag, 'action_accuracy': accuracy, 'action_macro_f1': macro_f1,
            'bbox': recorded['bbox'],
            'outcome_mcc': history['outcome_mcc'],
            'recovery_outcome_mcc': history['recovery_outcome_mcc'],
            'memory_mcc': history['memory_mcc'],
            'raw_action_loss': history['train_raw_action_loss'],
            'uncertainty_action_multiplier': history['train_uncertainty_multiplier_action'],
            'checkpoint_config_sha256': hashlib.sha256(json.dumps(checkpoint['config'], sort_keys=True).encode()).hexdigest(),
        }
        previous = weight.float().clone()
        rows.append(row)
        print(f"epoch {epoch}: verified; accuracy={accuracy:.4f}; macro-F1={macro_f1:.4f}", flush=True)
        del checkpoint, action, weight, bias
    parents = {Path(x['path']).parent for x in rows}
    expected = {x['path'] for x in rows}
    extras = []
    for path in sorted({p for parent in parents for p in parent.glob('*.ckpt')}):
        if str(path) in expected:
            continue
        checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        epoch = checkpoint['epoch']
        corresponding = next((r for r in rows if r['epoch'] == epoch), None)
        extras.append({'path': str(path), 'sha256': sha(path), 'epoch': epoch,
                       'epoch_complete': checkpoint['epoch_complete'],
                       'classifier_matches_epoch': corresponding is not None and tensor_sha(checkpoint['action']['action_type.weight']) == corresponding['classifier_weight_sha256']})
        del checkpoint
    assert rows[report['selected_epoch']]['sha256'] == report['selected_checkpoint_sha256']
    return {'scope': 'All ten epoch files and extra checkpoint files in their directory; stored validation diagnostics, not new inference.',
            'report_sha256': sha(report_path), 'diagnostics_sha256': sha(diagnostics_path),
            'train_action_counts': report['train_distribution']['action_type'],
            'action_class_weights': report['class_weights']['action'],
            'selection_replay': selection, 'epochs': rows, 'extra_checkpoints': extras,
            'distinct_classifier_weights': len({x['classifier_weight_sha256'] for x in rows}),
            'config_identity_counts': dict(Counter(x['checkpoint_config_sha256'] for x in rows)),
            'dataset_rows_read': 0, 'images_read': 0, 'new_model_inferences': 0,
            'checkpoint_changed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.results / 'report.json', args.results / 'diagnostics.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
