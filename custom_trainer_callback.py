"""
Custom trainer callback to store config and evaluation results
"""
import os
import json
from copy import deepcopy
from transformers import TrainerCallback


class CustomTrainerCallback(TrainerCallback):
    """Keep track of configs and evaluation results"""

    def __init__(self, args):
        self.config = args
        self.checkpoint_metrics = {}
        self.last_metrics = {'eval_accuracy': 0}


    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Get evaluation accuracy after each evaluation step"""
        metric_name = 'eval_accuracy'
        if metric_name in metrics:
            self.last_metrics[metric_name] = metrics[metric_name]


    def on_save(self, args, state, control, **kwargs):
        """Store evaluation results after each save step"""
        self.checkpoint_metrics[state.global_step] = deepcopy(self.last_metrics)


    def on_train_end(self, _train_args, state, control, **kwargs):
        """Store results and configs after training is done"""
        chk_dirs = [d for d in os.listdir(self.config['name']) if d.startswith('checkpoint-')]
        checkpoint_steps = [int(c.replace('checkpoint-', '').strip()) for c in chk_dirs]
        self.config['checkpoints'] = {c: self.checkpoint_metrics[c] for c in checkpoint_steps}
        config_path = os.path.join(self.config['name'], "config.json")
        with open(config_path, "w", encoding='utf-8') as outfile:
            json.dump(self.config, outfile, indent=4)
        best_accuracy = max([v['eval_accuracy'] for v in self.checkpoint_metrics.values()])
        print(f'==== BEST EVAL ACCURACY : {best_accuracy} ====')
