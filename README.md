# Reward Collapse in Aligning Large Language Models: A Prompt-Aware Approach to Preference Rankings

*Ziang Song, Tianle Cai, Jason D. Lee, Weijie J Su*

Codebase is adapted from [OpenAssistant](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-base).

## Reward Collapse
![reward collapse](fig/loss_comparison.png)
Illustration of reward collapse. One type of prompt is open-ended, which should result in a roughly uniform distribution of rewards, while the other is closed-ended, which should yield either high or low rewards (polarized). However, as evidenced in the first three plots, when a common utility function is employed, the two types of prompts result in a strikingly similar reward distribution. Conversely, when a prompt-aware utility is applied, as seen in the fourth plot, the two types of prompts exhibit distinct reward distributions.

## Codebase
- dataset_construction.ipynb: code for constructing the dataset from LongForm dataset.
- ranking_dataset_longform.pt: the constructed dataset.
- trainer.py: code for training the reward model.

## Training Commands:
```bash
python trainer.py configs/x.yml
python trainer.py configs/x_inv.yml
python trainer.py configs/log.yml
python trainer.py configs/x_adaptive.yml
python trainer.py configs/logsigmoid.yml
```

## Citation
```
@article{song2021reward,
  title={Reward Collapse in Aligning Large Language Models: A Prompt-Aware Approach to Preference Rankings},
  author={Song, Ziang and Cai, Tianle and Lee, Jason D and Su, Weijie J},
  journal={arXiv preprint},
  year={2023}
}
```