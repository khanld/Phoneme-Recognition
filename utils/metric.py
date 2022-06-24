from datasets import load_metric
import torch
from ctcdecode import CTCBeamDecoder


class Metric:
    def __init__(self):
        self.wer_metric = load_metric("wer")
        self.decoder = CTCBeamDecoder(
                        labels = [str(i) for i in range(75)],
                        model_path=None,
                        alpha=0,
                        beta=0,
                        cutoff_top_n=40,
                        cutoff_prob=1.0,
                        beam_width=1,
                        num_processes=4,
                        blank_id=74,
                        log_probs_input=True
                    )

    def __call__(self, logits, labels):
        beam_results, beam_scores, timesteps, out_lens = self.decoder.decode(logits.detach())
        # Get the top beam result
        top_beam_result = beam_results[0][0][:out_lens[0][0]].cpu().numpy()


        labels[labels == -100] = 74
        labels = labels.squeeze(0).detach().cpu().numpy()

        pred_strs = ' '.join(str(id) for id in top_beam_result)
        label_strs = ' '.join(str(id) for id in labels)

        wer = self.wer_metric.compute(predictions=[pred_strs], references=[label_strs])

        return wer