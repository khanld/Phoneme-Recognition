import librosa
import torch

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
from g2p.text_to_sequence import Text2Seq
from jiwer import wer
from ctcdecode import CTCBeamDecoder
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

device = "cuda:6"

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('khanhld/wav2vec2-to-phonemes', use_auth_token='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')
model = Wav2Vec2ForCTC.from_pretrained('khanhld/wav2vec2-to-phonemes', use_auth_token='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').to(device)
model.eval()

config = {
    'g2p_model_path': 'resource/05_vn_g2p_model.fst',
    'g2p_config': 'resource/config_phonetisaurus.vn.v3.north.20pau.yml',
    'phone_id_list_file': 'resource/phone_id-v3.0.1.map.merge_all.20pau',
    'delimiter': None,
    'ignore_white_space': True
}
text2seq = Text2Seq(
    g2p_model_path=config['g2p_model_path'], 
    g2p_config=config['g2p_config'], 
    phone_id_list_file=config['phone_id_list_file'], delimiter=config['delimiter'], 
    ignore_white_space=config['ignore_white_space'])


# read audio
wav, _ = librosa.load('/data1/speech/khanhld/datasets/dh-data-01-test-outScope/wav/speaker_249/speaker_249-055936.wav')
print(len(wav))
text = 'đội được tô đậm là đội có thành tích tốt nhất'

# run inference
batch = feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
output = model(batch)


decoder = CTCBeamDecoder(
    labels = [str(i) for i in range(182)],
    model_path=None,
    alpha=0,
    beta=0,
    cutoff_top_n=40,
    cutoff_prob=1.0,
    beam_width=100,
    num_processes=4,
    blank_id=181,
    log_probs_input=True
)
beam_results, beam_scores, timesteps, out_lens = decoder.decode(output.logits.detach())

# Get the top beam result
top_beam_result = beam_results[0][0][:out_lens[0][0]].cpu().numpy()
# Get the corresponding top beam timestep
top_beam_timestep = timesteps[0][0][:out_lens[0][0]].cpu().numpy()

print("Timestep: ", list(top_beam_timestep))
print("Predicted phoneme ids: ", list(top_beam_timestep))
print("True phoneme ids: ", text2seq.grapheme_to_sequence(text, padding=True))
print("Phone (character) error rate: ", wer(' '.join(str(id) for id in top_beam_result), \
                                            ' '.join(text2seq.grapheme_to_sequence(text, padding=True))))