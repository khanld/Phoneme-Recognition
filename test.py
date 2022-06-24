import librosa
import torch

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
from g2p.text_to_sequence import Text2Seq
from jiwer import wer
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

def remove_repeated_and_pad(ids):
    pointer = 0
    res = []
    while pointer < len(ids)-1:
        if ids[pointer+1] != ids[pointer]:
            res += [ids[pointer]]
        pointer += 1

    res += [ids[pointer]]
    res = [id for id in res if id < 181]
    return res

# read audio
wav, _ = librosa.load('/data1/speech/khanhld/datasets/dh-data-01-test-outScope/wav/speaker_249/speaker_249-055936.wav')
print(len(wav))
text = 'đội được tô đậm là đội có thành tích tốt nhất'

# run inference
batch = feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(device)
output = model(batch)

print("wave length: ", len(wav))
print("output logits: ", output.logits)
print("output logits shape: ", output.logits.shape)

# argmax to get label
pred_ids = torch.argmax(output.logits, dim=-1).squeeze(0).detach().cpu().numpy()

# remove repeated and padding tokens
# eg: <pad>hh<pad>eee<pad>ll<pad>lloo<pad> -> hello
pred_phoneme_ids = remove_repeated_and_pad(pred_ids)

print("Predicted phonemes: ", [text2seq.id_to_phone[str(id)] for id in pred_phoneme_ids])
print("Predicted phoneme ids: ", pred_phoneme_ids)
print("True phoneme ids: ", text2seq.grapheme_to_sequence(text, padding=True))
print("Phone (character) error rate: ", wer(' '.join(str(id) for id in pred_phoneme_ids), \
                                            ' '.join(text2seq.grapheme_to_sequence(text, padding=True))))
print("Phone (character) error rate: ", wer(' '.join(str(id) for id in decoded[0].values.numpy()), \
                                            ' '.join(text2seq.grapheme_to_sequence(text, padding=True))))