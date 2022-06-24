import json

def process_data(path):
    data = open(path)
    for line in data.readlines():
        item = json.loads(line)
        f.write(item['audio_filepath'] + '|' + item['text'] + '|' + str(item['duration']) + '\n')
    data.close()


f = open('/data1/speech/khanhld/ASR-Wa2vec-Finetune-Phoneme-En/dataset/train.txt', 'a+')

process_data('/data1/speech/khanhld/hi_fi_tts_v0/12787_manifest_other_train.json')
f.close()

