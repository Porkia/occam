from tkinter import Y
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import numpy as np
import torch
from scipy.linalg import fractional_matrix_power
import json
import torch.optim as optim
import soundfile
import wave

def SNR_singlech(clean, adv):
 
    length = min(len(clean), len(adv))
    est_noise = adv[:length] - clean[:length]#计算噪声语音
    
    #计算信噪比
    SNR = 10*np.log10((np.sum(clean**2))/(np.sum(est_noise**2)))
    print(SNR)

def wavread(wav_path):
    with wave.open(wav_path, 'rb') as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[0:4]
        strdata = f.readframes(nframes)
        data = np.fromstring(strdata, dtype=np.int16)
        data = data / 32768
    return data

def wavwrite(wav_path,data):
    nchannels=1
    sampwidth=2
    framerate=16000
    nframes=len(data)
    comptype='NONE'
    compname='not compressed'
    with wave.open(wav_path, 'wb') as fw:
        fw.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
        data=(data*32768).astype(np.int16)
        fw.writeframes(data.tostring())


def text2id(str,dict):
    ls = []
    for i in str:
        if i == ' ':
            ls.append(dict['|'])
        else:
            ls.append(dict[i])
    return ls




audio = wavread('ori.wav')
#adv_audio = wavread('test_cw_adv.wav')
adv_audio = wavread('init_adv.wav')
'''
target_text = 'THIS IS A TEST'

print(audio)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Wav2Vec2ForCTC.from_pretrained(r'yongjian/wav2vec2-large-a').to(device) # Note: PyTorch Model
processor = Wav2Vec2Processor.from_pretrained(r'yongjian/wav2vec2-large-a')
print(audio)

# Inference
sample_rate = processor.feature_extractor.sampling_rate
with torch.no_grad():
    model_inputs = processor(adv_audio, sampling_rate=sample_rate, return_tensors="pt", padding=True).to(device)
    print(model_inputs.input_values[0])
    logits = model(model_inputs.input_values, attention_mask=model_inputs.attention_mask).logits # use .cuda() for GPU acceleration
    pred_ids = torch.argmax(logits, dim=-1).cpu()
    pred_text = processor.batch_decode(pred_ids)


print('Transcription:', pred_text[0])
if (pred_text[0] == target_text):
    print(np.sqrt(np.sum((adv_audio - audio)**2)))'''
def distance(x, y):
    return np.sqrt(np.sum((x - y)**2))


N = 32000
D = np.ones((N, 1)).T  
m = np.zeros((N,1)).T
C = np.diag(D[0]) 
print(audio)

P = np.zeros((N))  
print(P)


#cov1_2 = fractional_matrix_power(cov, 0.5)
#print('cov1/2 : ', cov1_2)
#z_diag = cov1_2 * z_norm
#print(z_diag)


#z = np.random.multivariate_normal(m[0], cov)
#print(z)