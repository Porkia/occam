import numpy as np
from random import shuffle
import math
import random
from scipy.linalg import fractional_matrix_power
#import torch
import wave
import librosa
import soundfile
import aliyun_function
import time

#from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC  # Target models

# Audio files
original_path = './ali/01_original_crafted_fragment_classical_castale.wav'
initial_adv_path = './ali/openmydoor.wav'


# Hyper parameters
e = math.e
b = 15
T = 30000
l = 30
c_c = 0.01
c_cov = 0.001
rescale = 1/32768
#rescale = 1

# Fitness function, should be customized according to the target model.
target_text = 'Open my door.'  # Attack target
def ali_api_loss(x, x_adv):    
    wavwrite('adv_tmp.wav', x_adv)
    try_times = 0
    while(try_times < 10):
        try:
            re, success, na = aliyun_function.aliyun_recong('adv_tmp.wav')
            break
        except:
            try_times += 1
            time.sleep(15)
    re = re.strip()
    if (re == target_text):   # If the transcription is correct, return the l2 distance of the original audio
                                        # and the adversarial audio, else return the +inf(99999)
        return np.sqrt(np.sum((x_adv - x)**2))
    else :
        return 99999

def wavread(wav_path):  # Read the audio file as a numpy array, after normalization in [-1,1]
    with wave.open(wav_path, 'rb') as f:
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[0:4]
        strdata = f.readframes(nframes)
        data = np.fromstring(strdata, dtype=np.int16)
        data = data / 32768
    return data

def wavwrite(wav_path,data):    # Write the audio into a file.
    nchannels=1
    sampwidth=2
    framerate=16000
    nframes=len(data)
    comptype='NONE'
    compname='not compressed'
    with wave.open(wav_path, 'wb') as fw:
        fw.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
        data=(data*32768).astype(np.int16)
        fw.writeframes(data.tobytes())

def change_sample_rate(path,new_path,new_sample_rate = 8000):

    signal, sr = librosa.load(path, sr=None)    
    new_signal = librosa.resample(signal, sr, new_sample_rate)      
    print(new_path)
    soundfile.write(new_path, new_signal , new_sample_rate)

def padding(adv_audio, audio):
    padding_length = len(audio) - len(adv_audio)
    pad_left = int(padding_length/2)
    pad_right = padding_length - pad_left

    left_padded = np.concatenate([np.zeros(pad_left),adv_audio])
    right_padded = np.concatenate([left_padded,np.zeros(pad_right)])
    return right_padded


class Occam():
    def __init__(self, b = 15, T = 30000, l = 30,  c_c = 0.01, c_cov = 0.001 , rescale = 1):
        self.b = b              # binary search times
        self.T = T              # total time epoches
        self.t = 0
        self.l = l             # lambda, sample times
        self.miu = 0.08         # distance from orignal input
        self.c_c = c_c          
        self.c_cov = c_cov
        self.m = 16              # group numbers
        self.s = 1              # group dimensions
        self.N = 0              # dimensions of input audio vector
        self.C = None           # the covariance matirx
        self.P = None           # the evolution path
        self.R = [0,0,0,0]      # R = {r1, r2, r3, r4}
        self.delta = [0,0,0]    # delta = {delta_m/2, delta_m, delta_2m}
        self.m_base = self.m 
        self.rescale = rescale
        self.adv_count = 0

    def bin_search(self, x, x_adv, loss):     # Binary search, roughly approaching the decision boundary.
        count_b = 1
        result = x_adv.copy()
        x_like = x.copy()
        old_fitness = loss(x, x_adv)
        new_fitness = 0
        while (count_b < self.b):
            mid = x_like + (result - x_like)/2
            new_fitness = loss(x, mid)

            if (new_fitness < old_fitness):
                result = mid.copy()
                old_fitness = new_fitness
            else:
                x_like = mid.copy()
            count_b += 1
        print('bin search, get loss: ', old_fitness)
        return result, old_fitness

    def distance(self, x, y):       # Calculate the l2 distance.
        return np.sqrt(np.sum((x - y)**2))
    
    def decompose(self, strategy_id):    # Group the vector into sub-groups according to the grouping
                                         # strategy.(SG, RG, MiVG, MaVG)
        s = math.ceil(self.N/self.m)
        result_id = []
        if (strategy_id == 0):      #SG Static grouping
            lists=[i for i in range(self.N)]
        elif (strategy_id == 1):    #RG: Random grouping
            lists=[i for i in range(self.N)]
            shuffle(lists)            
        elif (strategy_id == 2):    #MiVG Minimize variance grouping
            
            lists = np.argsort(self.C).tolist()            
        elif (strategy_id == 3):    #MaVG Maximize variance grouping
            
            lists = np.argsort(self.C).tolist()
            lists.reverse()
        result_id = [lists[i:i + s] for i in range(0,len(lists),s)]
        return(result_id)
    
    def group_strategy(self):       # Get a grouping strategy according to the possibilities.
        total_score = 0
        for score in self.R:
            total_score += e**score
        prob = [e**score/total_score for score in self.R]
        stra_list = [0,1,2,3]
        cumulative_probability = 0
        rand_prob = random.uniform(0,1)
        final_stra = 0
        for stra, stra_prob in zip(stra_list, prob):
            cumulative_probability += stra_prob
            if rand_prob < cumulative_probability:
                final_stra = stra
                break
        return final_stra

    def pilot_test(self,fix_x, fix_x_adv, loss, strategy, c_b_f):    # Run a pilot test to update
                                                                     # the m size adaptively.
        groups = self.decompose(strategy)
        x = fix_x.copy()
        x_adv = fix_x_adv.copy()
        group = groups[0]

        x_sub = np.array([x[i] for i in group])
        x_adv_sub = np.array([x_adv[i] for i in group])
        C_sub = np.array([self.C[i] for i in group])
        P_sub = np.array([self.P[i] for i in group])
        s = len(group)

        current_best_fitness = c_b_f
        better_solu_found = False
        self.miu = 0.08
        for sub in range(self.l):
            sigma = 0.01 * self.distance(x_sub, x_adv_sub)
            z_norm = np.random.randn(s, 1).T  # Random sample from a standard Gaussian distribution.    
            z = np.sqrt((sigma**2) * C_sub) * z_norm     # z = Cov_matrix**0.5 * z_norm   
            z = z[0]                                     # z ~ G(0, sigma**2 * Cov_matrix)        
            z = z * self.rescale         #test codes
            x_adv_sub_tmp = x_adv_sub + self.miu*(x_sub - x_adv_sub) + z    # x* = x* + miu*(x - x*) + z
            solu = x_adv.copy()
            for i, item in zip(group, x_adv_sub_tmp):
                solu[i] = item    
            current_fitness = loss(x, solu)       
            if (current_fitness < current_best_fitness):   
                print('pilot test get a better loss: ', current_fitness)   

                if (self.adv_count % 20 == 0):
                    filename = './ali/tempadv/openmydoor/' + str(self.t) + '.wav'
                    wavwrite(filename, solu)
                self.adv_count += 1

                current_best_fitness = current_fitness
                x_adv_sub = x_adv_sub_tmp.copy()
                P_sub = (1-self.c_c) * P_sub + np.sqrt(self.c_c * (2 - self.c_c)) * z / sigma   # Update the P_sub.
                C_sub = (1-self.c_cov)*C_sub + self.c_cov*(P_sub**2)     # Update the C_sub.
                better_solu_found = True
                self.miu = 1.5 * self.miu
            else:
                better_solu_found = False
                self.miu = 1.5**(-1/4) * self.miu
            self.t +=1
        #if (better_solu_found):
            #self.miu = 1.5 * self.miu
        #else:
            #self.miu = 1.5**(-1/4) * self.miu
        update_count = 0
        while (update_count < s):   # Update the original C, P, and the adversarial example.
            i = group[update_count]
            x_adv[i] = x_adv_sub[update_count]
            self.C[i] = C_sub[update_count]
            self.P[i] = P_sub[update_count]
            update_count += 1

        return x_adv, current_best_fitness

    def diag_matrix_sqrt(self, cov):    # Easily calculate the square root of a matrix.
        cov_diag = np.diagonal(cov)
        cov_diag1_2 = np.sqrt(cov_diag)
        cov1_2 = np.diag(cov_diag1_2)
        return cov1_2

    def attack(self, x, x_init_adv, loss ):
        print('start initialization!')   # test codes
        self.N = len(x)
        self.C = np.ones((self.N))
        self.P = np.zeros((self.N))  

        x_adv = x_init_adv.copy()
        self.t = 0
        last_best_fitness = self.distance(x, x_adv)
        current_best_fitness = 0
        
        while (self.t < self.T):
            x_adv, current_best_fitness = self.bin_search(x, x_adv, loss)
            last_best_fitness = current_best_fitness
            self.t += self.b
            strategy = self.group_strategy()
            groups = self.decompose(strategy)
           
            for group in groups:
                x_sub = np.array([x[i] for i in group])
                x_adv_sub = np.array([x_adv[i] for i in group])
                C_sub = np.array([self.C[i] for i in group])
                P_sub = np.array([self.P[i] for i in group])
                s = len(group)
                better_solu_found = False
                self.miu = 0.08
                for sub in range(self.l):
                    sigma = 0.01 * self.distance(x_sub, x_adv_sub)
                    z_norm = np.random.randn(s, 1).T  # Random sample from a standard Gaussian distribution.
                  
                    z = np.sqrt((sigma**2) * C_sub) * z_norm     # z = Cov_matrix**0.5 * z_norm   
                    z = z[0]                                     # z ~ G(0, sigma**2 * Cov_matrix)        
                    z = z * self.rescale         #test codes
                  
                    x_adv_sub_tmp = x_adv_sub + self.miu*(x_sub - x_adv_sub) + z    # x* = x* + miu*(x - x*) + z
                    solu = x_adv.copy()
              
                    for i, item in zip(group, x_adv_sub_tmp):
                        solu[i] = item
                  
                    current_fitness = loss(x, solu)
                    if (current_fitness < current_best_fitness):
                        print('sampled a better noise, loss: ', current_fitness)    
    
                        if (self.adv_count % 20 == 0):
                            filename = './ali/tempadv/openmydoor/' + str(self.t) + '.wav'
                            wavwrite(filename, solu)
                        self.adv_count += 1

                        current_best_fitness = current_fitness
                        x_adv_sub = x_adv_sub_tmp.copy()
                        P_sub = (1-self.c_c) * P_sub + np.sqrt(self.c_c * (2 - self.c_c)) * z / sigma   # Update the P_sub.
                        C_sub = (1-self.c_cov)*C_sub + self.c_cov*(P_sub**2)     # Update the C_sub.
                        better_solu_found = True
                        self.miu = 1.5 * self.miu
                    else:
                        better_solu_found = False
                        self.miu = 1.5**(-1/4) * self.miu
                #if (better_solu_found):
                    #self.miu = 1.5 * self.miu
                #else:
                    #self.miu = 1.5**(-1/4) * self.miu
                    self.t += 1
             
                update_count = 0
                while (update_count < s):   # Update the original C, P, and the adversarial example.
                    i = group[update_count]
                    x_adv[i] = x_adv_sub[update_count]
                    self.C[i] = C_sub[update_count]
                    self.P[i] = P_sub[update_count]
                    update_count += 1

                if (self.t >= self.T):
                    return x_adv
            
            # Record the performance of grouping strategy.
            self.R[strategy] = abs((last_best_fitness - current_best_fitness) / last_best_fitness * self.m)
            self.delta[1] = self.R[strategy]
            last_best_fitness = current_best_fitness

            # Run a pilot test to decide the group size (m).
            if (self.delta[0] == 0 and self.m_base/2 >= 1):
                self.m = self.m_base/2
                x_adv, current_best_fitness = self.pilot_test(x, x_adv, loss, strategy, current_best_fitness)
                self.delta[0] = abs((last_best_fitness - current_best_fitness) / last_best_fitness)
                last_best_fitness = current_best_fitness
                #t += self.l
                 
            elif (self.delta[2] == 0 and self.m_base*2 <= self.N):
                self.m = self.m_base*2
                x_adv, current_best_fitness = self.pilot_test(x, x_adv, loss, strategy, current_best_fitness)
                self.delta[2] = abs((last_best_fitness - current_best_fitness) / last_best_fitness)
                last_best_fitness = current_best_fitness
                #t += self.l
            print('current delta:', self.delta, '   current m: ', self.m_base)

            if(self.delta[0] == 0 and self.delta[1] == 0 and self.delta[2] ==0):
                #if (self.m_base*2 <= self.N):
                    #self.m_base = self.m_base*2
                pass
            elif (self.delta[0] == self.delta[1] and self.delta[0] == self.delta[2]):
                pass
            elif (self.delta[0] >= self.delta[1] and self.delta[0] >= self.delta[2]):
                self.m_base = self.m_base/2
                self.delta[2] = self.delta[1]
                self.delta[1] = self.delta[0]
                self.delta[0] = 0
            elif (self.delta[2] >= self.delta[0] and self.delta[2] >= self.delta[1]):
                self.m_base = self.m_base*2
                self.delta[0] = self.delta[1]
                self.delta[1] = self.delta[2]
                self.delta[2] = 0
            self.m = self.m_base
            print('current time: ', self.t, '      current loss: ', current_best_fitness)
            


def main():
    occam = Occam(b , T, l ,  c_c , c_cov , rescale)
    change_sample_rate(original_path, original_path , 16000)
    change_sample_rate(initial_adv_path, initial_adv_path , 16000)
    x = wavread(original_path)
    x_init_adv = wavread(initial_adv_path)
    pad_x = padding(x_init_adv, x)
    re, success, na = aliyun_function.aliyun_recong(initial_adv_path)


    final_adv = occam.attack( x, pad_x, ali_api_loss )
    wavwrite('./ali/final_adv_openmydoor.wav', final_adv)

main()

