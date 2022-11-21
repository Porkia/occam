# -*- coding: utf-8 -*-
import os

import http.client
import json
import scipy.io.wavfile as wav
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest

'''
运行程序，只需要该aliyun_recong函数里的token字段即可

注意每次token有时间限制：
获取token要先登陆阿里云网址 https://nls-portal.console.aliyun.com/overview
然后点击左上角的功能菜单，选择我自定义的智能语音交互项目；
然后点击总览，然后在右上角找到“获取token”。
'''


def gettoken():
    # 创建AcsClient实例
    client = AcsClient(
        " ",            # AccessKey ID
        " ",            # AccessKey Secret
        "cn-shanghai"
    )

    # 创建request，并设置参数。
    request = CommonRequest()
    request.set_method('POST')
    request.set_domain('nls-meta.cn-shanghai.aliyuncs.com')
    request.set_version('2019-02-28')
    request.set_action_name('CreateToken')
    response = client.do_action_with_exception(request)
    jss = json.loads(response)
    token = jss['Token']['Id']
    
    return token


host = 'nls-gateway.cn-shanghai.aliyuncs.com'
# host = r'106.15.83.44'
proxy_ip = r'http://10.10.71.195'
proxy_port = 8080
def process(request, token, path):
    # 读取音频文件
    with open(path, mode='rb') as f:
        audioContent = f.read()
    # 设置HTTP请求头部
    httpHeaders = {
        'X-NLS-Token': token,
        'Content-type': 'application/octet-stream',
        'Content-Length': len(audioContent)
    }
    # Python 2.x使用httplib
    # conn = httplib.HTTPConnection(host)
    # Python 3.x使用http.client
    conn = http.client.HTTPConnection(host)
    # conn = http.client.HTTPConnection(proxy_ip, proxy_port)
    # conn.set_tunnel(host)
    conn.request(method='POST', url=request, body=audioContent, headers=httpHeaders)
    response = conn.getresponse()
    # print('Response status and response reason:')
    # print(response.status, response.reason)
    body = response.read()
    try:
        # print('Recognize response is:')
        body = json.loads(body)
        # print(body)
        status = body['status']
        if status == 20000000:
            result_output = body['result']
            return result_output, True
            # print('Recognize result: ' + result)
        else:
            # print('Recognizer failed!',body)
            conn.close()
            return f"Recognizer failed! ERROR number:{status}", False
    except ValueError:
        # print('The response is not json format string')
        conn.close()
        return "The response is not json format string",False
    conn.close()


def aliyun_recong(path):
    filename = path.split('/')[-1]

    appKey = ' '        # App key
    token = gettoken()

    url = f'http://{host}/stream/v1/asr'

    rate, _ = wav.read(path)
    # print("rate is :", rate)

    # if rate == 8000:
    format = 'pcm'
    # print(rate)
    sampleRate = rate# 8000
    enablePunctuationPrediction = True
    enableInverseTextNormalization = True
    enableVoiceDetection = False

    # 设置RESTful请求参数
    request = url + '?appkey=' + appKey
    request = request + '&format=' + format
    request = request + '&sample_rate=' + str(sampleRate)

    if enablePunctuationPrediction:
        request = request + '&enable_punctuation_prediction=' + 'true'

    if enableInverseTextNormalization:
        request = request + '&enable_inverse_text_normalization=' + 'true'

    if enableVoiceDetection:
        request = request + '&enable_voice_detection=' + 'true'

    # print('Request: ' + request)

    result,Success = process(request, token, path)
    # else:
    #     raise RuntimeError("rate must is 8000")
    return result, Success, filename


def getfile_path(filepath):
    l_dir = os.listdir(filepath)
    filepaths = []
    for one in l_dir:
        full_path = os.path.join('%s/%s' % (filepath, one))  # 构造路径
        filepaths.append(full_path)
    return filepaths


if __name__ == '__main__':
    # 音频文件
    #path = "/home/yxj/whereismycarTTSaudio.wav" # (16000)
    path = "init_adv.wav"
    re, success, na = aliyun_recong(path)
    #print("%s:\t%s" % (na, re))
    # for path in getfile_path(path):
    #     re, na = aliyun_recong(path)
    #     print("%s:\t%s" % (na, re))
    print(re)
