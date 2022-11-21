# coding=utf-8
import glob
import argparse

from aliyun_function import aliyun_recong
import time, os
from utils import *
from tools import find_all_ext

# os.environ['all_proxy'] = r'export all_proxy=socks5h://10.10.71.195:1080'

f = open("test.txt","a")
f.write('############ aliyun api ############\n\n')

def aliyun_decode_multi(audio_files: List[str], args = None, save_result: bool = True, max_workers: int = None, max_repeat_time: int = 1, wait_seconds: int = 10, output_format="transaction", language="cn", re_decode_failed: bool = False) -> List[str]:
    """
    using multi-process to decode audio files asynchronously.
    Because of the CancelException of cortana api, you can use max_repeat_time to control the re-decode times, and use wait_seconds in seconds to control the wait time between two queries.
    :return: transactions of every audio file.
    """

    '''
    with futures.ProcessPoolExecutor(max_workers=max_workers) as _executor_:
        jobs = []
        for audio_file in audio_files:
            jobs.append(
                _executor_.submit(
                    aliyun_decode, audio_file, save_result, language=language
                )
            )
        results = wait_for_jobs(jobs, _executor_, "[aliyun] Decode Sample Progress: ")
        return results
    '''

    for audio_file in audio_files:
        ret = aliyun_decode(audio_file, args)
        if ret == -1:
            continue
        time.sleep(0.5)

def aliyun_find(json_file: str, expected_string: str, delete_stop_words: bool = False) -> Tuple[bool, Union[str, None], Union[float, None]]:
    did_find, decode_string, find_confidence = False, None, None
    
    if os.path.exists(json_file)==True:
        with open(json_file, 'r') as _file_:
            decode_result = json.load(_file_)
        
        if decode_result['success']==True:
            decode_result['result'] = decode_result['result'].casefold()
            expected_string = expected_string.casefold()
        
            if all(expected_word in decode_result['result'] for expected_word in expected_string.split()):
                did_find = True
                decode_string = decode_result['result']
                find_confidence = decode_result['confidence']
    else:
        print('File '+json_file+' dosen\'t exist!\n')

    #print('did_find:\t'+str(did_find))
    #print('decode_string:\t'+str(decode_string))
    #print('confidence:\t'+str(find_confidence))
    
    return did_find, decode_string, find_confidence
    
# keywords = list()
# with open('keywords.txt', encoding='utf-8') as fkw:
#     for line in fkw:
#         word = line.strip()
#         if len(word) != 0:
#             keywords.append(word)
# print('keywords:', keywords)
# def is_success(cmd):
#     for kw in keywords:
#         if kw in cmd:
#             return True
#     return False

@exception_printer
def aliyun_decode(audio_file: str, args = None, save_result: bool = True, _wait_seconds_: int = 60, output_format="transaction", language="cn"):
    if (args!=None) and (args.re_decode_all==False):
        json_path = audio_file.replace('.wav', ALIYUN_JSON_SUFFIX)
        if os.path.exists(json_path) == True:
            # print('#################### Files Are Already Decoded #################')
            # sys.exit()
            print(f'#################### {audio_file} Is Already Decoded #################')
            return -1
        if 'cvte_3_output' in json_path:
            print(f'#################### {audio_file} Skipped #################')
            return -1
        # with open(audio_file[:-3] + 'txt', encoding='utf-8') as fpred:
        #     pred = fpred.readline().split(':')[-1].strip()
        # if is_success(pred) is False:
        #     print(f'#################### {audio_file} is not success #################')
        #     return -1


    res, success, file_name = aliyun_recong(audio_file)

    if success==True:
        wav_result = {
            "success": True,
            "errorMSG": None,
            "result": res,
            "confidence": None
        }
    else:
        wav_result = {
            "success": False,
            "errorMSG": "Miss Error",
            "result": "Miss Error",
            "confidence": None
        }
        wav_result['errorMSG']=res

    if save_result:
        with open(audio_file.replace('.wav', ALIYUN_JSON_SUFFIX), 'w') as _file_:
            json.dump(wav_result, _file_, ensure_ascii=False)
        
    print(audio_file.split('/')[-1])
    print('RESULT:\t'+str(wav_result)+'\n')
    
    f.write(audio_file.split('/')[-1])
    f.write(str(wav_result))
    f.write('\n\n')
    
    if output_format == "transaction":
        return wav_result['result']
        
    if output_format == "json":
        return wav_result

def api_recognize(args):
    logger.info("Start decode folder {} using aliyun.\n".format(args.wav_folder))

    # wav_files = glob.glob(os.path.join(args.wav_folder, "**", "*.wav"), recursive=True)
    wav_files = find_all_ext(args.wav_folder, 'wav')
    print('Number of wav filed found initially:', len(wav_files))
    wav_files = filter_irrelevant_wav(wav_files)
    print('Number of wav filed found:', len(wav_files))

    aliyun_decode_multi(wav_files, args, re_decode_failed = args.re_decode_failed)


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Google Speech to Text.')
    parser.add_argument('--wav_folder', type=str, help="Where is the wav_folder.")
    parser.add_argument('--re_decode_all', action="store_true", default=False, help="Whether re-decode all the wave files. Default is False.")
    parser.add_argument('--re_decode_failed', action="store_true", default=False, help="Whether to re-decode the failed decoding wave files. Default is False.")
    parser.add_argument('--max_workers', default=1, type=int, help="The number of the multi-process. Default is MAX.")
    parser.add_argument("--language", "-l", default='cn', choices=['en', 'cn'])  # 阿里云的语言要在网页端设置：全部项目->项目功能配置
    args = parser.parse_args()

    # Generate aliyun json files
    api_recognize(args)
    
    f.close()
    '''
    # Check json files to find the satisfied wav files
    Find = []
    json_files = [files for files in os.listdir(args.wav_folder) if IFLYTEC_JSON_SUFFIX in files]
    if len(json_files)==0:
        print('No json file is found! Please decode the wav file first!')
        sys.exit()
    
    print('Finding...')
    for i,json_file in enumerate(json_files):
        #print(i)
        did_find, decode_string, find_confidence = aliyun_find(os.path.join(args.wav_folder, json_file), 'play music')
        if did_find:
            Find.append(json_file)
            
    if len(Find)==0:
        print('Find Nothing satisfied!\n')
    else:
        print('\n'.join(Find))
    '''
