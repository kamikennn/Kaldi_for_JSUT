import os,sys
import glob
from sklearn.model_selection import train_test_split
import MeCab
import subprocess
import numpy as np
np.random.seed(seed=32)

def sort_file(fname):
    subprocess.call(f'sort {fname} > {fname}.sorted',shell=True)
    subprocess.call(f'rm {fname}',shell=True)
    subprocess.call(f'mv {fname}.sorted {fname}',shell=True)

def convert_wav(wav_data_path,out_dir):
    '''
    * sampling frequency must be 16kHz
    * wav file of JSUT is 48kHz, so convert to 16kHz using sox
        e.g. FILE_ID sox [input_wavfilename] -r 16000 [output_wavfilename]
    '''
    for wav_data in wav_data_path:
        fname = wav_data.split('/')[-1]
        subprocess.call(f'sox {wav_data} -r 16000 {out_dir}/{fname}',shell=True)
        subprocess.call(f'chmod 774 {out_dir}/{fname}',shell=True)

def make_wavscp(wav_data_path_list,out_dir,converted_jsut_data_dir):
    '''
    wav.scp: format -> FILE_ID cat PATH_TO_WAV |
    '''
    out_fname = f'{out_dir}/wav.scp'
    with open(out_fname,'w') as out:
        for wav_data_path in wav_data_path_list:
            file_id = wav_data_path.split('/')[-1].split('.')[0]
            out.write(f'{file_id} cat {converted_jsut_data_dir}/{file_id}.wav |\n')
    sort_file(out_fname)

def make_transcript(transcript_data_path_list,train_dir,eval_dir,error_dir,eval_wav_data_list):
    '''
    text: format -> UTT_ID  TRANSCRIPT
        * UTT_ID == FILE_ID (one wav file <-> one utterance)
    '''
    # change hankaku to zenkaku
    ZEN = "".join(chr(0xff01 + i) for i in range(94))
    HAN = "".join(chr(0x21 + i) for i in range(94))
    HAN2ZEN = str.maketrans(HAN,ZEN)

    eval_utt_id_list = []
    for eval_wav_data in eval_wav_data_list:
        eval_utt_id_list.append(eval_wav_data.split('/')[-1].split('.')[0])

    word_reading_fname = './word_reading.txt'
    word_reading_dict = {}  # {'word':'reading'}
    with open(word_reading_fname,'r') as f:
        lines = f.readlines()
        for line in lines:
            split_line = line.strip().split('+')
            word_reading_dict[split_line[0]] = split_line[1]

    out_train_fname = f'{train_dir}/transcript'
    out_eval_fname  = f'{eval_dir}/transcript'
    out_no_reading_word_fname = f'{error_dir}/no_reading_word.txt'
    no_reading_word_list = []
    chasen_tagger = MeCab.Tagger ("-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd")
    with open(out_train_fname,'w') as out_train, open(out_eval_fname,'w') as out_eval,\
            open(out_no_reading_word_fname,'w') as no_reading:
        for transcript_data_path in transcript_data_path_list:
            with open(transcript_data_path,'r') as trans:
                line = trans.readline()
                while line:
                    split_line = line.strip().split(':')
                    utt_id = split_line[0]
                    transcript = split_line[1].translate(HAN2ZEN)
                    transcript = transcript.replace('・',' ').replace('－',' ').replace('』',' ').replace('『',' ').replace('」',' ').replace('「',' ')
                    node = chasen_tagger.parseToNode(transcript)
                    transcript_line = []
                    while node:
                        feature = node.feature
                        if feature != 'BOS/EOS,*,*,*,*,*,*,*,*':
                            surface = node.surface
                            split_feature = feature.split(',')
                            reading = split_feature[-1]
                            part_of_speech = '/'.join(split_feature[:2]).replace('/*','')
                            # extract no reading word to error/no_reading_word_list.txt
                            if reading == '*':
                                if surface not in no_reading_word_list:
                                    no_reading_word_list.append(surface)
                                    no_reading.write(f'{surface}\n')
                            if surface == '、' or surface == '。' or surface == '，' or surface == '．':
                                transcript_line.append('<sp>')
                            elif surface != 'ー':
                                if reading == '*':
                                    reading = word_reading_dict[surface]
                                    transcript_line.append('{}+{}+{}'.format(surface,reading,part_of_speech))
                                else:
                                    transcript_line.append('{}+{}+{}'.format(surface,reading,part_of_speech))
                        node = node.next
                    transcript_line = ' '.join(transcript_line)
                    if utt_id in eval_utt_id_list:
                        out_eval.write(f'{utt_id}  {transcript_line}\n')
                    else:
                        out_train.write(f'{utt_id}  {transcript_line}\n')
                    line = trans.readline()

    sort_file(out_train_fname)
    sort_file(out_eval_fname)

def make_text(train_dir,eval_dir):
    train_transcript_fname  = f'{train_dir}/transcript'
    eval_transcript_fname   = f'{eval_dir}/transcript'
    out_train_fname         = f'{train_dir}/text'
    out_eval_fname          = f'{eval_dir}/text'
    with open(train_transcript_fname,'r') as trian_trans, open(eval_transcript_fname,'r') as eval_trans, \
            open(out_train_fname,'w') as out_train, open(out_eval_fname,'w') as out_eval:
        train_trans_line = trian_trans.readline()
        while train_trans_line:
            split_train_trans_line = train_trans_line.strip().split(' ')
            # if <sp> is in End of Sentence then remove it.
            if split_train_trans_line[-1] == "<sp>":
                split_train_trans_line.pop(-1)
            out_train.write(split_train_trans_line[0]+' ')  # write utt_id
            for i,word in enumerate(split_train_trans_line[2:]):
                if word == '<sp>':
                    out_train.write(' <sp>')
                else:
                    split_word = word.split('+')
                    out_train.write(' {}+{}'.format(split_word[0],split_word[2]))
            out_train.write('\n')
            train_trans_line = trian_trans.readline()

        eval_trans_line = eval_trans.readline()
        while eval_trans_line:
            split_eval_trans_line = eval_trans_line.strip().split(' ')
            # if <sp> is in End of Sentence then remove it.
            if split_eval_trans_line[-1] == "<sp>":
                split_eval_trans_line.pop(-1)
            out_eval.write(split_eval_trans_line[0]+' ')  # write utt_id
            for i,word in enumerate(split_eval_trans_line[2:]):
                if word == '<sp>':
                    out_eval.write(' <sp>')
                else:
                    split_word = word.split('+')
                    out_eval.write(' {}+{}'.format(split_word[0],split_word[2]))
            out_eval.write('\n')
            eval_trans_line = eval_trans.readline()

    sort_file(out_train_fname)
    sort_file(out_eval_fname)

def make_lexicon(train_dir,lexicon_dir):
    '''
    lexicon: format -> 'word'+'part of speech'
    '''
    transcript_fname        = f'{train_dir}/transcript'
    out_lexicon_fname       = f'{lexicon_dir}/lexicon.txt'
    out_lexicon_htk_fname   = f'{lexicon_dir}/lexicon_htk.txt'
    with open(transcript_fname,'r') as trans, open(out_lexicon_fname,'w') as out:
        trans_line = trans.readline()
        while trans_line:
            split_trans_line = trans_line.strip().split(' ')[2:]
            for word in split_trans_line:
                if word != '<sp>':
                    out.write(word+'\n')
            trans_line = trans.readline()

    subprocess.call(f'sort -u {out_lexicon_fname} > {out_lexicon_htk_fname}',shell=True)
    subprocess.call(f'./local/csj_make_trans/vocab2dic.pl -p local/csj_make_trans/kana2phone -e ./data/lexicon/ERROR_v2d -o {out_lexicon_fname} {out_lexicon_htk_fname}',shell=True)
    subprocess.call(f"cut -d'+' -f1,3- {out_lexicon_fname} >{out_lexicon_htk_fname}",shell=True)
    subprocess.call(f"cut -f1,3- {out_lexicon_htk_fname} | perl -ape 's:\t: :g' >{out_lexicon_fname}",shell=True)

def make_utt2spk(dir):
    '''
    In JSUT corpus, speaker number is one person.
    It is not good for training Acoustic Model.
    '''
    text_fname            = f'{dir}/text'
    out_utt2spk_fname     = f'{dir}/utt2spk'
    speaker_id = "jsut_speaker"
    with open(text_fname,'r') as text, open(out_utt2spk_fname,'w') as out:
        text_line = text.readline()
        while text_line:
            utt_id = text_line.split(' ')[0]
            out.write(f'{utt_id} {speaker_id}\n')
            text_line = text.readline()


def make_spk2utt(dir):
    utt2spk_fname     = f'{dir}/utt2spk'
    out_spk2utt_fname = f'{dir}/spk2utt'
    with open(utt2spk_fname,'r') as utt2spk, open(out_spk2utt_fname,'w') as out:
        speaker_utt_dict = {}   # {'speaker_id':'utt_id'}
        utt2spk_line = utt2spk.readline()
        while utt2spk_line:
            split_utt2spk_line = utt2spk_line.strip().split(' ')
            utt_id = split_utt2spk_line[0]
            spk_id = split_utt2spk_line[1]
            if spk_id in speaker_utt_dict:
                speaker_utt_dict[spk_id].append(utt_id)
            else:
                speaker_utt_dict[spk_id] = [utt_id]
            utt2spk_line = utt2spk.readline()

        for spk_id, utt_id_list in speaker_utt_dict.items():
            out.write(f'{spk_id}')
            for utt_id in utt_id_list:
                out.write(f' {utt_id}')
            out.write('\n')


def main(args):
    data_dir    = './data'
    train_dir   = f'{data_dir}/train'
    eval_dir    = f'{data_dir}/eval'
    lexicon_dir = f'{data_dir}/lexicon'
    error_dir   = f'{data_dir}/error'
    original_jsut_data_dir = args[1]
    converted_jsut_data_dir = '/path/to/converted/JSUT/corpus'

    # make wav.scp of train and eval
    wav_data_path = glob.glob(f'{original_jsut_data_dir}/*/wav/*.wav')
    # convert jsut wav data to 16kHz
    convert_wav(wav_data_path,converted_jsut_data_dir)
    # split data  [train_size = 7196, test_size = 500]
    train_wav_data_list, eval_wav_data_list = train_test_split(wav_data_path, test_size=500)
    make_wavscp(train_wav_data_list,train_dir,converted_jsut_data_dir)
    make_wavscp(eval_wav_data_list,eval_dir,converted_jsut_data_dir)

    # make text of train and eval
    transcript_data_path = glob.glob(f'{original_jsut_data_dir}/*/transcript_utf8.txt')
    make_transcript(transcript_data_path,train_dir,eval_dir,error_dir,eval_wav_data_list)
    make_text(train_dir,eval_dir)

    # make lexicon fomr data/train/transcript
    make_lexicon(train_dir,lexicon_dir)

    # make utt2spk
    make_utt2spk(train_dir)
    make_utt2spk(eval_dir)

    # make spk2utt
    make_spk2utt(train_dir)
    make_spk2utt(eval_dir)

if __name__ == "__main__":
    args = sys.argv
    main(args)