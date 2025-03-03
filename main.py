from tqdm import tqdm
from Bio.SeqUtils import MeltingTemp
from seqfold import dg
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('fastafile', type=str, help='Fasta file with an analyte sequence')
parser.add_argument('--length', '-l', type=int, default=100, required=True, help='Desired length of target sequence')
parser.add_argument('--arms', '-a', choices=[3, 4], default=3, type=int, required=True, help='Number of arms in sensor machine')
sensor_types_group = parser.add_mutually_exclusive_group(required=True)
sensor_types_group.add_argument('--dnazyme', '-dz', action='store_true', help='Sensors will be calculated for a DNAzyme-type machine')
sensor_types_group.add_argument('--g-quadruplex', '-gq', action='store_true', help='Sensors will be calculated for a G-quadruplex-type machine')

def load_fasta(filepath):
    try:
        with open(filepath, 'r') as file:
            header = file.readline()
            sequence = ''.join(line.strip() for line in file)
        return sequence
    
    except Exception as e:
        print(f'Error during loading fasta file: {e}')

def sliding_window(analyte_seq: str, win_len: int) -> list:
    '''Creates a list of targets of desired length within analyte sequence.'''
    if len(analyte_seq) < win_len:
        raise ValueError('Analyte sequence has to have more NBs than desired window length.')
    
    try:
        complementarity_map = {'A': 'T',
                               'T': 'G',
                               'G': 'C',
                               'C': 'G'}
        sensors = [
            ''.join(complementarity_map[base] for base in analyte_seq[i:(i+win_len)])
            for i in range(len(analyte_seq) - win_len + 1)]
        return sensors
    
    except Exception as e:
        print(f'Error during generating potential targets with sliding window: {e}')


def arms_validation(seqs: list, arms_num: int) -> list:
    '''Checks sensor sequences for arms with valid CG percentage and melting temperature.'''
    
    def _gc_perc_t_melt_calc(seq):
        gc_perc = ((seq.count('G') + seq.count('C')) / len(seq)) * 100
        return 40 <= gc_perc <= 60 and 55 <= MeltingTemp.Tm_NN(seq) <= 60

    results = []
    for sensor_seq in tqdm(seqs, desc='Generating valid arms'):
        seq_len = len(sensor_seq)
        for arm_len in range(18, 26):
            total_arms_len = arm_len * arms_num
            if total_arms_len > seq_len:
                continue #TODO if arms are longer than sequence what to do
            for start in range(seq_len - total_arms_len + 1):
                chunks = [sensor_seq[(start + i * arm_len):(start + (i + 1) * arm_len)]
                          for i in range(arms_num)]
                if all(_gc_perc_t_melt_calc(chunk) for chunk in chunks):
                    validated_arms = {
                                    'sequence': sensor_seq,
                                    'arms_len': arm_len,
                                    'arms': chunks
                                }
                    results.append(validated_arms)
    return results

def get_total_dg(entry):
    return sum(dg(arm, temp = 37) for arm in entry['arms'])

def get_sorted_arms(arms, key_func):
    result = []
    for arm in tqdm(arms, desc='Evaluating arms by delta G'):
        key_value = key_func(arm)
        result.append((key_value, arm))
    result.sort(key=lambda x: x[0])
    return [item[1] for item in result][:20]

def conctruct_arms(arms, sensor_type, arms_num):
    if sensor_type in ["dnazyme", "dz"]:
        if arms_num == 3:
            pass #TODO
        else:
            pass #TODO
    else: 
        if arms_num == 3:
            pass #TODO
        else:
            pass #TODO



if __name__ == '__main__':
    args = parser.parse_args()
    analyte = load_fasta(args.fastafile)
    sensors = sliding_window(analyte, args.length)
    print(f'{len(sensors)} potential sensor sequences of length {args.length} generated.')
    valid_arms = arms_validation(sensors, args.arms)
    print(f'{len(valid_arms)} arms of length 18-25, GC% 40-50 and Tmelt 55-60 generated.')
    best_dg_arms = get_sorted_arms(valid_arms, get_total_dg)
    print('10 arms with lowest sum dG:')
    #construct_arms
    for arm in best_dg_arms:
        print(arm)
    
