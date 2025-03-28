import argparse
import pandas as pd
from Bio.SeqUtils import MeltingTemp
from tqdm import tqdm
import numpy as np
from seqfold import dg
from joblib import load
from datetime import datetime

complementarity_map = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G'
    }

def create_parser() -> argparse.ArgumentParser():
    '''
    Creates argument parser for usage through command line
    
    Returnes:
        parser: Configured argument parser
    '''
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('analyte_fastafile',
                            type=str,
                            help='FASTA format file containing analyte sequence')
        parser.add_argument('--window_length', '-wl',
                            type=int,
                            default=100,
                            help='Desired length of target sequence (default 100 bases)')
    
        sensor_types_group = parser.add_mutually_exclusive_group(required=True)
        sensor_types_group.add_argument('--dnazyme_binary', '-dz_bi',
                                        action='store_true',
                                        help='Binary DNAzymes with 2 arms will be generated')
        sensor_types_group.add_argument('--dnazyme_dnm', '-dz_dnm',
                                        action='store_true',
                                        help='DNM DNAzymes with 3 arms will be generated')
        sensor_types_group.add_argument('--g_quadruplex', '-g4',
                                        action='store_true',
                                        help='G-quadruplexes with 4 arms will be generated')
    
        analyte_types_group = parser.add_mutually_exclusive_group(required=True)
        analyte_types_group.add_argument('--dsDNA', '-ds', 
                                         action='store_true',
                                         help='Sensor will be tested for dsDNA analyte')
        analyte_types_group.add_argument('--ssDNA', '-ss', 
                                         action='store_true',
                                         help='Sensor will be tested for ssDNA analyte')
        analyte_types_group.add_argument('--amplicon', '-ampl', 
                                         action='store_true',
                                         help='Sensor will be tested for amplicon analyte')
        
        parser.add_argument('--compare_fastafile', '-c',
                            type=str,
                            help='FASTA format files of sequences to cross-compare arm binding')
        parser.add_argument('--primers', '-p',
                            type=str,
                            help='TXT format file with F0 and B0 primers; Text format: F0: {primer_sequence} \new_line B0: {primer_sequence}')
        
        return parser
    
    except Exception as e:
        raise Exception(f'Error parsing arguments: {e}')

def load_fasta(fastafile:str) -> tuple[str, str]:
    '''
    Loads and parses FASTA format file
    
    Args:
        fastafile: Path to the FASTA file
    
    Returns:
        header: Header text from the FASTA file (without '>' character)
        sequence: Sequence string containing valid nucleotide characters (ATGC)
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or not in FASTA format
    '''
    try:
        with open(fastafile, 'r') as file:
            header = file.readline().strip()

            if not header.startswith('>'):
                raise ValueError("Not a valid FASTA file - must start with '>'")

            header = header[1:]
            sequence = ''.join(line.strip() for line in file).upper()

            if not sequence:
                raise ValueError('Empty sequence in fasta file')
            bases = 'ATGC'
            invalid_bases = set(base for base in sequence if base not in bases)
            if invalid_bases:
                raise ValueError('Sequence contains invalid characters: ', ', '.join(invalid_bases))

            return header, sequence

    except FileNotFoundError:
        raise FileNotFoundError(f'File not found: {fastafile}')
    except Exception as e:
        raise Exception(f'Error processing FASTA file: {e}') 

def setup_constants(args:argparse.Namespace) -> pd.DataFrame:
    '''
    Sets up DataFrame to use in Random Forest model according to arguments passed to program
    
    Args:
        args: Arguments passed to the program that includes information about sensor and analyte
    
    Returns:
        constants: DataFrame with information about sensor and analyte
    
    Raises:
        Exception: If unable to set up constants
    '''
    try:
        constants = pd.DataFrame({
            'sensor_Binary': args.dnazyme_binary,
            'sensor_DNM': args.dnazyme_dnm,
            'core_DNAzyme': args.dnazyme_binary or args.dnazyme_dnm,
            'core_G-quadruplex': args.g_quadruplex,
            'analyte_amplicon': args.amplicon,
            'analyte_dsDNA': args.dsDNA,
            'analyte_ssDNA': args.ssDNA, 
        }, index=[0])
        
        return constants
    
    except Exception as e:
        raise Exception(f'Error setting up experimental constants: {e}')
    

def find_amplified_region_sequence(analyte_sequence:str, primers_filepath:str) -> str:
    '''
    Shortens analyte sequence according to primers

    Args:
        analyte_sequence: Initial full DNA sequence if analyte
        primers_filepath: TXT format file with F0 and B0 primers

    Returns:
        amplified_region_sequence: Shortened DNA sequence that is amplified with selected primers

    Raises:
        ValueError: If unable to find amplification sequence for analyte with given primers
    '''
    try:
        with open(primers_filepath, 'r') as file:
            fo_primer = file.readline().strip().split(': ')[1]
            bo_primer = file.readline().strip().split(': ')[1]
        bo_primer_reverse = ''.join(complementarity_map[base] for base in reversed(bo_primer))
        fo_position = analyte_sequence.find(fo_primer)
        bo_position = analyte_sequence.find(bo_primer_reverse)
        primers = [fo_primer, bo_primer]

        if fo_position == -1 and bo_position == -1:
            raise ValueError('No amplified region found for these primers')

        amplified_region_sequence = analyte_sequence[fo_position:(bo_position + len(bo_primer))]

        return amplified_region_sequence, primers

    except Exception as e:
        raise Exception(f'Error finding amplified region: {e}')

def sliding_window(analyte_sequence:str, window_length:int) -> list[str]:
    '''
    Generates possible target sequences of given length from analyte sequence, reversed and complementary
    
    Args:
        analyte_sequence: Full sequence to be cut into shorter target sequences
        window_legth: Length of target sequences

    Returns:
        target_reverse_complemetary_sequences: List of all target sequences of given length, reversed and complementary

    Raises:
        ValueError: If target sequences can't be generated with given parameters
    '''
    try:
        if window_length < 18*4:
            raise ValueError(f'Window length must be longer than {18*4}')
        if window_length > len(analyte_sequence):
            raise ValueError('Window length must be less or equal to analyte sequance length')

        target_reverse_complementary_sequences = []
        for i in range(len(analyte_sequence) - window_length + 1):
            target = ''.join(complementarity_map[base] for base in analyte_sequence[i:(i+window_length)][::-1])
            target_reverse_complementary_sequences.append(target)

        if not target_reverse_complementary_sequences:
            raise ValueError('No target sequences generated')

        return target_reverse_complementary_sequences

    except Exception as e:
        raise Exception(f'Error during generating target sequences with sliding window: {e}')


def validate_arms(target_sequences:list[str], args:argparse.Namespace, compare_sequence:str) -> pd.DataFrame:
    '''
    Generates and validates arms of 18-26 bases based on GC%, melting temperature, cross-complemetarity with other sequences and leaves only unique arm sets
    
    Args:
        target_sequences: List of target sequences to which arms are generated
        args: Parsed arguments with information about sensor and analyte
        compare_sequence: DNA sequence that is compared with each arm to avoid more than one complementary site
    
    Returns:
        validated_arms: DataFrame with information of all validated arms and their target sequence
    '''
    try:
        valid_arms_list = []
        arms_num = (2 if args.dnazyme_binary else
                    3 if args.dnazyme_dnm else
                    4 if args.g_quadruplex else None)
        compare_sequence_reverse_complement = ''.join(complementarity_map[base] for base in compare_sequence)[::-1]
        unique_arms = set()
    
        def _is_valid_arm(arm_sequence):
            '''Validation of GC% and melting temperature of an arm'''
            if not arm_sequence:
                return False
            gc_count = arm_sequence.count('G') + arm_sequence.count('C')
            gc_percentage = (gc_count / len(arm_sequence)) * 100
            if not (40 <= gc_percentage <= 60):
                return False
            try:
                melting_temp = MeltingTemp.Tm_GC(arm_sequence)
                return 55 <= melting_temp <= 60
            except:
                return False
        
        def _is_not_cross_complementary(arms, compare_sequence):
            '''Comparing arm set with given sequence so that there is no more than one complementarity site'''
            for arm in arms:
                if compare_sequence.count(arm) > 1:
                    return False
            return True
    
        for target in tqdm(target_sequences, desc='Validating arms'):
            target_length = len(target)
    
            for arm_length in range(18, min(26, target_length // arms_num + 1)):
                total_arms_length = arm_length * arms_num
                max_start = target_length - total_arms_length
    
                if max_start < 0:
                    continue
    
                for start in range(max_start + 1):
                    arms = tuple(target[(start + (i * arm_length)):(start + ((i + 1) * arm_length))]
                                 for i in range(arms_num))[::-1]
    
                    if arms in unique_arms:
                        continue
                    
                    if not all(_is_valid_arm(arm) for arm in arms):
                        continue
                    
                    if not _is_not_cross_complementary(arms, compare_sequence_reverse_complement):
                        continue
    
                    unique_arms.add(arms)
                    row = {'whole_seq': target, '1arm_seq': None, '2arm_seq': None, '3arm_seq': None, '4arm_seq': None}

                    for i, arm in enumerate(arms, 1):
                        row[f'{i}arm_seq'] = arm

                    valid_arms_list.append(row)
    
        valid_arms = pd.DataFrame(valid_arms_list) 

    except Exception as e:
        raise Exception(f'Error validating arms: {e}')
    
    return valid_arms

def modificate_arms(valid_arms_data:pd.DataFrame, args:argparse.Namespace) -> pd.DataFrame:
    '''
    Adds modification to arms based on sensor type and saves them as new columns in DataFrame
    
    Args:
        valid_arms: DataFrame that contains information about pregenerated arms
        args: Parsed arguments with information about sensor and analyte
    
    Returns:
        arms_data: DataFrame that contains information about pregenerated arms and their modifications
    '''
    arms_data = valid_arms_data.copy()
    if args.dnazyme_binary:
        arms_data['1arm_mod_seq'] = 'TGCCCAGGGAGGCTAGCT' + arms_data['1arm_seq']
        arms_data['2arm_mod_seq'] = arms_data['2arm_seq'] +  'ACAACGAGAGGAAACCTT'
        arms_data['3arm_mod_seq'] = None
        arms_data['4_1arm_mod_seq'] = None

    if args.dnazyme_dnm:
        arms_data['1arm_mod_seq'] = 'TGCCCAGGGAGGCTAGCT' + arms_data['1arm_seq']
        arms_data['2arm_mod_seq'] = 'GTACGTCAGGTGACAGTAGTCTGCTTTTTT' + arms_data['2arm_seq'] +  'ACAACGAGAGGAAACCTT'
        arms_data['3arm_mod_seq'] = arms_data['3arm_seq'] + 'TTTTTTGCAGACTACTGTCACCTGACGTAC'
        arms_data['4_1arm_mod_seq'] = None

    if args.g_quadruplex:
        arms_data['1arm_mod_seq'] = None
        arms_data['2arm_mod_seq'] = arms_data['2arm_seq'] + 'TTTTTTGGTTGGG'
        arms_data['3arm_mod_seq'] = 'ACCCGCTAATCTAACTAATCTACTTATTATACTATCTCTTTTTT' + arms_data['3arm_seq'] + 'TTTTTTGGTTGGG'
        arms_data['4_1arm_mod_seq'] = arms_data['4arm_seq'] + 'TTTTTTTCTATATCTTCTTCATCTATATCTTTTTT' + arms_data['1arm_seq']

    return arms_data

def calculate_dg(arms_data:pd.DataFrame, temperature:int) -> pd.DataFrame:
    '''
    Calculates dg for each modificated arm and saves it as new columns in DatFrame
    
    Args:
        arms_data: DataFrame with modificated arms sequences
        temperature: Temperature of ??? 
    ''' #XXX temp 
    # TODO add raise
    
    arm_columns = ['1arm_mod_seq', '2arm_mod_seq', '3arm_mod_seq', '4_1arm_mod_seq']
    dg_columns = ['1arm_mod_dg', '2arm_mod_dg', '3arm_mod_dg', '4_1arm_mod_dg']
    arms_energy_data = arms_data.copy()

    for arm_col, dg_col in zip(arm_columns, dg_columns):
        arms_energy_data[dg_col] = arms_energy_data[arm_col].progress_apply(
            lambda seq: dg(seq, temperature) if pd.notna(seq) else np.nan
        )
    
    arms_energy_data = arms_energy_data[~arms_energy_data[dg_columns].isin([float('inf')]).any(axis=1)].reset_index(drop=True)
    arms_energy_data['dg_sum'] = arms_energy_data[dg_columns].sum(axis=1)

    return arms_energy_data

def predicting_arms_success(arms_data, constants): #TODO Docstring
    constants_repeated = pd.concat([constants] * len(arms_data), ignore_index=True)
    y_df =  arms_data.assign(**constants_repeated.iloc[0].to_dict())
    rf_model = load('rf_model.joblib')
    feature_names = rf_model.feature_names_in_
    y_df_filtered = y_df[feature_names]
    probabilities = rf_model.predict_proba(y_df_filtered)
    class_1_probs = probabilities[:, 1]
    y_df['class_1_probability'] = class_1_probs
    top_3_predictions = y_df.nlargest(3, 'class_1_probability')
    predictions = rf_model.predict(y_df_filtered)
    df_class_1 = y_df[predictions == 1]
    
    return top_3_predictions

def generate_report(analyte_data, compare_data, args, prediction_results):
    report = []
    report.append('APTAMER DESIGN REPORT')
    report.append('=' * 50 + '\n')
    report.append(f'Analyte: {analyte_data}')
    analyte_type = ('dsDNA' if args.dsDNA else
                   'ssDNA' if args.ssDNA else
                   'amplicon' if args.amplicon else None)
    report.append(f'Analyte type: {analyte_type}')
    report.append(f'Compared against: {compare_data}')
    sensor_type = ('DNAzyme_binary' if args.dnazyme_binary else
                   'DNAzyme_DNM' if args.dnazyme_dnm else
                   'G-quadruplex' if args.g_quadruplex else None)
    report.append(f'Sensor type: {sensor_type}')
    report.append('')
    i = 1
    for index, result in prediction_results.iterrows():
        report.append(f'{i} best prediction:')
        report.append(f'Succes probability: {(result["class_1_probability"] * 100):.2f}%')
        target = ''.join(complementarity_map[base] for base in result['whole_seq'][::-1])
        report.append(f'Target: {target}')
        arms = '   '.join([
            arm for arm in [
                result['1arm_seq'],
                result['2arm_seq'],
                result['3arm_seq'],
                result['4arm_seq']
                ] if pd.notna(arm)
            ])
        report.append(f'Arms: {arms}')
        arms_mod = ['1arm_mod', '2arm_mod', '3arm_mod', '4_1arm_mod']
        for arm_mod in arms_mod:
            if result[f'{arm_mod}_seq']:
                arm_seq = result[f'{arm_mod}_seq']
                arm_dg = result[f'{arm_mod}_dg']
                report.append(f'{arm_mod}: {arm_seq}; dG: {arm_dg}')
        report.append(f'Sum dG: {result["dg_sum"]:.2f}')
        report.append('')
        i += 1
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sensor_{sensor_type}_{analyte_type}_{timestamp}.txt"
    with open(filename, 'w') as f:
        f.write('\n'.join(report))
        
def main(test_args):
    parser = create_parser()
    args = parser.parse_args(test_args)
    analyte_data, analyte_sequence = load_fasta(args.analyte_fastafile)
    compare_data, compare_sequence = load_fasta(args.compare_fastafile)
    constants = setup_constants(args)
    temperature = (23 if (args.dnazyme_binary or args.dnazyme_dnm) else 
                   55 if args.g_quadruplex else None)
    if args.primers:
        amplificated_sequence, primers = find_amplified_region_sequence(analyte_sequence, args.primers)
    else:
        amplificated_sequence = analyte_sequence
    target_reverse_complementary_sequences = sliding_window(amplificated_sequence, args.window_length)
    valid_arms = validate_arms(target_reverse_complementary_sequences, args, compare_sequence)
    if not valid_arms.empty:
        modificated_arms = modificate_arms(valid_arms, args)
        tqdm.pandas()
        arms_dg_data = calculate_dg(modificated_arms, temperature)
        top_arms_by_dg_sum = arms_dg_data.sort_values('dg_sum', ascending=True).head(100)
        
        top_arms_by_dg_sum['whole_dg'] = top_arms_by_dg_sum['whole_seq'].progress_apply(
            lambda seq: dg(seq, temperature) 
        )
        top_arms_by_dg_sum = top_arms_by_dg_sum[~np.isinf(top_arms_by_dg_sum['whole_dg'])]
        prediction_results = predicting_arms_success(top_arms_by_dg_sum, constants)   
            
        generate_report(analyte_data, compare_data, args, prediction_results)
    
if __name__ == '__main__':
    test_args_set = [
        ['test/test/stau.fasta', '-dz_dnm', '-ds', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p1.txt'],
        ['test/stau.fasta', '-dz_bi', '-ds', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p1.txt'],
        ['test/stau.fasta', '-g4', '-ds', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p1.txt'],
        ['test/stau.fasta', '-dz_bi', '-ss', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p1.txt'],
        ['test/stau.fasta', '-dz_dnm', '-ss', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p1.txt'],
        ['test/stau.fasta', '-g4', '-ss', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p1.txt'],
        ['test/stau.fasta', '-dz_bi', '-ampl', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p1.txt'],
        ['test/stau.fasta', '-dz_dnm', '-ampl', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p1.txt'],
        ['test/stau.fasta', '-g4', '-ampl', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p1.txt'],
        
        ['test/stau.fasta', '-dz_dnm', '-ds', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p2.txt'],
        ['test/stau.fasta', '-dz_bi', '-ds', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p2.txt'],
        ['test/stau.fasta', '-g4', '-ds', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p2.txt'],
        ['test/stau.fasta', '-dz_bi', '-ss', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p2.txt'],
        ['test/stau.fasta', '-dz_dnm', '-ss', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p2.txt'],
        ['test/stau.fasta', '-g4', '-ss', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p2.txt'],
        ['test/stau.fasta', '-dz_bi', '-ampl', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p2.txt'],
        ['test/stau.fasta', '-dz_dnm', '-ampl', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p2.txt'],
        ['test/stau.fasta', '-g4', '-ampl', '-c', 'test/stau_comp.fasta', '-p', 'primers/stau_p2.txt'],
        
        ['test/influ.fasta', '-dz_dnm', '-ds', '-c', 'test/influ_comp.fasta', '-p', 'test/primers/influ.txt'],
        ['test/influ.fasta', '-dz_bi', '-ds', '-c', 'test/influ_comp.fasta', '-p', 'test/primers/influ.txt'],
        ['test/influ.fasta', '-g4', '-ds', '-c', 'test/influ_comp.fasta', '-p', 'test/primers/influ.txt'],
        ['test/influ.fasta', '-dz_bi', '-ss', '-c', 'test/influ_comp.fasta', '-p', 'test/primers/influ.txt'],
        ['test/influ.fasta', '-dz_dnm', '-ss', '-c', 'test/influ_comp.fasta', '-p', 'test/primers/influ.txt'],
        ['test/influ.fasta', '-g4', '-ss', '-c', 'test/influ_comp.fasta', '-p', 'test/primers/influ.txt'],
        ['test/influ.fasta', '-dz_bi', '-ampl', '-c', 'test/influ_comp.fasta', '-p', 'test/primers/influ.txt'],
        ['test/influ.fasta', '-dz_dnm', '-ampl', '-c', 'test/influ_comp.fasta', '-p', 'test/primers/influ.txt'],
        ['test/influ.fasta', '-g4', '-ampl', '-c', 'test/influ_comp.fasta', '-p', 'test/primers/influ.txt'],
        ]
    for test_args in test_args_set:
        main(test_args)