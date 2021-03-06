import os
import argparse
import re


def get_top_crp_prf_from_file(file_path):
    crp = None
    with open(file_path, 'r') as f:
        line = f.readline()
        m = re.match(r'.*class_label #C=(\d+),#R=(\d+),#P=(\d+)', line)
        if m:
            C = int(m.group(1))
            R = int(m.group(2))
            P = int(m.group(3))
            precision = float(C)/float(P) if P > 0 else 0.0
            recall = float(C)/float(R) if R > 0 else 0.0
            f1 = (2 * precision * recall)/( precision + recall) if (precision + recall) > 0 else 0.0
            crp = {
                'C': int(m.group(1)),
                'R': int(m.group(2)),
                'P': int(m.group(3))
                }
            prf = {
                'P': precision,
                'R': recall,
                'F1': f1
                }
        else:
            print(file_path)
    return crp, prf


def _main(args):
    crp_total = {
        'C': 0,
        'R': 0,
        'P': 0
    }

    prf_total = {
        'P':  0,
        'R':  0,
        'F1': 0,
        'n':  0
    }

    out_lines = []

    if args.input_files:
        files = args.input_files
    else:
        files = []
        with open(args.input_filelist, 'r') as fh:
            for line in fh:
                files.append(line.strip())
    
    for file_ in files:
        crp, prf = get_top_crp_prf_from_file(file_)
        crp_total['C'] += crp['C']
        crp_total['R'] += crp['R']
        crp_total['P'] += crp['P']
        prf_total['P'] += prf['P']
        prf_total['R'] += prf['R']
        prf_total['F1'] += prf['F1']
        prf_total['n'] +=1

    P = float(crp_total['C'])/float(crp_total['P']) if float(crp_total['P']) > 0 else 0.0 
    R = float(crp_total['C'])/float(crp_total['R']) if float(crp_total['R']) > 0 else 0.0
    F = (2 * P * R) / ( R + P) if ( R + P) > 0 else 0.0 

    macro_P = prf_total['P'] / float(prf_total['n']) if float(prf_total['n']) > 0 else 0.0
    macro_R = prf_total['R'] / float(prf_total['n']) if float(prf_total['n']) > 0 else 0.0
    macro_F = prf_total['F1'] / float(prf_total['n']) if float(prf_total['n']) > 0 else 0.0
    
    micro_line = 'micro class_label #C={},#R={},#P={} R,P,F={:.2},{:.2},{:.2}\n'.format(
        crp_total['C'],
        crp_total['R'],
        crp_total['P'],
        R,
        P,
        F
    )
    macro_line = 'macro class_label #C={},#R={},#P={} R,P,F={:.2},{:.2},{:.2}\n'.format(
        crp_total['C'],
        crp_total['R'],
        crp_total['P'],
        macro_R,
        macro_P,
        macro_F
    )
    out_lines.append(micro_line)
    out_lines.append(macro_line)
    with open(args.output_file, 'w') as f:
        f.writelines(out_lines)
    

def setup_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', type=str, nargs='+', required=False)
    parser.add_argument('--input_filelist', type=str, required=False)
    parser.add_argument('--output_file', type=str, required=True)
    return parser


if __name__ == "__main__":
    parsed_args = setup_parse().parse_args()
    assert (parsed_args.input_files is None or parsed_args.input_filelist is None)
    assert not (parsed_args.input_files is None and parsed_args.input_filelist is None)
    _main(parsed_args)
