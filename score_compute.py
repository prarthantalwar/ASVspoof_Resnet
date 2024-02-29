import numpy as np
import eval_metrics as em

def compute_metrics(cm_score_file):
    asv_score_file = "/home/serb-s2st/workspace/matlab/PRT_SFF_Files/ASVspoof2019_root/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"  # Provide the path to your ASV score file

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  
        'Ptar': (1 - Pspoof) * 0.99,  
        'Pnon': (1 - Pspoof) * 0.01,  
        'Cmiss_asv': 1,  
        'Cfa_asv': 10,  
        'Cmiss_cm': 1,  
        'Cfa_cm': 10,  
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(float)

    other_cm_scores = -cm_scores

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    other_eer_cm = em.compute_eer(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_keys == 'spoof'])[0]

    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    if eer_cm < other_eer_cm:
        tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)
        min_tDCF_index = np.argmin(tDCF_curve)
        min_tDCF = tDCF_curve[min_tDCF_index]
    else:
        tDCF_curve, CM_thresholds = em.compute_tDCF(other_cm_scores[cm_keys == 'bonafide'], other_cm_scores[cm_keys == 'spoof'],
                                                    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model, True)
        min_tDCF_index = np.argmin(tDCF_curve)
        min_tDCF = tDCF_curve[min_tDCF_index]

    return min(eer_cm, other_eer_cm), min_tDCF

if __name__ == "__main__":
    cm_score_file = "/home/serb-s2st/workspace/matlab/PRT_SFF_Files/AIR-ASVspoof/models/softmax/checkpoint_cm_score.txt"  # Provide the actual path to your CM score file
    eer_cm, min_tDCF = compute_metrics(cm_score_file)
    print("EER:", eer_cm)
    print("Min tDCF:", min_tDCF)
