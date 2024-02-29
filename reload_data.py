import pickle
from librosa.util import find_files
import scipy.io as sio

access_type = "LA"
# # on air station gpu
path_to_mat = '/media/serb-s2st/PortableSSD/ASV2021/Mat_features/'
path_to_audio = '/data/neil/DS_10283_3336/'+access_type+'/ASVspoof2019_'+access_type+'_'
path_to_features = '/media/serb-s2st/PortableSSD/ASV2021/Features/'

def reload_data(path_to_features, part):
    matfiles = find_files(path_to_mat + part + '/', ext='mat')
    for i in range(len(matfiles)):
        if matfiles[i][len(path_to_mat)+len(part)+1:].startswith('SFFCC'):
            key = matfiles[i][len(path_to_mat) + len(part) + 6:-4]
            sffcc = sio.loadmat(matfiles[i], verify_compressed_data_integrity=False)['x']
            with open(path_to_features + part +'/'+ key + 'SFFCC.pkl', 'wb') as handle2:
                pickle.dump(sffcc, handle2, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    #reload_data(path_to_features, 'train')
    #reload_data(path_to_features, 'dev')
    reload_data(path_to_features, 'eval')
