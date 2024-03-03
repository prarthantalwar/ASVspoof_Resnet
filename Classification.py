import os
import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score


# Define the path to the trained model (.pth file) and the file containing list of audio file names
model_path = "/home/serb-s2st/workspace/matlab/PRT_SFF_Files/AIR-ASVspoof/models/softmax/checkpoint/anti-spoofing_sffcc_model_100.pt"
audio_list_path = "/home/serb-s2st/workspace/matlab/PRT_SFF_Files/ASVspoof2019_root/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.eval_with_label.txt"
feature_file_path= "/media/serb-s2st/PortableSSD/Resnet/Features/eval/"

bonafide_pred=[]
spoof_pred=[]

# Define other necessary functions and modules
def process_audio_file(model, device, feature, label):
    # Convert feature to tensor and move to device
    feature_tensor = torch.tensor(feature).unsqueeze(0).unsqueeze(1).float().to(device)

    # Ensure model is in evaluation mode
    model.eval()

    # Forward pass
    with torch.no_grad():
        output, x = model(feature_tensor)  # Assuming the model returns both feature and predictions
        # print(output, x)

    # Apply softmax to get probabilities
    pred_values = torch.softmax(x, dim=1)
    yu, batch_pred = x.max(dim=1)
    
    # Map class index to label
    predicted_class_label = 0 if batch_pred == 0 else 1

    if label=="spoof":
        spoof_pred.append(predicted_class_label)
    else:
        bonafide_pred.append(predicted_class_label)

    # Return predicted class label
    return predicted_class_label


def load_feature_file(feature_file):
    with open(feature_file, 'rb') as f:
        features = pickle.load(f)
    return features

def test_individual_files(model, device, audio_list_path, feature_file_path):
    # Load the list of audio file names and their corresponding labels
    with open(audio_list_path, 'r') as f:
        audio_files_and_labels = f.readlines()
        
    audio_files = [line.strip().split(" ")[0] for line in audio_files_and_labels]
    labels = [line.strip().split(" ")[1] for line in audio_files_and_labels]

    results = {}
    for audio_file, label in tqdm(zip(audio_files, labels)):
        feature_file = os.path.join(feature_file_path, f"_{audio_file}SFFCC.pkl")
        if os.path.exists(feature_file):
            feature = load_feature_file(feature_file)
            result = process_audio_file(model, device, feature,label)
            results[audio_file] = result
        else:
            print(f"Feature file not found for {audio_file}")

    return results

if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = torch.load(model_path, map_location=device)
    model = model.to(device)
    model.eval()

    # Test individual audio files
    results = test_individual_files(model, device, audio_list_path, feature_file_path)

    # Print or save results as needed
    # print(results)

    # Count bonafide and spoof predictions
    bonafide_count = sum(1 for label in results.values() if label == "bonafide")
    spoof_count = sum(1 for label in results.values() if label == "spoof")

    # # Print counts of bonafide and spoof predictions
    # print("Counts:")
    # print("Bonafide:", bonafide_count)
    # print("Spoof:", spoof_count)


    # # Print counts of bonafide and spoof predictions
    # print("List Counts:")
    # print("Bonafide:", len(bonafide_pred), bonafide_pred)
    # print("Spoof:", len(spoof_pred), spoof_pred)


    # Create bonafide_actual list initialized with all 0s
    bonafide_actual = [0] * len(bonafide_pred)

    # Create spoof_actual list initialized with all 1s
    spoof_actual = [1] * len(spoof_pred)

    # Print lenghts
    print("Bonafide length: ",len(bonafide_pred))
    print("Spoof length: ",len(spoof_pred))
    
    
    # Calculate accuracy for bonafide class
    bonafide_accuracy = accuracy_score(bonafide_actual, bonafide_pred)

    # Calculate accuracy for spoof class
    spoof_accuracy = accuracy_score(spoof_actual, spoof_pred)

    print("Bonafide Classification Accuracy:", bonafide_accuracy)
    print("Spoof Classification Accuracy:", spoof_accuracy)
