import os
import librosa
import soundfile as sf
input_folder = 'F:\\yhc\\bone\\enhanced'
output_folder = 'F:\\yhc\\bone\\enhanced\\dccrn16to8'
sr = 8000

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for file_name in os.listdir(input_folder):
    if file_name.endswith('.wav'):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, file_name)
        y, _ = librosa.load(input_path, sr=sr)
        # librosa.output.write_wav(output_path, y, sr)
        sf.write(output_path, y, sr)

