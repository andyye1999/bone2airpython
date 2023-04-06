
import os
import soundfile as sf

# Set the path to the parent folder containing all the subfolders with audio files
parent_folder = r"F:\\yhc\\ABCS\\ABCS_database_ciaic\\ABCS_database\\Audio\\train"

# Set the paths to the two output folders for left and right channels
left_folder = r"F:\\yhc\\ABCS\\ABCS_database_ciaic\\ABCS_database\\train\\air"
right_folder = r"F:\\yhc\\ABCS\\ABCS_database_ciaic\\ABCS_database\\train\\bone"

# Loop through all subfolders in the parent folder
for foldername, subfolders, filenames in os.walk(parent_folder):
    # Loop through all audio files in the subfolder
    for filename in filenames:
        if filename.endswith(".mp3") or filename.endswith(".wav"):
            # Set the input and output file paths
            input_file = os.path.join(foldername, filename)
            left_output_file = os.path.join(left_folder, filename)
            right_output_file = os.path.join(right_folder, filename)
            # Use soundfile to split the audio into left and right channels
            data, samplerate = sf.read(input_file)
            left_data = data[:, 0]
            right_data = data[:, 1]
            sf.write(left_output_file, left_data, samplerate)
            sf.write(right_output_file, right_data, samplerate)
