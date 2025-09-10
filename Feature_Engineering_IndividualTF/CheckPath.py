import os

INPUT_DIR = "/Users/lukesalter/Library/CloudStorage/GoogleDrive-luke.salter111@gmail.com/My Drive/Machine_Learning/Project7_data/Data_Medium/Data_Medium_Resampled"

print(os.listdir(INPUT_DIR))  # list all files in the folder

file_path = os.path.join(INPUT_DIR, "Data_1m.csv")
print("File exists:", os.path.exists(file_path))
