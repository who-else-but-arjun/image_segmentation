import os

# Define the directory where the images are located
folder_path = "pred"

# Iterate over all files in the directory
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        # Remove the '_leftImg8bit' part from the filename
        new_filename = filename.replace(".jpg", ".png")
        
        # Get the full paths of the old and new filenames
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)
        
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {old_file} -> {new_file}")

print("Renaming complete.")
