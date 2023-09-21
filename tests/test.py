import glob

folders = glob.glob('dataset/tennis_*')
for ind,folder_name in enumerate(folders):
    print(f"{ind+1}/{len(folders)} {folder_name}")