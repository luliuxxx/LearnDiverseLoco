import glob
from pathlib import Path

data_path = str(Path(__file__).resolve().parents[1] / "data")
txt_files = glob.glob(data_path + '/*.txt')
stxt_files = sorted(txt_files)

with open('motion_names.txt', 'w') as f:
  for txt_file in stxt_files:
    file_name = txt_file.split('/')[-1]
    name_strs = txt_file.split('/')[-1].split("_")
    motion_type = name_strs[0]
    motion_idx = name_strs[-1].split(".")[0]
    f.write(f'["{file_name[:-4]}" ,')
    f.write(f'"{file_name}" ,')
    f.write('None, None],\n')
