import os

base_dir = 'Pexels_kookie'

# base_dir 내 하위 폴더 목록 가져오기
subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

# 검사할 파일 리스트
required_files = ['points_flowdrag.json', 'points.json']

missing_info = []

for folder in subfolders:
    for filename in required_files:
        filepath = os.path.join(folder, filename)
        if not os.path.isfile(filepath):
            missing_info.append((folder, filename))

if missing_info:
    for folder, filename in missing_info:
        print(f"Missing '{filename}' in folder '{folder}'")
else:
    print("All required files are present in all subfolders.")
