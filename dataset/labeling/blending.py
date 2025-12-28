import os
from PIL import Image
from tqdm import tqdm

# INPUT_ROOT = 'VFD_Bench_Dataset/Doll_kookie'
# INPUT_ROOT = 'VFD_Bench_Dataset/loveu-tgve-2023_Ours'
# INPUT_ROOT = 'VFD_Bench_Dataset/Pexels_kookie'
# INPUT_ROOT = 'VFD_Bench_Dataset/Pexels_younghwan'
# INPUT_ROOT = 'VFD_Bench_Dataset/TVR_512'
# INPUT_ROOT = 'VFD_Bench_Dataset/loveu-tgve-2023_Ours'
INPUT_ROOT = "dataset/VFD_Bench_Dataset"
dataset = ['Pexels_16610853_meerkat_head_rotation_2', 'TVR_Human_met_1_head_rotation']

def overlay_images():
    # dataset 리스트에 있는 폴더만 처리
    for folder_name in tqdm(dataset, desc="Processing datasets"):
        folder_path = os.path.join(INPUT_ROOT, folder_name)
        
        # 폴더가 존재하는지 확인
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping...")
            continue
        
        # 각 폴더 내의 모든 하위 디렉토리 순회
        for root, dirs, files in os.walk(folder_path):
            # 1. 대소문자 구분 없이 파일명 찾기
            files_lower = [f.lower() for f in files]
            input_name = None
            gt_name = None

            # input 파일명 찾기
            if 'input.png' in files_lower:
                input_name = files[files_lower.index('input.png')]
            elif 'input.PNG' in files:
                input_name = 'input.PNG'
                # 파일명 변경
                os.rename(os.path.join(root, input_name), os.path.join(root, 'input.png'))
                input_name = 'input.png'
            elif 'input.jpg' in files_lower:
                # 파일명 변경
                input_name = 'input.jpg'
                os.rename(os.path.join(root, input_name), os.path.join(root, 'input.png'))
                input_name = 'input.png'

            # gt 파일명 찾기
            if 'gt.png' in files_lower:
                gt_name = files[files_lower.index('gt.png')]
            elif 'gt.PNG' in files:
                gt_name = 'gt.PNG'
                # 파일명 변경
                os.rename(os.path.join(root, gt_name), os.path.join(root, 'gt.png'))
                gt_name = 'gt.png'
            elif 'gt.jpg' in files_lower:
                # 파일명 변경
                gt_name = 'gt.jpg'
                os.rename(os.path.join(root, gt_name), os.path.join(root, 'gt.png'))
                gt_name = 'gt.png'

            # if 'overlap.PNG' in files:
            #     overlap_name = 'overlap.PNG'
            #     # 파일명 변경
            #     os.rename(os.path.join(root, overlap_name), os.path.join(root, 'overlap.png'))
            #     overlap_name = 'overlap.png'

            # if 'overlap.png' exists, skip
            if 'overlap.png' in files:
                continue
            
            # 둘 다 있으면 blending
            if input_name and gt_name:
                input_path = os.path.join(root, input_name)
                gt_path = os.path.join(root, gt_name)
                output_path = os.path.join(root, 'overlap.png')

                # 이미지 열기 (RGBA로 변환하여 투명도 조정 가능하게 함)
                input_img = Image.open(input_path).convert('RGBA')
                gt_img = Image.open(gt_path).convert('RGBA')

                # 크기 맞추기 (같지 않으면 gt 이미지를 input과 동일 크기로 resize)
                if gt_img.size != input_img.size:
                    gt_img = gt_img.resize(input_img.size)

                # 두 이미지 반반 섞기
                blended = Image.blend(input_img, gt_img, alpha=0.5)
                
                # 저장
                blended.save(output_path)

if __name__ == '__main__':
    overlay_images()


