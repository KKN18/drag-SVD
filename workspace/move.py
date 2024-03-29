import numpy as np
import os

# 원본 및 저장할 폴더 경로 설정
source_dir = '/home/nas2_userG/junhahyung/kkn/drag-SVD/simple_pixels/'
destination_dir = '/home/nas2_userG/junhahyung/kkn/drag-SVD/mvright_pixels/'

# 저장할 폴더가 없다면 생성
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# i = 1부터 7까지 반복
for i in range(1, 8):
    # 원본 파일 경로
    source_path = os.path.join(source_dir, f'target_0_{i}.npy')
    
    # 원본 픽셀 데이터 로드
    pixels = np.load(source_path)
    
    # pixel[2] = pixel[0] + i*5, pixel[3] = pixel[1] 적용
    modified_pixels = pixels.copy()
    modified_pixels[:, 2] = pixels[:, 0] + i * 5
    modified_pixels[:, 3] = pixels[:, 1]

    # 수정된 픽셀 데이터를 새 위치에 저장
    destination_path = os.path.join(destination_dir, f'target_0_{i}.npy')
    np.save(destination_path, modified_pixels)
    print(f'Saved: {destination_path}')