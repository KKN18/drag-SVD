import numpy as np
import os

# A Directory 경로
a_directory = '/home/nas2_userG/junhahyung/kkn/LucidDreamer/local/data/pixels'  # A Directory의 실제 경로로 변경하세요.

# B Directory 경로
b_directory = '/home/nas2_userG/junhahyung/kkn/drag-SVD/pixels'  # B Directory의 실제 경로로 변경하세요.

# B Directory 생성 (이미 존재하지 않는 경우)
if not os.path.exists(b_directory):
    os.makedirs(b_directory)

# combined_0.npy 불러오기
combined_0_path = os.path.join(a_directory, 'combined_0.npy')
combined_0 = np.load(combined_0_path)

# combined_0과 나머지 combined_i 파일들을 일대일 대응시키기
for i in range(1, 8):  # combined_1.npy부터 combined_7.npy까지
    combined_i_path = os.path.join(a_directory, f'combined_{i}.npy')
    combined_i = np.load(combined_i_path)
    
    # 결과 저장을 위한 배열 초기화
    # 각 행은 [pixel_coords_view0_x, pixel_coords_view0_y, pixel_coords_viewi_x, pixel_coords_viewi_y]
    combined_correspondence = np.zeros((combined_0.shape[0], 4))
    
    # 일대일 대응
    for index in range(combined_0.shape[0]):
        # 뷰 0에서의 픽셀 좌표
        pixel_coords_view0 = combined_0[index, 3:5]  # 2차원 좌표라 원소 2개 추출
        
        # 뷰 i에서의 픽셀 좌표
        pixel_coords_viewi = combined_i[index, 3:5]  # 2차원 좌표라 원소 2개 추출
        
        # 결과 배열에 저장
        combined_correspondence[index, :2] = pixel_coords_view0
        combined_correspondence[index, 2:] = pixel_coords_viewi
    
    # B Directory에 저장
    output_path = os.path.join(b_directory, f'target_0_{i}.npy')
    np.save(output_path, combined_correspondence)
    print(f'Saved: {output_path}')
