import numpy as np
import os

# B Directory 경로
b_directory = "/home/nas2_userG/junhahyung/kkn/drag-SVD/pixels"  # B Directory의 실제 경로로 변경하세요.

# C Directory 경로 - 여기에 간소화된 결과를 저장할 것입니다.
c_directory = "/home/nas2_userG/junhahyung/kkn/drag-SVD/simple_pixels"  # C Directory의 실제 경로로 변경하세요.

# C Directory 생성 (이미 존재하지 않는 경우)
if not os.path.exists(c_directory):
    os.makedirs(c_directory)

# B Directory 내의 모든 .npy 파일을 순회
for i in range(1, 8):  # combined_1.npy부터 combined_7.npy까지
    file_path = os.path.join(b_directory, f'target_0_{i}.npy')
    
    # 파일이 존재하면 처리
    if os.path.exists(file_path):
        data = np.load(file_path)
        
        # 데이터에서 무작위로 2개의 픽셀 쌍을 선택
        if data.shape[0] >= 2:
            chosen_indices = np.random.choice(data.shape[0], 2, replace=False)
        else:
            chosen_indices = np.arange(data.shape[0])
        
        simplified_data = data[chosen_indices]
        
        # C Directory에 저장
        output_path = os.path.join(c_directory, f'target_0_{i}.npy')
        np.save(output_path, simplified_data)
        print(f'Simplified data saved: {output_path}')