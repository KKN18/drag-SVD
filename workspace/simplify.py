import numpy as np
import os

# 파일 경로 설정
pixels_dir = '/home/nas2_userG/junhahyung/kkn/drag-SVD/pixels'
simple_pixels_dir = '/home/nas2_userG/junhahyung/kkn/drag-SVD/simple_pixels'

# simple_pixels 디렉터리가 없다면 생성
if not os.path.exists(simple_pixels_dir):
    os.makedirs(simple_pixels_dir)

# 극단점 및 중심점 찾는 함수 정의
def find_points(pixels):
    # 극단점 찾기
    top_left = pixels[np.argmin(pixels[:, 0] + pixels[:, 1])]
    bottom_right = pixels[np.argmax(pixels[:, 0] + pixels[:, 1])]
    top_right = pixels[np.argmax(pixels[:, 0] - pixels[:, 1])]
    bottom_left = pixels[np.argmin(pixels[:, 0] - pixels[:, 1])]
    
    # 중심점 찾기
    center = np.mean(pixels, axis=0)
    mid_top = [(top_left[0] + top_right[0]) / 2, (top_left[1] + top_right[1]) / 2]
    mid_bottom = [(bottom_left[0] + bottom_right[0]) / 2, (bottom_left[1] + bottom_right[1]) / 2]
    mid_left = [(top_left[0] + bottom_left[0]) / 2, (top_left[1] + bottom_left[1]) / 2]
    mid_right = [(top_right[0] + bottom_right[0]) / 2, (top_right[1] + bottom_right[1]) / 2]
    
    return np.array([top_left, top_right, bottom_left, bottom_right, center, mid_top, mid_bottom, mid_left, mid_right])

# 각 npy 파일 처리
for i in range(1, 8):  # target_0_1.npy부터 target_0_7.npy까지
    npy_path = os.path.join(pixels_dir, f'target_0_{i}.npy')
    pixels = np.load(npy_path)
    
    # image_0과 image_i의 극단점 및 중심점 찾기
    points_0 = find_points(pixels[:, :2])
    points_i = find_points(pixels[:, 2:])
    
    # (9, 4) 형태로 합치기
    combined_points = np.hstack((points_0, points_i))
    
    # simple_pixels/ 폴더에 저장
    output_path = os.path.join(simple_pixels_dir, f'target_0_{i}.npy')
    np.save(output_path, combined_points)
    print(f'Saved: {output_path}')