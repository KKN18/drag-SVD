import numpy as np
from PIL import Image, ImageDraw
import os

# 파일 경로 설정
pixels_dir = '/home/nas2_userG/junhahyung/kkn/drag-SVD/fixed_pixels/'
images_dir = '/home/nas2_userG/junhahyung/kkn/drag-SVD/workspace/images/'
output_dir = '/home/nas2_userG/junhahyung/kkn/drag-SVD/workspace/output/'

# output 디렉터리가 없다면 생성
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 이미지 인덱스에 따라 처리
for i in range(8):  # 0_image.png부터 7_image.png까지
    # 이미지 파일 불러오기
    image_path = os.path.join(images_dir, f'{i}_image.png')
    image = Image.open(image_path)

    # 해당하는 npy 파일 불러오기
    if i == 0:
        npy_path = os.path.join(pixels_dir, 'target_0_1.npy')  # 0_image.png는 target_0_1.npy 사용
    else:
        npy_path = os.path.join(pixels_dir, f'target_0_{i}.npy')  # 나머지 이미지는 target_0_{i}.npy 사용

    pixels = np.load(npy_path)

    # ImageDraw 객체 생성
    draw = ImageDraw.Draw(image)

    # 불러온 픽셀 위치에 빨간 점 찍기
    for pixel in pixels:
        if i == 0:
            x, y = pixel[0], pixel[1]  # 0_image.png는 pixel[0], pixel[1] 사용
        else:
            x, y = pixel[2], pixel[3]  # 나머지 이미지는 pixel[2], pixel[3] 사용
        width = 3
        draw.ellipse([(x-width, y-width), (x+width, y+width)], fill='red')

    # 결과 이미지 저장
    output_path = os.path.join(output_dir, f'{i}_image.png')
    image.save(output_path)
    print(f'Saved: {output_path}')
