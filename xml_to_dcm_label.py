import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from xml.etree import ElementTree as ET
from tqdm import tqdm

# DICOM 파일 읽기
def load_dicom_images(dicom_dir):
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    image_data = np.stack([s.pixel_array for s in slices])
    
    return image_data, slices

# XML 파일 파싱
def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    images_dict = root.find('dict')
    images_array = images_dict.find('array')
    
    rois = []
    
    for image_dict in images_array.findall('dict'):
        image_index = None
        rois_for_image = []
        
        for elem in image_dict:
            if elem.tag == 'key' and elem.text == 'ImageIndex':
                image_index = int(next(image_dict.iter('integer')).text)
                
            if elem.tag == 'key' and elem.text == 'ROIs':
                rois_array = next(image_dict.iter('array'))
                
                for roi_dict in rois_array.findall('dict'):
                    points_mm = []
                    
                    for roi_elem in roi_dict:
                        if roi_elem.tag == 'key' and roi_elem.text == 'Point_mm':
                            points_array = next(roi_dict.iter('array'))
                            points_mm = [tuple(map(float, p.text.strip('()').split(','))) for p in points_array.iter('string')]

                    rois_for_image.append(points_mm)
        
        if image_index is not None:
            rois.append((image_index, rois_for_image))
    
    return rois

# ROI 좌표를 DICOM 이미지의 좌표로 변환
def convert_mm_to_pixel(points_mm, slices):
    pixel_coords = []
    
    for point in points_mm:
        point = np.array(point)
        slice_index = min(range(len(slices)), key=lambda i: abs(slices[i].ImagePositionPatient[2] - point[2]))
        slice = slices[slice_index]
        
        image_orientation = np.array(slice.ImageOrientationPatient)
        image_position = np.array(slice.ImagePositionPatient)
        pixel_spacing = np.array(slice.PixelSpacing)
        
        row_cosine = image_orientation[0:3]
        col_cosine = image_orientation[3:6]
        
        row = np.dot(point - image_position, row_cosine) / pixel_spacing[1]
        col = np.dot(point - image_position, col_cosine) / pixel_spacing[0]
        pixel_coords.append((slice_index, int(round(row)), int(round(col))))
    
    return pixel_coords

# DICOM 이미지와 라벨을 겹쳐서 PNG 파일로 저장
def save_slices_with_labels(image_data, roi_pixels, dicom_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    total_slices = image_data.shape[0]
    
    for slice_index in range(total_slices):
        # 현재 슬라이스 이미지 가져오기
        slice_img = image_data[slice_index]
        
        # ROI 픽셀 좌표 중 현재 슬라이스에 해당하는 것만 필터링
        label_overlay = np.zeros_like(slice_img, dtype=np.uint8)
        for si, row, col in roi_pixels:
            if si == slice_index:
                label_overlay[row, col] = 255  # 라벨 부분을 흰색으로 표시
                
        # 이미지와 라벨 모두 좌우 반전
        slice_img = np.fliplr(slice_img)
        label_overlay = np.transpose(np.fliplr(label_overlay)[::-1, ::-1])
        
        # DICOM 파일 이름을 기반으로 저장 파일 이름 생성
        dicom_filename = os.path.basename(dicom_files[slice_index])
        dicom_basename, dicom_ext = os.path.splitext(dicom_filename)
        
        # 이미지와 라벨 겹치기
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(slice_img, cmap='gray')
        ax.imshow(label_overlay, cmap='Reds', alpha=0.5)  # 라벨을 빨간색으로 표시, 투명하게 겹치기
        ax.axis('off')
        
        # 슬라이스를 PNG로 저장
        output_file = os.path.join(output_dir, f'{dicom_basename}-{slice_index:03d}.png')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == '__main__':
    root_dir = './COCA/COCA_final'
    
    with tqdm(os.listdir(root_dir), desc="Processing dirs") as pbar:
        for dir_name in pbar:
            pbar.set_postfix({"Current Dir": dir_name})
            
            dir_path = os.path.join(root_dir, dir_name)
            
            if os.path.isdir(dir_path):
                dicom_dir = os.path.join(dir_path, 'dcm')
                xml_file = os.path.join(dir_path, 'xml', f'{dir_name}.xml')
                output_dir = dicom_dir
                
                if os.path.exists(dicom_dir) and os.path.exists(xml_file):
                    # 1. DICOM 이미지 로드
                    image_data, slices = load_dicom_images(dicom_dir)

                    # 2. XML 파일 파싱
                    rois = parse_xml(xml_file)

                    # 3. ROI를 이미지 픽셀 좌표로 변환
                    all_roi_pixels = []
                    for image_index, rois_for_image in rois:
                        for roi_points_mm in rois_for_image:
                            roi_pixels = convert_mm_to_pixel(roi_points_mm, slices)
                            all_roi_pixels.extend(roi_pixels)

                    # 4. 슬라이스별로 DICOM 이미지와 라벨을 겹쳐서 저장
                    save_slices_with_labels(image_data, all_roi_pixels, [s.filename for s in slices], output_dir)
                else:
                    print(f"Skipping directory: {dir_name} (Missing DICOM or XML files)")