import os
import numpy as np
import cv2
import pydicom
import nibabel as nib
from xml.etree import ElementTree as ET
from tqdm import tqdm

# DICOM 파일 읽기
def load_dicom_images(dicom_dir):
    # DICOM 디렉토리에서 모든 .dcm 파일을 읽어서 slices 리스트에 저장
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    # 이미지의 z 축 위치를 기준으로 정렬
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # 각 slice의 픽셀 데이터를 스택으로 쌓아 3D 이미지 데이터 생성
    image_data = np.stack([s.pixel_array for s in slices])
    
    return image_data, slices

# XML 파일 파싱
def parse_xml(xml_file):
    # XML 파일을 파싱하여 트리 구조 생성
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # XML 구조에서 이미지와 ROI 정보를 담고 있는 배열을 찾음
    images_dict = root.find('dict')
    images_array = images_dict.find('array')
    
    rois = []
    
    for image_dict in images_array.findall('dict'):
        image_index = None
        rois_for_image = []
        
        for elem in image_dict:
            # 이미지 인덱스를 가져옴
            if elem.tag == 'key' and elem.text == 'ImageIndex':
                image_index = int(next(image_dict.iter('integer')).text)
                
            # 이미지에 포함된 ROI 정보를 가져옴
            if elem.tag == 'key' and elem.text == 'ROIs':
                rois_array = next(image_dict.iter('array'))
                
                for roi_dict in rois_array.findall('dict'):
                    points_mm = []
                    
                    for roi_elem in roi_dict:
                        # ROI의 좌표를 mm 단위로 가져옴
                        if roi_elem.tag == 'key' and roi_elem.text == 'Point_mm':
                            points_array = next(roi_dict.iter('array'))
                            points_mm = [tuple(map(float, p.text.strip('()').split(','))) for p in points_array.iter('string')]

                    rois_for_image.append(points_mm)
        
        # 인덱스와 ROI를 리스트에 추가
        if image_index is not None:
            rois.append((image_index, rois_for_image))
    
    return rois

# ROI 좌표를 DICOM 이미지의 픽셀 좌표로 변환
def convert_mm_to_pixel(points_mm, slices, image_data):
    pixel_coords = []
    
    for point in points_mm:
        point = np.array(point)
        # z 위치가 가장 가까운 slice를 찾음
        slice_index = min(range(len(slices)), key=lambda i: abs(slices[i].ImagePositionPatient[2] - point[2]))
        slice = slices[slice_index]
        
        # 이미지의 방향 및 위치, 픽셀 간격을 가져옴
        image_orientation = np.array(slice.ImageOrientationPatient, dtype=np.float64)
        image_position = np.array(slice.ImagePositionPatient, dtype=np.float64)
        pixel_spacing = np.array(slice.PixelSpacing, dtype=np.float64)
        
        # DICOM 이미지의 행과 열 방향 코사인 벡터
        row_cosine = image_orientation[0:3]
        col_cosine = image_orientation[3:6]
        
        # 물리 좌표를 픽셀 좌표로 변환
        row = np.dot(point - image_position, row_cosine) / pixel_spacing[1]
        col = np.dot(point - image_position, col_cosine) / pixel_spacing[0]
        
        # NIfTI 좌표계 변환 (좌표계 차이에 따른 변환 수행)
        row = image_data.shape[1] - 1 - row  # DICOM의 row 좌표를 NIfTI의 y축으로 변환
        col = image_data.shape[2] - 1 - col  # DICOM의 col 좌표를 NIfTI의 x축으로 변환

        # 변환된 픽셀 좌표를 리스트에 추가
        pixel_coords.append((slice_index, int(round(row)), int(round(col))))
    
    return pixel_coords

# NIfTI 파일 생성 및 저장 (개별 처리된 ROI들을 결합)
def create_label_nii(image_shape, all_roi_pixels, output_file, voxel_spacing, origin, image_orientation, fill=False):
    # 레이블 데이터를 위한 빈 3D 배열 생성
    label_data = np.zeros(image_shape, dtype=np.int16)
    
    for roi_pixels in all_roi_pixels:
        roi_label = np.zeros(image_shape, dtype=np.int16)
        
        # 각 ROI의 픽셀을 라벨 데이터에 추가
        for slice_index, row, col in roi_pixels:
            roi_label[slice_index, row, col] = 1

        # Morphology 연산 적용 (filling)
        if fill:
            for i in range(roi_label.shape[0]):
                roi_label[i] = cv2.morphologyEx(roi_label[i].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # 여러 ROI가 겹치지 않도록 최대값을 취하여 결합
        label_data = np.maximum(label_data, roi_label)

    # 레이블 데이터를 (row, col, slice) 순서로 변환
    label_data = np.transpose(label_data, (1, 2, 0))  # (slice, row, col) -> (row, col, slice)
    
    # DICOM의 좌표계를 반영한 affine matrix 생성
    row_cosine = np.array(image_orientation[:3], dtype=np.float64)
    col_cosine = np.array(image_orientation[3:], dtype=np.float64)
    slice_cosine = np.cross(row_cosine, col_cosine)
    
    affine = np.eye(4)
    affine[0:3, 0] = row_cosine * float(voxel_spacing[0])
    affine[0:3, 1] = col_cosine * float(voxel_spacing[1])
    affine[0:3, 2] = slice_cosine * float(voxel_spacing[2])
    affine[0:3, 3] = np.array(origin, dtype=np.float64)
    
    # NIfTI 이미지로 저장
    nii_img = nib.Nifti1Image(label_data, affine)
    nii_img.header.set_zooms(voxel_spacing)
    nib.save(nii_img, output_file)

if __name__ == '__main__':
    root_dir = './COCA/COCA_final'
    
    with tqdm(os.listdir(root_dir), desc="Processing patients") as pbar:
        for patient_id in pbar:
            pbar.set_postfix({"Current Patient": patient_id})
            
            patient_dir = os.path.join(root_dir, patient_id)
            
            if os.path.isdir(patient_dir):
                dicom_dir = os.path.join(patient_dir, 'dcm')
                xml_file = os.path.join(patient_dir, 'xml', f'{patient_id}.xml')
                nii_output_dir = os.path.join(patient_dir, 'nii')
                
                # DICOM 및 XML 파일이 존재하는지 확인
                if os.path.exists(dicom_dir) and os.path.exists(xml_file):
                    os.makedirs(nii_output_dir, exist_ok=True)
                    
                    # 1. DICOM 이미지 로드
                    image_data, slices = load_dicom_images(dicom_dir)

                    # 2. XML 파일 파싱
                    rois = parse_xml(xml_file)

                    # 3. ROI를 이미지 픽셀 좌표로 변환
                    all_roi_pixels = []
                    for image_index, rois_for_image in rois:
                        for roi_points_mm in rois_for_image:
                            roi_pixels = convert_mm_to_pixel(roi_points_mm, slices, image_data)
                            all_roi_pixels.append(roi_pixels)

                    # DICOM 파일의 voxel spacing 및 origin 가져오기
                    voxel_spacing = np.array([float(slices[0].PixelSpacing[0]), float(slices[0].PixelSpacing[1]), float(slices[1].SliceThickness)])
                    origin = np.array(slices[0].ImagePositionPatient, dtype=np.float64)
                    image_orientation = np.array(slices[0].ImageOrientationPatient, dtype=np.float64)
                    
                    # 4. NIfTI 라벨 파일 생성 및 저장 (두 가지 버전: fill O, fill X)
                    nii_output_path_filled = os.path.join(nii_output_dir, f'{patient_id}_label_filled.nii.gz')
                    nii_output_path_unfilled = os.path.join(nii_output_dir, f'{patient_id}_label_unfilled.nii.gz')
                    
                    create_label_nii(image_data.shape, all_roi_pixels, nii_output_path_filled, voxel_spacing, origin, image_orientation, fill=True)
                    create_label_nii(image_data.shape, all_roi_pixels, nii_output_path_unfilled, voxel_spacing, origin, image_orientation, fill=False)
                else:
                    print(f"Skipping patient: {patient_id} (Missing DICOM or XML files)")
                    
    print("XML to NIfTI Label 작업 완료!")