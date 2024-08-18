import os
import numpy as np
import cv2
import pydicom
import nibabel as nib
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
def convert_mm_to_pixel(points_mm, slices, image_data):
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
        
        # 기존 코드에서 했던 NIfTI 좌표계 변환
        row = image_data.shape[1] - 1 - row  # DICOM의 row 좌표를 NIfTI의 y축으로 변환
        col = image_data.shape[2] - 1 - col  # DICOM의 col 좌표를 NIfTI의 x축으로 변환

        pixel_coords.append((slice_index, int(round(row)), int(round(col))))
    
    return pixel_coords

# NIfTI 파일 생성 및 저장 (개별 처리된 ROI들을 결합)
def create_label_nii(image_shape, all_roi_pixels, output_file, voxel_spacing, fill=False):
    label_data = np.zeros(image_shape, dtype=np.int16)
    
    for roi_pixels in all_roi_pixels:
        roi_label = np.zeros(image_shape, dtype=np.int16)
        for slice_index, row, col in roi_pixels:
            roi_label[slice_index, row, col] = 1

        if fill:
            for i in range(roi_label.shape[0]):
                roi_label[i] = cv2.morphologyEx(roi_label[i].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # ROI가 겹치지 않도록 현재 레이블에 병합
        label_data = np.maximum(label_data, roi_label)

    label_data = np.transpose(label_data, (1, 2, 0))  # (slice, row, col) -> (row, col, slice)
    
    # NIfTI의 좌표계에 맞게 반전된 방향성을 반영
    nii_img = nib.Nifti1Image(label_data, np.eye(4))
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

                    # DICOM 파일의 voxel spacing 가져오기
                    voxel_spacing = (slices[0].PixelSpacing[0], slices[0].PixelSpacing[1], slices[1].SliceThickness)
                    
                    # 4. NIfTI 라벨 파일 생성 및 저장 (두 가지 버전: fill O, fill X)
                    nii_output_path_filled = os.path.join(nii_output_dir, f'{patient_id}_label_filled.nii.gz')
                    nii_output_path_unfilled = os.path.join(nii_output_dir, f'{patient_id}_label_unfilled.nii.gz')
                    
                    create_label_nii(image_data.shape, all_roi_pixels, nii_output_path_filled, voxel_spacing, fill=True)
                    create_label_nii(image_data.shape, all_roi_pixels, nii_output_path_unfilled, voxel_spacing, fill=False)
                else:
                    print(f"Skipping patient: {patient_id} (Missing DICOM or XML files)")
