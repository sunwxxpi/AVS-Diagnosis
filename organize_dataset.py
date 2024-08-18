import os
import shutil
import numpy as np
import pydicom
import nibabel as nib
from tqdm import tqdm

def dicom_to_nifti(dicom_dir, output_path):
    # DICOM 파일들이 저장된 디렉토리의 모든 파일을 가져옴
    dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
    
    # DICOM 파일 로드 및 정렬
    dicoms = []
    
    for f in dicom_files:
        dcm = pydicom.dcmread(f)
        
        if hasattr(dcm, 'ImagePositionPatient') or hasattr(dcm, 'PixelSpacing'):
            dicoms.append(dcm)
        else:
            print(f"Warning: {f} does not have 'ImagePositionPatient' or 'PixelSpacing' attribute and will be skipped.")
    
    # 정렬할 DICOM 파일이 없다면 함수 종료
    if not dicoms:
        print(f"No valid DICOM files found in {dicom_dir}. Skipping NIfTI conversion.")
        return
    
    dicoms.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # z 좌표를 기준으로 정렬
    
    # DICOM 이미지 크기와 스페이싱 정보
    image_shape = list(dicoms[0].pixel_array.shape)
    image_shape.append(len(dicoms))  # z-axis 추가
    
    # Z 축의 스페이싱 계산
    if len(dicoms) > 1:
        z_spacing = dicoms[1].ImagePositionPatient[2] - dicoms[0].ImagePositionPatient[2]
        
        if z_spacing != 3.0:
            print(f"Z spacing of {dicom_dir}: {z_spacing}")
        if z_spacing == 0 or np.isclose(z_spacing, 0):
            print(f"Warning: Z spacing {z_spacing} is zero or close to zero for {output_path}. Using default value of 3.")
            z_spacing = 3.0
    else:
        z_spacing = 3.0  # 기본값 (안전 장치)
    
    voxel_spacing = (dicoms[0].PixelSpacing[0], dicoms[0].PixelSpacing[1], z_spacing)
    
    # 3D 볼륨 생성
    volume = np.zeros(image_shape, dtype=dicoms[0].pixel_array.dtype)
    
    # 각 DICOM 슬라이스를 볼륨에 쌓을 때, 270도 회전 후 axial 단면의 상하 반전
    for i, dcm in enumerate(dicoms):
        axial_slice = np.flip(np.rot90(dcm.pixel_array, k=3), axis=0)
        volume[:, :, i] = axial_slice
    
    # NIfTI 파일 생성
    affine = np.eye(4)
    affine[0, 0] = voxel_spacing[0]
    affine[1, 1] = voxel_spacing[1]
    affine[2, 2] = voxel_spacing[2]

    nifti_img = nib.Nifti1Image(volume, affine)
    nib.save(nifti_img, output_path)
    
if __name__ == "__main__":
    base_dir = "./COCA/Gated_release_final"
    calcium_xml_dir = os.path.join(base_dir, "calcium_xml")
    patient_dir = os.path.join(base_dir, "patient")
    dataset_final_dir = os.path.join("COCA", "COCA_final")

    os.makedirs(dataset_final_dir, exist_ok=True)

    with tqdm(os.listdir(calcium_xml_dir), desc="Processing patients") as pbar:
        for xml_file in pbar:
            if xml_file.endswith(".xml"):
                pbar.set_postfix({"Current File": xml_file})
                
                # 파일 이름에서 환자 번호 추출
                patient_number = os.path.splitext(xml_file)[0]
                
                # 새로운 환자 디렉토리 생성
                patient_final_dir = os.path.join(dataset_final_dir, patient_number)
                os.makedirs(patient_final_dir, exist_ok=True)
                
                dcm_dir = os.path.join(patient_final_dir, "dcm")
                xml_dir = os.path.join(patient_final_dir, "xml")
                nii_dir = os.path.join(patient_final_dir, "nii")
                
                os.makedirs(dcm_dir, exist_ok=True)
                os.makedirs(xml_dir, exist_ok=True)
                os.makedirs(nii_dir, exist_ok=True)
                
                shutil.copy(os.path.join(calcium_xml_dir, xml_file), xml_dir)
                
                # DCM 파일들 복사 및 NIfTI 변환
                patient_dcm_dir = os.path.join(patient_dir, patient_number)
                if os.path.exists(patient_dcm_dir):
                    dicom_found = False
                    
                    for root, dirs, files in os.walk(patient_dcm_dir):
                        for dcm_file in files:
                            if dcm_file.endswith(".dcm"):
                                dicom_found = True
                                shutil.copy(os.path.join(root, dcm_file), dcm_dir)
                    
                    if dicom_found:
                        # DICOM을 NIfTI로 변환하여 nii 디렉토리에 저장
                        nifti_output_path = os.path.join(nii_dir, f"{patient_number}.nii.gz")
                        dicom_to_nifti(dcm_dir, nifti_output_path)
                    else:
                        print(f"No DICOM files found for patient {patient_number} in {patient_dcm_dir}")

    print("모든 작업이 완료되었습니다!")