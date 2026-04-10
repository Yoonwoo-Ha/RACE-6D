#!/usr/bin/env python3
"""
YCB-V 데이터셋의 모든 scene_gt.json 파일들을 처리하여 
obj_id 19, 20에 대해 paligned_values를 cam_R_m2c로 변환하여 cam_t_m2c에 더하는 스크립트
"""

import json
import numpy as np
from pathlib import Path

def process_scene_gt_file(scene_gt_path, paligned_values):
    """
    단일 scene_gt.json 파일 처리
    """
    # JSON 파일 읽기
    try:
        with open(scene_gt_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  ❌ JSON read error: {e}")
        return 0, 0
    
    modified_frames = 0
    modified_objects = 0
    
    # 각 프레임 처리
    for frame_id, objects in data.items():
        frame_modified = False
        
        for obj in objects:
            obj_id = obj['obj_id']
            
            # obj_id가 19 또는 20인 경우에만 처리
            if obj_id in paligned_values:
                # cam_R_m2c를 3x3 매트릭스로 변환 (9개 원소를 3x3으로)
                cam_R_m2c = np.array(obj['cam_R_m2c']).reshape(3, 3)
                
                # 원래 translation
                cam_t_m2c = np.array(obj['cam_t_m2c'])
                
                # paligned_values 변환
                paligned_vec = np.array(paligned_values[obj_id])
                
                # 새로운 translation = 원래 translation + rotation_matrix @ paligned_values
                new_cam_t_m2c = cam_t_m2c + cam_R_m2c @ paligned_vec
                
                # 첫 번째 수정 시에만 예시 출력
                if modified_objects == 0:
                    print(f"    Example obj_id {obj_id} transformation:")
                    print(f"      Original cam_t_m2c: {cam_t_m2c}")
                    print(f"      paligned_values[{obj_id}]: {paligned_vec}")
                    print(f"      cam_R_m2c @ paligned: {cam_R_m2c @ paligned_vec}")
                    print(f"      New cam_t_m2c: {new_cam_t_m2c}")
                
                # 변경 사항 적용
                obj['cam_t_m2c'] = new_cam_t_m2c.tolist()
                
                modified_objects += 1
                frame_modified = True
        
        if frame_modified:
            modified_frames += 1
    
    # 수정된 내용이 있으면 파일 저장
    if modified_objects > 0:
        try:
            # 백업 파일 생성 (없을 때만)
            backup_path = scene_gt_path.with_suffix('.json.backup_paligned')
            if not backup_path.exists():
                import shutil
                shutil.copy2(scene_gt_path, backup_path)
                print(f"  💾 Backup created: {backup_path.name}")
            
            # 수정된 파일 저장
            with open(scene_gt_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            return modified_frames, modified_objects
            
        except Exception as e:
            print(f"  ❌ Save error: {e}")
            return 0, 0
    
    return 0, 0

def process_all_folders(base_path):
    """
    base_path 내의 모든 서브폴더에서 scene_gt.json을 찾아 처리
    """
    base_path = Path(base_path)
    
    paligned_values = {
        19: [10.4796698, -5.41739619, -1.23077576],
        20: [-8.82785585, -10.93032056, 0.09932552]
    }
    
    print("=" * 80)
    print("Paligned Values Transformation for obj_id 19, 20")
    print("=" * 80)
    print(f"Base path: {base_path}")
    print(f"paligned_values:")
    print(f"  obj_id 19: {paligned_values[19]}")
    print(f"  obj_id 20: {paligned_values[20]}")
    print("=" * 80)
    print()
    
    total_folders = 0
    total_processed = 0
    total_modified_objects = 0
    folders_with_modifications = []
    
    # 모든 서브디렉토리 순회
    for folder in sorted(base_path.iterdir()):
        if folder.is_dir():
            scene_gt_path = folder / "scene_gt.json"
            
            if scene_gt_path.exists():
                total_folders += 1
                print(f"Folder: {folder.name}")
                
                modified_frames, modified_objects = process_scene_gt_file(
                    scene_gt_path, 
                    paligned_values
                )
                
                if modified_objects > 0:
                    print(f"  ✅ Modified: {modified_frames} frames, {modified_objects} objects")
                    total_processed += 1
                    total_modified_objects += modified_objects
                    folders_with_modifications.append(folder.name)
                else:
                    print(f"  ℹ️  No obj_id 19 or 20 found, skipped")
                
                print()
    
    # 최종 요약
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total folders checked: {total_folders}")
    print(f"Folders with modifications: {total_processed}")
    print(f"Total objects modified: {total_modified_objects}")
    
    if folders_with_modifications:
        print(f"\nModified folders:")
        for folder_name in folders_with_modifications:
            print(f"  - {folder_name}")
    
    print("\nBackup files: *.json.backup_paligned")
    print("=" * 80)
    
    return total_folders, total_processed, total_modified_objects

if __name__ == "__main__":
    dataset = "ycbv"
    base_folder = f"/home/rise/Downloads/{dataset}/test_real"
    
    # 또는 직접 경로 지정
    # base_folder = "/path/to/your/ycbv/test"
    
    process_all_folders(base_folder)