#!/usr/bin/env python3
"""
YCB-V obj_id 19, 20의 PLY 모델을 paligned_values만큼 이동
"""

import numpy as np
import trimesh
from pathlib import Path
import shutil

def transform_ply(ply_path, translation):
    """
    PLY 파일을 읽어서 translation만큼 이동하고 저장
    
    Args:
        ply_path: PLY 파일 경로
        translation: [x, y, z] 이동 벡터
    """
    try:
        # PLY 로드
        mesh = trimesh.load(str(ply_path))
        
        print(f"  Original mesh:")
        print(f"    Vertices: {len(mesh.vertices)}")
        print(f"    Bounds: {mesh.bounds}")
        print(f"    Centroid: {mesh.centroid}")
        
        # Translation 적용 (원점을 대칭 중심으로 이동: v_new = v - p)
        transform_matrix = np.eye(4)
        transform_matrix[:3, 3] = -translation
        mesh.apply_transform(transform_matrix)
        
        print(f"  After translation by {translation}:")
        print(f"    Bounds: {mesh.bounds}")
        print(f"    Centroid: {mesh.centroid}")
        
        return mesh
        
    except Exception as e:
        print(f"  ❌ Error loading PLY: {e}")
        return None

def process_ply_models(models_path):
    """
    obj_id 19, 20의 PLY 모델을 paligned_values만큼 이동
    """
    models_path = Path(models_path)
    
    paligned_values = {
        19: np.array([10.4796698, -5.41739619, -1.23077576]),
        20: np.array([-8.82785585, -10.93032056, 0.09932552])
    }
    
    print("=" * 80)
    print("PLY Model Transformation for obj_id 19, 20")
    print("=" * 80)
    print(f"Models path: {models_path}")
    print(f"Transformations:")
    print(f"  obj_id 19: translation by {paligned_values[19]}")
    print(f"  obj_id 20: translation by {paligned_values[20]}")
    print("=" * 80)
    print()
    
    for obj_id, translation in paligned_values.items():
        ply_filename = f"obj_{obj_id:06d}.ply"
        ply_path = models_path / ply_filename
        backup_path = models_path / f"obj_{obj_id:06d}.ply.backup_paligned"
        
        if not ply_path.exists():
            print(f"⚠️  {ply_filename} not found, skipping")
            continue
        
        print(f"Processing: {ply_filename}")
        
        # 백업 생성 (없을 때만)
        if not backup_path.exists():
            shutil.copy2(ply_path, backup_path)
            print(f"  💾 Backup created: {backup_path.name}")
        else:
            print(f"  ℹ️  Backup already exists: {backup_path.name}")
        
        # PLY 변환
        transformed_mesh = transform_ply(ply_path, translation)
        
        if transformed_mesh is not None:
            # 변환된 PLY 저장
            try:
                transformed_mesh.export(str(ply_path))
                print(f"  ✅ Saved transformed PLY: {ply_filename}")
            except Exception as e:
                print(f"  ❌ Error saving PLY: {e}")
        
        print()
    
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print("Transformed models:")
    print(f"  - obj_000019.ply")
    print(f"  - obj_000020.ply")
    print("\nBackup files: obj_*.ply.backup_paligned")
    print("\n⚠️  Important: This transformation should be done BEFORE using the models")
    print("   for training or evaluation to maintain consistency with scene_gt.json")
    print("=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="YCB-V obj_id 19, 20의 PLY 모델을 paligned_values만큼 이동")
    parser.add_argument("models_path", help="모델 PLY 파일들이 있는 폴더 경로 (예: ycbv/models)")
    args = parser.parse_args()

    process_ply_models(args.models_path)