#!/usr/bin/env python3
"""
카메라 intrinsic 매개변수가 다른 경우 poses를 표준 cam_K에 맞게 변환하는 스크립트
3D 점군 데이터를 활용한 정밀한 pose 변환 (반복적 정제 적용)
"""

import os
import json
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from tqdm import tqdm


def load_camera_matrix(cam_k_list):
    """cam_K 리스트를 3x3 매트릭스로 변환"""
    return np.array(cam_k_list).reshape(3, 3)


def load_points_from_ply(file_path, max_points=1000):
    """PLY 파일에서 3D 점군 로딩"""
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        
        if len(points) == 0:
            print(f"⚠️  PLY 파일에 점이 없습니다: {file_path}")
            return None
        
        # BOP 데이터셋은 mm 단위 (문서 확인됨)
        # 일관성을 위해 항상 m 단위로 변환
        max_coord = np.max(np.abs(points))
        points = points / 1000.0  # mm -> m (BOP 표준)
        print(f"    📏 단위 변환: mm → m (max coord: {max_coord:.1f}mm → {max_coord/1000.0:.3f}m)")
        
        if len(points) <= max_points:
            return points
        else:
            # 균등한 서브샘플링
            indices = np.round(np.linspace(0, len(points) - 1, max_points)).astype(int)
            subsampled_points = points[indices]
            return subsampled_points
            
    except Exception as e:
        print(f"❌ PLY 파일 로딩 실패 {file_path}: {e}")
        return None


def load_3d_models(models_dir):
    """모든 객체의 3D 모델 로딩"""
    print(f"🔄 3D 모델들을 로딩 중: {models_dir}")
    models_path = Path(models_dir)
    points_3d = {}
    
    # obj_XXXXXX.ply 파일들 찾기
    ply_files = list(models_path.glob("obj_*.ply"))
    
    for ply_file in tqdm(ply_files, desc="3D 모델 로딩"):
        # obj_000001.ply -> obj_id = 1
        obj_name = ply_file.stem  # obj_000001
        obj_id = int(obj_name.split('_')[1])  # 000001 -> 1
        
        points = load_points_from_ply(str(ply_file))
        if points is not None:
            points_3d[obj_id] = points
            print(f"  📁 obj_{obj_id:06d}: {len(points)} 점")
    
    print(f"✅ 총 {len(points_3d)}개 객체의 3D 모델 로딩 완료")
    return points_3d


def project_3d_to_2d_with_depth(points_3d, cam_K):
    """3D 점들을 2D로 투영 (카메라 좌표계 점들)"""
    # 카메라 좌표계에서 이미지 평면으로 투영
    points_2d_homo = (cam_K @ points_3d.T).T
    
    # 깊이 값 저장
    depths = points_2d_homo[:, 2]
    
    # 동차좌표에서 일반좌표로 변환
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    
    return points_2d, depths


def select_robust_3d_points(points_3d_obj, obj_R_m2c, obj_t_m2c, cam_K, 
                           img_width=640, img_height=480, grid_size=80):
    """기하학적으로 잘 분포된 안정적인 3D 점들 선택"""
    
    # 객체 좌표계에서 카메라 좌표계로 변환
    points_3d_cam = (obj_R_m2c @ points_3d_obj.T).T + obj_t_m2c
    
    # 모든 점들 사용 (깊이 필터링 제거)
    return points_3d_obj, np.arange(len(points_3d_obj))


def refine_pose_iteratively(points_world, points_2d, cam_K, R_init, t_init, 
                           max_iterations=5, convergence_threshold=0.1):
    """반복적으로 pose를 정제하여 재투영 오차 최소화"""
    
    R_current = R_init.copy()
    t_current = t_init.copy()
    prev_error = float('inf')
    
    print(f"    🔄 반복적 정제 시작...")
    
    for iteration in range(max_iterations):
        # 현재 pose로 재투영
        points_cam = (R_current @ points_world.T).T + t_current
        points_2d_proj, depths = project_3d_to_2d_with_depth(points_cam, cam_K)
        
        # 재투영 오차 계산 (깊이 필터링 제거)
        errors = np.linalg.norm(points_2d_proj - points_2d, axis=1)
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        
        print(f"    반복 {iteration+1}: 평균 오차 = {mean_error:.2f} pixels, " +
              f"최대 오차 = {max_error:.2f} pixels")
        
        # 수렴 확인
        if abs(prev_error - mean_error) < convergence_threshold:
            print(f"    ✅ 수렴 달성 (변화량 < {convergence_threshold})")
            break
        
        # 오차가 증가하면 중단
        if mean_error > prev_error:
            print(f"    ⚠️  오차 증가, 이전 결과 사용")
            break
        
        prev_error = mean_error
        
        # Adaptive threshold: 오차 분포에 따라 동적으로 결정
        # RANSAC 0.5 픽셀 기준에 맞춰 더 엄격하게 조정
        threshold = min(0.8 * np.median(errors), np.percentile(errors, 75))
        good_mask = errors < threshold
        
        if np.sum(good_mask) < 6:
            print(f"    ⚠️  반복 {iteration+1}: 좋은 점 부족")
            break
        
        # 좋은 점들로만 다시 PnP
        rvec, _ = cv2.Rodrigues(R_current)
        
        # Levenberg-Marquardt 최적화 사용
        success, rvec_new, tvec_new = cv2.solvePnP(
            points_world[good_mask].astype(np.float32),
            points_2d[good_mask].astype(np.float32),
            cam_K.astype(np.float32),
            None,
            rvec=rvec,
            tvec=t_current.reshape(3, 1),
            useExtrinsicGuess=False,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success:
            R_new, _ = cv2.Rodrigues(rvec_new)
            t_new = tvec_new.flatten()
            
            # 새 pose가 합리적인지 확인 (큰 변화 방지)
            rotation_change = np.linalg.norm(R_new - R_current, 'fro')
            translation_change = np.linalg.norm(t_new - t_current)
            
            if rotation_change > 0.5 or translation_change > 0.5:
                print(f"    ⚠️  반복 {iteration+1}: 변화량이 너무 큼, 건너뜀")
                continue
            
            R_current = R_new
            t_current = t_new
            
            print(f"    ✅ 반복 {iteration+1}: 사용된 점 = {np.sum(good_mask)}개")
        else:
            print(f"    ⚠️  반복 {iteration+1}: solvePnP 실패")
            break
    
    return R_current, t_current


def collect_3d_2d_correspondences(points_3d_models, obj_poses, old_cam_k, 
                                 img_width=640, img_height=480):
    """모든 visible 객체의 3D-2D 대응점 수집 (카메라 좌표계 점들 반환)"""
    all_points_cam = []
    all_points_2d = []
    
    for obj_info in obj_poses:
        obj_id = obj_info['obj_id']
        if obj_id not in points_3d_models:
            print(f"    ⚠️  객체 {obj_id}의 3D 모델이 없습니다")
            continue
        
        # 객체 pose 추출
        obj_R_m2c = np.array(obj_info['cam_R_m2c']).reshape(3, 3)
        obj_t_m2c = np.array(obj_info['cam_t_m2c'])
        
        # BOP 데이터셋 translation은 mm 단위 (문서 확인됨)
        obj_t_m2c = obj_t_m2c / 1000.0  # mm → m (BOP 표준)
        
        # PLY 점들을 현재 pose로 카메라 좌표계에 변환
        ply_points = points_3d_models[obj_id]
        points_3d_cam = (obj_R_m2c @ ply_points.T).T + obj_t_m2c
        
        # 2D 투영 (old_cam_k 사용)
        points_2d, depths = project_3d_to_2d_with_depth(points_3d_cam, old_cam_k)
        
        # 양의 깊이를 가진 점들만 사용
        valid_mask = depths > 0
        valid_points_cam = points_3d_cam[valid_mask]
        valid_points_2d = points_2d[valid_mask]
        
        if len(valid_points_cam) > 0:
            all_points_cam.extend(valid_points_cam)  # 카메라 좌표계 점들!
            all_points_2d.extend(valid_points_2d)
            print(f"    ✅ 객체 {obj_id}: {len(valid_points_cam)}개 점 추가")
    
    return np.array(all_points_cam), np.array(all_points_2d)


def convert_pose_with_3d_points_fixed(points_3d_models, obj_poses, 
                                     old_cam_k, new_cam_k, 
                                     cam_R_w2c, cam_t_w2c):
    """완전히 단순화된 객체 pose 변환 (PLY 점들 직접 사용)"""
    
    print("  🔍 각 객체별 pose 변환 시작...")
    
    # 각 객체별로 개별적으로 pose 변환
    for obj_info in obj_poses:
        obj_id = obj_info['obj_id']
        if obj_id not in points_3d_models:
            print(f"    ⚠️  객체 {obj_id}의 3D 모델이 없습니다")
            continue
        
        print(f"    📦 객체 {obj_id} 처리 중...")
        
        # 기존 객체 pose
        old_obj_R_m2c = np.array(obj_info['cam_R_m2c']).reshape(3, 3)
        old_obj_t_m2c = np.array(obj_info['cam_t_m2c']) / 1000.0  # mm → m
        
        # PLY 점들 (원본 그대로!)
        ply_points = points_3d_models[obj_id]
        
        # 1. PLY 점들을 old_pose로 변환 → old_cam_k로 투영 → 2D 점들
        points_3d_cam = (old_obj_R_m2c @ ply_points.T).T + old_obj_t_m2c
        points_2d, depths = project_3d_to_2d_with_depth(points_3d_cam, old_cam_k)
        
        # 유효한 점들만 사용
        valid_mask = depths > 0
        valid_ply_points = ply_points[valid_mask]
        valid_2d_points = points_2d[valid_mask]
        
        if len(valid_ply_points) < 6:
            print(f"      ⚠️  객체 {obj_id}: 유효한 점이 부족 ({len(valid_ply_points)}개)")
            continue
        
        # 2. solvePnP로 new_cam_k에 맞는 new_pose 계산
        try:
            success, rvec, tvec = cv2.solvePnP(
                objectPoints=valid_ply_points.astype(np.float32),  # PLY 원본 점들!
                imagePoints=valid_2d_points.astype(np.float32),   # 타겟 2D 점들
                cameraMatrix=new_cam_k.astype(np.float32),        # 새 카메라 매트릭스
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # 새로운 객체 pose
                new_obj_R_m2c, _ = cv2.Rodrigues(rvec)
                new_obj_t_m2c = tvec.flatten()
                
                # 검증: 재투영 오차 확인
                verify_points = (new_obj_R_m2c @ valid_ply_points.T).T + new_obj_t_m2c
                verify_2d, _ = project_3d_to_2d_with_depth(verify_points, new_cam_k)
                errors = np.linalg.norm(verify_2d - valid_2d_points, axis=1)
                mean_error = np.mean(errors)
                
                print(f"      ✅ 객체 {obj_id}: 평균 오차 {mean_error:.2f} pixels")
                
                # 원본 데이터 업데이트 (mm 단위로 변환)
                obj_info['cam_R_m2c'] = new_obj_R_m2c.flatten().tolist()
                obj_info['cam_t_m2c'] = (new_obj_t_m2c * 1000.0).tolist()  # m → mm
                
            else:
                print(f"      ❌ 객체 {obj_id}: solvePnP 실패")
                
        except Exception as e:
            print(f"      ❌ 객체 {obj_id}: 예외 발생 - {e}")
    
    print("  ✅ 모든 객체 pose 변환 완료")
    
    # 카메라 pose는 그대로 유지 (객체 pose만 변경)
    return cam_R_w2c, cam_t_w2c


def verify_pose_result(points_world, points_2d_target, cam_K, R_w2c, t_w2c):
    """Pose 변환 결과 검증"""
    # 월드 → 카메라 변환
    points_cam = (R_w2c @ points_world.T).T + t_w2c
    
    # 깊이 정보 (참고용)
    depths = points_cam[:, 2]
    
    # 2D 재투영
    points_2d_proj, _ = project_3d_to_2d_with_depth(points_cam, cam_K)
    
    # 재투영 오차
    reprojection_errors = np.linalg.norm(points_2d_proj - points_2d_target, axis=1)
    mean_error = np.mean(reprojection_errors)
    median_error = np.median(reprojection_errors)
    
    print(f"  📏 최종 검증 결과:")
    print(f"     평균 재투영 오차: {mean_error:.2f} pixels")
    print(f"     중앙값 재투영 오차: {median_error:.2f} pixels")
    print(f"     최대 재투영 오차: {np.max(reprojection_errors):.2f} pixels")
    print(f"     < 0.5 pixels: {np.sum(reprojection_errors < 0.5)}/{len(reprojection_errors)} " +
          f"({100*np.sum(reprojection_errors < 0.5)/len(reprojection_errors):.1f}%)")
    print(f"     깊이 범위: [{np.min(depths):.3f}, {np.max(depths):.3f}] m")


def convert_poses_for_camera_matrix(old_cam_k, new_cam_k, cam_R_w2c, cam_t_w2c, 
                                   scene_gt_frame=None, points_3d_models=None):
    """
    다른 카메라 매트릭스에 맞게 poses를 변환
    """
    # 카메라 매트릭스가 같으면 변환 불필요
    if np.allclose(old_cam_k, new_cam_k, rtol=1e-5):
        return cam_R_w2c, cam_t_w2c
    
    print(f"🔄 카메라 매트릭스 변환 수행:")
    print(f"   원본 cam_K: [{old_cam_k[0,0]:.3f}, {old_cam_k[1,1]:.3f}, " +
          f"{old_cam_k[0,2]:.3f}, {old_cam_k[1,2]:.3f}]")
    print(f"   목표 cam_K: [{new_cam_k[0,0]:.3f}, {new_cam_k[1,1]:.3f}, " +
          f"{new_cam_k[0,2]:.3f}, {new_cam_k[1,2]:.3f}]")
    
    # BOP 데이터셋 translation은 mm 단위 - m 단위로 변환
    cam_t_w2c_m = cam_t_w2c / 1000.0  # mm → m (BOP 표준)
    t_magnitude = np.linalg.norm(cam_t_w2c_m)
    print(f"   현재 translation 크기: {np.linalg.norm(cam_t_w2c):.1f}mm → {t_magnitude:.3f}m")
    
    # 3D 점군이 있고 객체 정보가 있으면 정밀한 변환 수행
    if (points_3d_models is not None and 
        scene_gt_frame is not None and 
        len(scene_gt_frame) > 0):
        
        print("  🎯 3D 점군 기반 정밀 변환 시도...")
        
        # 3D 점군 기반 pose 변환 (m 단위 translation 사용)
        new_R, new_t = convert_pose_with_3d_points_fixed(
            points_3d_models, scene_gt_frame, 
            old_cam_k, new_cam_k, cam_R_w2c, cam_t_w2c_m
        )
        
        if new_R is not None:
            print("  ✅ 3D 점군 기반 변환 성공")
            # 결과를 다시 mm 단위로 변환해서 반환 (BOP 표준)
            new_t_mm = new_t * 1000.0
            return new_R, new_t_mm
        else:
            print("  ❌ 3D 점군 기반 변환 실패 - PnP 사용 불가")
            return None, None
    else:
        print("  ❌ 3D 모델 또는 객체 정보 없음 - PnP 사용 불가")
        return None, None


def load_scene_gt(scene_gt_path):
    """scene_gt.json 파일에서 객체별 pose 정보 로딩"""
    try:
        with open(scene_gt_path, 'r') as f:
            scene_gt_data = json.load(f)
        return scene_gt_data
    except Exception as e:
        print(f"⚠️  scene_gt.json 로딩 실패: {e}")
        return {}


def process_scene_camera_file(scene_camera_path, target_cam_k, 
                             scene_gt_data=None, points_3d_models=None):
    """scene_camera.json 파일에서 poses를 변환"""
    
    target_cam_k_array = np.array(target_cam_k).reshape(3, 3)
    
    with open(scene_camera_path, 'r') as f:
        scene_data = json.load(f)
    
    modified = False
    scene_gt_modified = False  # scene_gt.json 변경 여부 추적
    successful_frames = 0
    
    for frame_id, frame_data in scene_data.items():
        current_cam_k = np.array(frame_data['cam_K']).reshape(3, 3)
        
        # 카메라 매트릭스가 다른 경우에만 변환
        if not np.allclose(current_cam_k, target_cam_k_array):
            print(f"\n  🖼️  프레임 {frame_id}: cam_K 변환 필요")
            
            # 회전과 변환 매트릭스 추출
            cam_R_w2c = np.array(frame_data['cam_R_w2c']).reshape(3, 3)
            cam_t_w2c = np.array(frame_data['cam_t_w2c'])
            
            # 해당 프레임의 객체 정보 가져오기
            scene_gt_frame = None
            if scene_gt_data is not None and frame_id in scene_gt_data:
                scene_gt_frame = scene_gt_data[frame_id]
                print(f"     발견된 객체: {len(scene_gt_frame)}개")
            
            # poses 변환
            new_R, new_t = convert_poses_for_camera_matrix(
                current_cam_k, target_cam_k_array, cam_R_w2c, cam_t_w2c,
                scene_gt_frame=scene_gt_frame, points_3d_models=points_3d_models
            )
            
            if new_R is not None and new_t is not None:
                # 데이터 업데이트
                frame_data['cam_K'] = target_cam_k
                frame_data['cam_R_w2c'] = new_R.flatten().tolist()
                frame_data['cam_t_w2c'] = new_t.tolist()
                modified = True
                successful_frames += 1
                
                # 객체 정보가 있었다면 scene_gt도 변경됨
                if scene_gt_frame is not None:
                    scene_gt_modified = True
                    
            else:
                print(f"     ❌ 프레임 {frame_id} 변환 실패 - 프로그램 종료")
                print(f"💀 PnP 변환 실패로 인해 프로그램을 종료합니다.")
                exit(1)
    
    if modified:
        print(f"\n  📊 변환 통계: 성공 {successful_frames}개")
        if scene_gt_modified:
            print(f"  📦 객체 pose도 변경되었습니다.")
    
    return scene_data, modified, scene_gt_modified


def fix_camera_matrices(base_dir, target_cam_k, models_dir="ycbv2coco_train/models"):
    """
    지정된 디렉토리의 모든 scene_camera.json 파일에서 cam_K를 수정
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ 경로가 존재하지 않습니다: {base_dir}")
        return
    
    print(f"🔍 {base_dir}에서 scene_camera.json 파일들을 처리 중...")
    print(f"목표 cam_K: {target_cam_k}")
    
    # 3D 점군 모델 로딩
    points_3d_models = None
    if Path(models_dir).exists():
        print(f"\n📦 3D 점군 모델 로딩 시작...")
        points_3d_models = load_3d_models(models_dir)
        if len(points_3d_models) > 0:
            print(f"✅ {len(points_3d_models)}개 3D 모델 로딩 완료")
        else:
            print("⚠️  3D 모델을 찾을 수 없습니다. 기본 변환만 사용됩니다.")
            points_3d_models = None
    else:
        print(f"⚠️  3D 모델 디렉토리가 없습니다: {models_dir}")
        print("📐 기본 변환만 사용됩니다.")
    
    # 모든 시나리오 폴더 찾기
    scene_folders = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    # # 테스트를 위해 특정 시나리오만 처리
    # # 전체 처리하려면 이 부분을 주석처리
    # test_scenes = ["000048", "000049", "000050", "000051", "000052", 
    #                "000053", "000054", "000055", "000056", "000057",
    #                "000058", "000059", "000060"]  # 다른 cam_K를 가진 시나리오들
    # scene_folders = [f for f in scene_folders if f.name in test_scenes]
    # print(f"🧪 테스트 모드: {len(scene_folders)}개 시나리오만 처리")
    
    total_modified = 0
    total_processed = 0
    
    for scene_folder in tqdm(scene_folders, desc="시나리오 처리"):
        print(f"\n📁 처리 중: {scene_folder.name}")
        
        scene_camera_file = scene_folder / "scene_camera.json"
        scene_gt_file = scene_folder / "scene_gt.json"
        
        if not scene_camera_file.exists():
            print(f"⚠️  {scene_camera_file} 파일이 없습니다.")
            continue
        
        try:
            # scene_gt.json 로딩 (객체 정보)
            scene_gt_data = None
            if scene_gt_file.exists():
                scene_gt_data = load_scene_gt(scene_gt_file)
                if scene_gt_data:
                    print(f"  📋 scene_gt.json 로딩 완료")
            
            # scene_camera.json 처리
            scene_data, modified, scene_gt_modified = process_scene_camera_file(
                scene_camera_file, target_cam_k, 
                scene_gt_data=scene_gt_data, 
                points_3d_models=points_3d_models
            )
            
            if modified:
                # 백업 생성
                backup_file = scene_camera_file.with_suffix('.json.backup')
                if not backup_file.exists():
                    import shutil
                    shutil.copy(scene_camera_file, backup_file)
                    print(f"  📁 백업 생성: {backup_file}")
                
                # 수정된 scene_camera.json 저장
                with open(scene_camera_file, 'w') as f:
                    json.dump(scene_data, f, indent=2)
                
                # scene_gt.json도 저장 (객체 pose가 변경되었으므로)
                if scene_gt_data is not None and scene_gt_modified:
                    # scene_gt.json 백업 생성
                    backup_gt_file = scene_gt_file.with_suffix('.json.backup')
                    if not backup_gt_file.exists():
                        shutil.copy(scene_gt_file, backup_gt_file)
                        print(f"  📁 scene_gt.json 백업 생성: {backup_gt_file}")
                    
                    # 수정된 scene_gt.json 저장
                    with open(scene_gt_file, 'w') as f:
                        json.dump(scene_gt_data, f, indent=2)
                    print(f"  ✅ scene_gt.json 저장 완료")
                elif scene_gt_data is not None and not scene_gt_modified:
                    print(f"  ℹ️  scene_gt.json 변경 없음 - 저장 생략")
                
                total_modified += 1
                print(f"  ✅ {scene_folder.name}: cam_K 수정 완료")
            else:
                print(f"  ℹ️  {scene_folder.name}: 수정 불필요 (이미 목표 cam_K와 일치)")
            
            total_processed += 1
            
        except Exception as e:
            print(f"❌ {scene_folder.name} 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ 처리 완료:")
    print(f"   총 처리된 시나리오: {total_processed}")
    print(f"   수정된 시나리오: {total_modified}")
    print(f"   변경 없는 시나리오: {total_processed - total_modified}")
    
    if points_3d_models is not None and len(points_3d_models) > 0:
        print(f"   🎯 3D 점군 기반 정밀 변환 사용됨")


def check_camera_matrices(base_dir, full_check=False):
    """디렉토리의 모든 cam_K 값을 확인"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"❌ 경로가 존재하지 않습니다: {base_dir}")
        return None, {}
    
    print(f"🔍 {base_dir}의 cam_K 값들을 확인 중...")
    
    scene_folders = sorted([d for d in base_path.iterdir() if d.is_dir()])
    unique_cam_k = set()
    cam_k_to_scenes = {}
    
    check_folders = scene_folders if full_check else scene_folders[:10]
    desc = "전체 확인" if full_check else "샘플 확인"
    
    for scene_folder in tqdm(check_folders, desc=desc):
        scene_camera_file = scene_folder / "scene_camera.json"
        
        if scene_camera_file.exists():
            try:
                with open(scene_camera_file, 'r') as f:
                    scene_data = json.load(f)
                
                # 첫 번째 프레임의 cam_K 확인
                first_frame = next(iter(scene_data.values()))
                cam_k_tuple = tuple(first_frame['cam_K'])
                unique_cam_k.add(cam_k_tuple)
                
                # 시나리오별 cam_K 매핑
                if cam_k_tuple not in cam_k_to_scenes:
                    cam_k_to_scenes[cam_k_tuple] = []
                cam_k_to_scenes[cam_k_tuple].append(scene_folder.name)
                
            except Exception as e:
                print(f"⚠️  {scene_folder.name} 읽기 오류: {e}")
    
    print(f"\n발견된 고유한 cam_K 값들:")
    for i, cam_k in enumerate(unique_cam_k, 1):
        scenes = cam_k_to_scenes[cam_k]
        cam_k_array = np.array(cam_k).reshape(3, 3)
        print(f"  {i}. fx={cam_k_array[0,0]:.1f}, fy={cam_k_array[1,1]:.1f}, " +
              f"cx={cam_k_array[0,2]:.1f}, cy={cam_k_array[1,2]:.1f}")
        print(f"     사용 시나리오: {len(scenes)}개 " +
              f"({', '.join(scenes[:5])}{'...' if len(scenes) > 5 else ''})")
    
    if len(unique_cam_k) == 1:
        print("✅ 모든 시나리오가 동일한 cam_K를 사용합니다.")
    else:
        print(f"⚠️  {len(unique_cam_k)}개의 서로 다른 cam_K가 발견되었습니다.")
    
    return unique_cam_k, cam_k_to_scenes


def choose_target_cam_k(unique_cam_k, cam_k_to_scenes):
    """사용자가 목표 cam_K를 선택하도록 함"""
    if len(unique_cam_k) <= 1:
        return list(unique_cam_k)[0] if unique_cam_k else None
    
    print(f"\n목표로 할 cam_K를 선택하세요:")
    cam_k_list = list(unique_cam_k)
    
    for i, cam_k in enumerate(cam_k_list, 1):
        cam_k_array = np.array(cam_k).reshape(3, 3)
        num_scenes = len(cam_k_to_scenes[cam_k])
        print(f"  {i}. fx={cam_k_array[0,0]:.1f}, fy={cam_k_array[1,1]:.1f}, " +
              f"cx={cam_k_array[0,2]:.1f}, cy={cam_k_array[1,2]:.1f} " +
              f"({num_scenes}개 시나리오)")
    
    while True:
        try:
            choice = input(f"\n선택 (1-{len(cam_k_list)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(cam_k_list):
                return list(cam_k_list[idx])
            else:
                print(f"1에서 {len(cam_k_list)} 사이의 숫자를 입력하세요.")
        except ValueError:
            print("올바른 숫자를 입력하세요.")


if __name__ == "__main__":
    # 처리할 디렉토리
    BASE_DIR = "ycbv/train_real"
    MODELS_DIR = "ycbv/models"  # 3D 점군 모델 디렉토리
    
    print("=" * 60)
    print("3D 점군 기반 정밀 카메라 매트릭스 수정 도구")
    print("(반복적 정제 알고리즘 적용)")
    print("=" * 60)
    
    print(f"📁 처리 대상 디렉토리: {BASE_DIR}")
    print(f"📦 3D 모델 디렉토리: {MODELS_DIR}")
    
    # 디렉토리 존재 확인
    if not Path(BASE_DIR).exists():
        print(f"❌ 처리 대상 디렉토리가 없습니다: {BASE_DIR}")
        exit(1)
    
    if not Path(MODELS_DIR).exists():
        print(f"⚠️  3D 모델 디렉토리가 없습니다: {MODELS_DIR}")
        print("📐 기본 변환만 사용됩니다.")
    else:
        print(f"✅ 3D 모델 디렉토리 확인됨")
    
    print("\n" + "=" * 60)
    
    # 전체 샘플 확인
    unique_cam_k, cam_k_to_scenes = check_camera_matrices(BASE_DIR, full_check=True)
    
    if not unique_cam_k:
        print("❌ cam_K 값을 찾을 수 없습니다.")
        exit(1)
    
    if len(unique_cam_k) == 1:
        print("✅ 모든 시나리오가 동일한 cam_K를 사용합니다. 변환이 불필요합니다.")
        exit(0)
    
    print("\n" + "=" * 60)
    
    # 목표 cam_K 선택
    target_cam_k = choose_target_cam_k(unique_cam_k, cam_k_to_scenes)
    
    if target_cam_k is None:
        print("❌ 목표 cam_K가 선택되지 않았습니다.")
        exit(1)
    
    print(f"\n선택된 목표 cam_K: {target_cam_k}")
    
    # 사용자 확인
    response = input(f"\n{BASE_DIR}의 cam_K를 위 값으로 통일하시겠습니까? (y/N): ")
    
    if response.lower() == 'y':
        fix_camera_matrices(BASE_DIR, target_cam_k, models_dir=MODELS_DIR)
    else:
        print("작업이 취소되었습니다.")