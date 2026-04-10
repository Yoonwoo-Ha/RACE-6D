import numpy as np
import argparse
import random
from pathlib import Path
import open3d as o3d
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

import torch
# torch.set_grad_enabled(False);
import matplotlib.patches as patches

from src.misc.box_ops import box_xyxy_to_mxmywh, box_cxcywh_to_mxmywh
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

# COCO classes
CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
    '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
]

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

cam_K = torch.tensor([1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0]).reshape(3, 3)

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def rescale_poses(out_pose, scale = 1000):
    scale_factor = scale
    poses = out_pose * torch.tensor([scale_factor, scale_factor, scale_factor], dtype=torch.float32)
    return poses

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def plot_bboxes(clses, image, bboxes):
    plt.figure(figsize=(16,10))
    plt.imshow(image)
    ax = plt.gca()
    colors = COLORS * 100

    for cls, bbox, c in zip(clses, bboxes, colors):
        # bbox는 [x_min, y_min, width, height] 형식이어야 합니다.
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   fill=False, color=c, linewidth=3))
        text = f'{CLASSES[cls]}'
        ax.text(x_min, y_min, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
        # rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        # ax.add_patch(rect)

    plt.axis('off')
    plt.show()

def plot_bboxes(ax, target):
    colors = COLORS * 100
    bboxes = target['boxes']

    for bbox, c in zip(bboxes, colors):
        # bbox는 [x_min, y_min, width, height] 형식이어야 합니다.
        if ax.get_title() == 'Original Image':
            bbox = box_xyxy_to_mxmywh(bbox)
        elif ax.get_title() == 'Rotated Image':
            bbox = box_cxcywh_to_mxmywh(bbox * torch.tensor([640, 480, 640, 480], dtype=torch.float32))
        x_min, y_min, width, height = bbox
        ax.add_patch(plt.Rectangle((x_min, y_min), width, height, edgecolor=c, linewidth=3, facecolor='white', fill=False))

def plot_results_comparison_bbox(original_img, rotated_img, original_target, rotated_target, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot original image
    ax1.imshow(original_img)
    ax1.set_title("Original Image")
    plot_bboxes(ax1, original_target)
    ax1.axis('off')
    ax1.set_aspect('equal')
    
    # Plot rotated image
    ax2.imshow(rotated_img)
    ax2.set_title("Rotated Image")
    plot_bboxes(ax2, rotated_target)
    ax2.axis('off')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()  # 메모리 절약을 위해 figure 닫기
    else:
        plt.show()

def plot_results_comparison_points(original_img, rotated_img, points_original, original_target, points_transformed, rotated_target, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot original image
    ax1.imshow(original_img)
    ax1.set_title("Original Image")
    plot_points(ax1, points_original, original_target)
    ax1.axis('off')
    ax1.set_aspect('equal')
    
    # Plot rotated image
    ax2.imshow(rotated_img)
    ax2.set_title("Rotated Image")
    plot_points(ax2, points_transformed, rotated_target)
    ax2.axis('off')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
        plt.close()  # 메모리 절약을 위해 figure 닫기
    else:
        plt.show()

def plot_bboxes_only(image, bboxes, condition=None, fill=False, save=False):
    plt.figure(figsize=(16,10))
    dpi = 100
    # fig_width = 640 / dpi  # 인치 단위로 계산
    # fig_height = 480 / dpi
    # plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    plt.imshow(image)
    plt.axis('off')
    # plt.gca().set_position([0, 0, 1, 1])
    ax = plt.gca()
    colors = COLORS * 100

    for bbox, c in zip(bboxes, colors):
        # bbox는 [x_min, y_min, width, height] 형식이어야 합니다.
        if condition is not None:
            if condition == 'xywh':
                x_min, y_min, width, height = bbox
            elif condition == 'xyxy':
                bbox = box_xyxy_to_mxmywh(bbox * torch.tensor([640, 480, 640, 480], dtype=torch.float32))
                x_min, y_min, width, height = bbox
            elif condition == 'cxcywh':
                bbox = box_cxcywh_to_mxmywh(bbox * torch.tensor([640, 480, 640, 480], dtype=torch.float32))
                x_min, y_min, width, height = bbox
        else:
            x_min, y_min, width, height = bbox
        if fill == True:
            ax.add_patch(plt.Rectangle((x_min, y_min), width, height, edgecolor=c, linewidth=0, facecolor='white', fill=True))
        else:
            ax.add_patch(plt.Rectangle((x_min, y_min), width, height, edgecolor=c, linewidth=4, facecolor='white', fill=False))

    if save:
        plt.savefig('/home/yoonwoo/Downloads/output.png',dpi=dpi, pad_inches=0)
    plt.show()

def plot_results_with_points(pil_img, prob, points, translation_matrix, rotation_matrix, camera_intrinsics):
    plt.figure(figsize=(8, 6))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    cls = []

    for p, rot, tran, col in zip(prob, rotation_matrix, translation_matrix, colors):
        cl = p.argmax()
        cl = int(cl)
        cls.append(cl)
        points_cl = points[cl]
        translation_matrix_np = tran.numpy()
        rotation_matrix_np = rot.numpy()

        # Apply rotation and translation to the axes
        points_3d_transformed = np.dot(rotation_matrix_np, points_cl.T).T + translation_matrix_np

        # Project the 3D points onto the 2D image plane
        points_2D = project_to_image_plane(points_3d_transformed, camera_intrinsics)

        plt.scatter(points_2D[:, 0], points_2D[:, 1], color=col, s=2)

    plt.axis('off')
    plt.show()

    return cls

def plot_results_with_points_anno(pil_img, prob, points, translation_matrix, rotation_matrix, camera_intrinsics):
    plt.figure(figsize=(8, 5))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100

    for p, rot, tran, col in zip(prob, rotation_matrix, translation_matrix, colors):
        cl = int(p)
        points_cl = points[cl]
        translation_matrix_np = tran.numpy()
        rotation_matrix_np = rot.numpy()

        # Apply rotation and translation to the axes
        points_3d_transformed = np.dot(rotation_matrix_np, points_cl.T).T + translation_matrix_np

        # Project the 3D points onto the 2D image plane
        points_2D = project_to_image_plane(points_3d_transformed, camera_intrinsics)

        plt.scatter(points_2D[:, 0], points_2D[:, 1], color=col, s=2)

    plt.axis('off')
    plt.show()

def _c2t(translation, cam_K, bbox_info=None):
        """
        벡터화된 정규화된 상대 좌표를 3D 공간 좌표로 변환
        ConvertPose의 역변환을 수행
        translation: rx, ry, rz (정규화된 상대 좌표)
        cam_K: fx, fy, px, py
        bbox_info: [cx, cy, w, h] (정규화된 좌표 [0,1] 범위)
        """
        # 입력이 1D인 경우 2D로 확장
        if translation.dim() == 1:
            translation = translation.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        rx, ry, rz = translation[:, 0], translation[:, 1], translation[:, 2]
        
        # 카메라 내부 파라미터 (벡터화)
        fx = cam_K[0, 0]
        fy = cam_K[1, 1]
        px = cam_K[0, 2]
        py = cam_K[1, 2]
        
        if bbox_info.dim() == 1:
            bbox_info = bbox_info.unsqueeze(0)

        # bbox_info가 정규화된 좌표 [0,1] 범위인지 확인하고 픽셀 좌표로 변환
        if bbox_info.max() <= 1.0:
            # 정규화된 [cx, cy, w, h] 좌표를 픽셀 좌표로 변환
            cxbbox = bbox_info[:, 0] * 640  # cx
            cybbox = bbox_info[:, 1] * 480  # cy
            wbbox = bbox_info[:, 2] * 640   # w
            hbbox = bbox_info[:, 3] * 480   # h
        else:
            # 이미 픽셀 좌표인 [cx, cy, w, h] 형태
            cxbbox = bbox_info[:, 0]
            cybbox = bbox_info[:, 1]
            wbbox = bbox_info[:, 2]
            hbbox = bbox_info[:, 3]

        tz = rz * 1000.0
        
        # ✅ 올바른 역변환 공식
        tx = ((rx * wbbox + cxbbox - px) * tz) / fx
        ty = ((ry * hbbox + cybbox - py) * tz) / fy
        
        result = torch.stack([tx, ty, tz], dim=1)
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result

def plot_points(ax, points, target):
    colors = COLORS * 100
    cam_K_0 = cam_K
    
    if ax.get_title() == 'Original Image':
        # ===============================================
        # Original Image: 변환 전 데이터
        # - Boxes: [x1, y1, x2, y2] (픽셀 좌표)
        # - Poses: [tx, ty, tz, R...] (절대 좌표)
        # ===============================================
        
        if target['poses'].shape[1] == 12:  # [tx, ty, tz, R(3x3)]
            rotation_matrix = target['poses'][:, 3:].reshape(-1, 3, 3)
            translation_matrix = target['poses'][:, :3]
        else:
            print(f"Unexpected pose format for Original Image: {target['poses'].shape}")
            return
            
    elif ax.get_title() == 'Rotated Image':
        # ===============================================
        # Rotated Image: 변환 후 데이터
        # - Boxes: [cx, cy, w, h] (정규화된 좌표)
        # - Poses: [rx, ry, rz, R6D] (상대 좌표)
        # ===============================================
        
        if target['poses'].shape[1] == 9:  # [rx, ry, rz, R6D(6)]
            rotation_matrix = rotation_6d_to_matrix(target['poses'][:, 3:]).reshape(-1, 3, 3)
            # 정규화된 상대 좌표를 절대 좌표로 변환
            translation_matrix = _c2t(target['poses'][:, :3], cam_K_0, target['boxes'])
        elif target['poses'].shape[1] == 12:  # [rx, ry, rz, R(3x3)]
            rotation_matrix = target['poses'][:, 3:].reshape(-1, 3, 3)
            # 정규화된 상대 좌표를 절대 좌표로 변환
            translation_matrix = _c2t(target['poses'][:, :3], cam_K_0, target['boxes'])
        else:
            print(f"Unexpected pose format for Rotated Image: {target['poses'].shape}")
            return
    else:
        print(f"Unknown image title: {ax.get_title()}")
        return
    
    classes = target['labels']
    
    for cl, rot, tran, col in zip(classes, rotation_matrix, translation_matrix, colors):
        cl = int(cl)
        
        # 클래스 키가 존재하는지 확인
        if cl not in points:
            print(f"Warning: Class {cl} not found in points dictionary")
            continue
            
        translation_matrix_np = tran.detach().cpu().numpy()
        rotation_matrix_np = rot.detach().cpu().numpy()
        points_cl = points[cl]
        
        # 3D 포인트 변환
        points_3d_transformed = np.dot(rotation_matrix_np, points_cl.T).T + translation_matrix_np
        
        # 2D 투영
        cam_K_0_np = np.array(cam_K_0.detach().cpu().numpy())
        points_2D = project_to_image_plane(points_3d_transformed, cam_K_0_np)
        
        # 화면 범위 내의 점들만 표시
        valid_points = (points_2D[:, 0] >= 0) & (points_2D[:, 0] < 640) & \
                      (points_2D[:, 1] >= 0) & (points_2D[:, 1] < 480)
        
        if valid_points.any():
            ax.scatter(points_2D[valid_points, 0], points_2D[valid_points, 1], color=col, s=2)
        else:
            print(f"Class {cl}: No valid points projected (all out of bounds)")
            # 첫 몇 개 점의 좌표를 출력해서 디버깅
            print(f"  Sample 2D points: {points_2D[:3]}")
            print(f"  Translation: {translation_matrix_np}")
            print(f"  3D transformed sample: {points_3d_transformed[:3]}")

def load_points_from_ply(file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        return points
        
def load_points_from_ply_s(file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points, dtype=np.float32)
        if len(points) < 1500:
            return points
        else:
            # np.random.seed(42)
            indices = np.random.choice(len(points), 1500, replace=False)
            subsampled_points = points[indices]
            return subsampled_points

# Function to project 3D points to 2D image plane
def project_to_image_plane(points_3D, camera_intrinsics):
    points_2D = np.dot(camera_intrinsics, points_3D.T).T
    points_2D /= points_2D[:, 2:3]
    return points_2D[:, :2]

# Define the function to plot results
def plot_results_with_target(pil_img, points, target):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100

    translation_matrix = target['poses'][:, :3]
    # rotation이 R6D 포맷인 경우
    if target['poses'].shape[1] == 9:  # translation(3) + R6D(6)
        rotation_matrix = rotation_6d_to_matrix(target['poses'][:, 3:9])
    else:  # 기존 rotation matrix 포맷
        rotation_matrix = target['poses'][:, 3:].reshape(-1, 3, 3)
    classes = target['labels']
    cam_K = target['cam_K']

    for cl, rot, tran, camK, col in zip(classes, rotation_matrix, translation_matrix, cam_K, colors):

        translation_matrix_np = tran.numpy()
        rotation_matrix_np = rot.numpy()
        cl = int(cl)
        points_cl = points[cl]
        
        # Apply rotation and translation to the axes
        points_3d_transformed = np.dot(rotation_matrix_np, points_cl.T).T + translation_matrix_np

        # Project the 3D points onto the 2D image plane
        camK = np.array(camK).reshape(3, 3)
        points_2D = project_to_image_plane(points_3d_transformed, camK)

        plt.scatter(points_2D[:, 0], points_2D[:, 1], color=col, s=2)

    plt.axis('off')
    plt.show()