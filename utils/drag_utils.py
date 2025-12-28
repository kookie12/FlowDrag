# *************************************************************************
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *************************************************************************


import torch
from typing import List
import pdb


def calculate_l1_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Calculate the L1 (Manhattan) distance between two tensors."""
    return torch.sum(torch.abs(tensor1 - tensor2), dim=1)


def calculate_l2_distance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Calculate the L2 (Euclidean) distance between two tensors."""
    return torch.sqrt(torch.sum((tensor1 - tensor2) ** 2, dim=1))


def calculate_cosine_similarity(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    """Calculate the Cosine Similarity between two tensors."""
    numerator = torch.sum(tensor1 * tensor2, dim=1)
    denominator = torch.sqrt(torch.sum(tensor1 ** 2, dim=1)) * torch.sqrt(torch.sum(tensor2 ** 2, dim=1))
    return numerator / denominator


def get_neighboring_patch(tensor: torch.Tensor, center: tuple, radius: int) -> torch.Tensor:
    """Get a neighboring patch from a tensor centered at a specific point."""
    r1, r2 = int(center[0]) - radius, int(center[0]) + radius + 1
    c1, c2 = int(center[1]) - radius, int(center[1]) + radius + 1
    return tensor[:, :, r1:r2, c1:c2]


def update_handle_points(handle_points: torch.Tensor, all_dist: torch.Tensor, r2) -> torch.Tensor:
    """Update handle points based on computed distances."""
    try:
        row, col = divmod(all_dist.argmin().item(), all_dist.shape[-1]) # all_dist=[25,25], argmin()은 작은 값이 아니라, 작은 값이 있는 index를 반환
    except:
        print("check") # debug by kookie
        pdb.set_trace()
    updated_point = torch.tensor([
        handle_points[0] - r2 + row,
        handle_points[1] - r2 + col
    ])
    return updated_point

#F0.shape = torch.Size([1, 64, 256, 256]), f0.shape = torch.Size([1, 64]) => F0는 original feature
def point_tracking(F0: torch.Tensor, F1: torch.Tensor, handle_points: List[torch.Tensor],
                   handle_points_init: List[torch.Tensor], r2, distance_type: str = 'l1') -> List[torch.Tensor]:
    """Track points between F0 and F1 tensors."""
    H, W = F1.shape[2], F1.shape[3]

    with torch.no_grad():
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]

            # Check boundary before tracking - skip if out of bounds
            if (int(pi[0]) - r2 < 0 or int(pi[0]) + r2 + 1 > H or
                int(pi[1]) - r2 < 0 or int(pi[1]) + r2 + 1 > W):
                # Skip tracking, keep current position
                continue

            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]
            f0_expanded = f0.unsqueeze(dim=-1).unsqueeze(dim=-1) #[1, 640, 1, 1]

            F1_neighbor = get_neighboring_patch(F1, pi, r2) #r2=12, F1는 updated되는 feature, F1_neighbor.shape = torch.Size([1, 64, 25, 25])

            # Switch case for different distance functions
            if distance_type == 'l1':
                all_dist = calculate_l1_distance(f0_expanded, F1_neighbor) #all_dist.shape=torch.Size([1, 25, 25])
            elif distance_type == 'l2':
                all_dist = calculate_l2_distance(f0_expanded, F1_neighbor)
            elif distance_type == 'cosine':
                all_dist = -calculate_cosine_similarity(f0_expanded, F1_neighbor)  # Negative for minimization

            all_dist = all_dist.squeeze(dim=0) #[25,25]
            handle_points[i] = update_handle_points(pi, all_dist, r2)

    return handle_points

#NOTE: 
def point_tracking_kookie(F0: torch.Tensor, F1: torch.Tensor, handle_points: List[torch.Tensor],
                   handle_points_init: List[torch.Tensor], r2, vector_field: torch.Tensor = None, 
                   interp_mask: torch.Tensor = None, distance_type: str = 'l1',) -> List[torch.Tensor]:
    """Track points between F0 and F1 tensors with vector field application."""
    
    with torch.no_grad():        
        U = vector_field[:, :, 0]
        V = vector_field[:, :, 1]
        mask_indices = interp_mask.nonzero(as_tuple=True)
        U_masked = U[mask_indices]
        V_masked = V[mask_indices]
        # mask 혹은 F0 shape 보고 지금 u, v shape 낮춰야함
        vector_magnitudes = torch.sqrt(U_masked**2 + V_masked**2)  # 벡터 크기 계산
        mean_magnitude = torch.mean(vector_magnitudes)  # 벡터 크기의 평균값 계산
        
        for i in range(len(handle_points)):
            pi0, pi = handle_points_init[i], handle_points[i]

            # Apply vector field to the point if inside the mask
            if interp_mask is not None and interp_mask[int(pi[0]), int(pi[1])] > 0:
                di = vector_field[int(pi[0]), int(pi[1])]
                di_norm = di.norm()
                 
                if di_norm > mean_magnitude:  # 평균 이상 벡터 크기만 고려
                    pi += di  # Adjust the point based on the vector field

            # Proceed with original point tracking logic
            f0 = F0[:, :, int(pi0[0]), int(pi0[1])]
            f0_expanded = f0.unsqueeze(dim=-1).unsqueeze(dim=-1)  # [1, 640, 1, 1]

            F1_neighbor = get_neighboring_patch(F1, pi, r2)  # Get neighboring patch in F1

            # Calculate distance based on the specified distance type
            if distance_type == 'l1':
                all_dist = calculate_l1_distance(f0_expanded, F1_neighbor)
            elif distance_type == 'l2':
                all_dist = calculate_l2_distance(f0_expanded, F1_neighbor)
            elif distance_type == 'cosine':
                all_dist = -calculate_cosine_similarity(f0_expanded, F1_neighbor)  # Negative for minimization

            all_dist = all_dist.squeeze(dim=0)  # [25, 25]

            # Update handle point based on the calculated distances
            handle_points[i] = update_handle_points(pi, all_dist, r2)

    return handle_points

def interpolate_feature_patch(feat: torch.Tensor, y: float, x: float, r: int) -> torch.Tensor: # f1_patch = interpolate_feature_patch(F1, pi_new[0], pi_new[1], self.r_1)
    """Obtain the bilinear interpolated feature patch."""
    x0, y0 = torch.floor(x).long(), torch.floor(y).long()
    x1, y1 = x0 + 1, y0 + 1

    weights = torch.tensor([(x1 - x) * (y1 - y), (x1 - x) * (y - y0), (x - x0) * (y1 - y), (x - x0) * (y - y0)])
    weights = weights.to(feat.device)

    try: #NOTE: added by kookie 2024-08-12 08:24:42
        patches = torch.stack([
            feat[:, :, y0 - r:y0 + r + 1, x0 - r:x0 + r + 1],
            feat[:, :, y1 - r:y1 + r + 1, x0 - r:x0 + r + 1],
            feat[:, :, y0 - r:y0 + r + 1, x1 - r:x1 + r + 1],
            feat[:, :, y1 - r:y1 + r + 1, x1 - r:x1 + r + 1]
        ])

    except:
        print("check")

    return torch.sum(weights.view(-1, 1, 1, 1, 1) * patches, dim=0)


def check_handle_reach_target(handle_points: list, target_points: list) -> bool:
    """Check if handle points are close to target points."""
    all_dists = torch.tensor([(p - q).norm().item() for p, q in zip(handle_points, target_points)])
    return (all_dists < 2.0).all().item()
