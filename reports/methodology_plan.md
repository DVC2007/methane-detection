# Methodology and Advanced Experiment Research
**Owner:** Member 3
**Date:** October 7, 2026

## 1. Primary Advanced Method: FiLM-based Wind-Conditioned U-Net
**Description:** 
Instead of basic input-level concatenation, we will use Feature-wise Linear Modulation (FiLM). The model will pass $u$ and $v$ wind vectors through a dense network to generate scale ($\gamma$) and shift ($\beta$) parameters. These parameters will globally modulate the feature maps at the U-Net bottleneck, guiding the network to activate filters relevant to the current wind direction.

**Prerequisites & Expected Inputs:**
*   **Image Data:** `[B, C, H, W]` float32 satellite image tensor.
*   **Wind Data:** `[B, 2]` tensor containing $u$ and $v$ wind vectors.
*   **Baseline:** Standard U-Net architecture (from Member 2) for fair comparison.

**Risks & Limitations (Go/No-Go factor):**
*   *Data Risk:* Member 1 has flagged that ERA5 wind vectors currently lack external spatial alignment for the STARCOP dataset. 
*   *Mitigation:* FiLM applies wind conditions globally to latent features rather than pixel-by-pixel, making it highly robust to spatial misalignment. However, if the wind data is completely inaccessible, this method fails.

## 2. Fallback Method: Weak-Plume Curriculum Learning
**Description:** 
A purely data-driven approach requiring no weather data. We will implement a custom PyTorch DataLoader that sorts training samples by difficulty (plume strength/pixel count). The model will train on massive, obvious leaks first, and gradually be exposed to faint leaks.

**Prerequisites & Expected Inputs:**
*   **Image Data:** Standard image `[B, C, H, W]` and binary mask `[B, 1, H, W]`.
*   **Metadata:** A "difficulty" or "plume concentration" metric assigned to every sample in the canonical reader.

**Risks & Limitations:**
*   Requires Member 1 to verify that plume strength metadata exists and is reliable in the chosen dataset.

## 3. Verification of Claims
*   **Literature-supported facts:** Curriculum learning accelerates convergence on difficult datasets (Bengio et al., 2009). FiLM is a proven architecture for visual reasoning and conditioning without spatial constraints (Perez et al., 2018).
*   **Proposed ideas:** Utilizing FiLM specifically at the U-Net bottleneck to overcome the resolution mismatch between coarse ERA5 weather data and high-resolution Sentinel-2 imagery.
*   **Claims requiring team experiments:** We hypothesize that the FiLM-based U-Net will outperform both the baseline U-Net and early-concatenation methods in reducing False Positives, but this must be empirically measured.

## 4. Controlled Variables & Integration
To ensure a fair comparison against Member 2's baseline, our experiment will lock the following variables:
*   Identical train/test split policy.
*   Identical random seeds and augmentations.
*   Identical evaluation script and metrics.

## 5. Recommendation
**Provisional GO on FiLM Wind-Conditioning.** 
This provides above-medium novelty while actively circumventing the spatial alignment risks identified in Issue #2.
