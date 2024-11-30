from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle
from skimage.morphology import erosion, dilation, disk
from sklearn.metrics import jaccard_score, f1_score

sl = []

for slice_index in slices:
    ct_slice = ct_volume[slice_index, :, :]
    liver_mask_slice = liver_mask_volume[slice_index, :, :]
    tumor_mask_slice = tumors_mask_volume[slice_index, :, :]
    
    liver_extracted_ct_slice = ct_slice * liver_mask_slice

    liver_pixels_slice = liver_extracted_ct_slice[liver_mask_slice > 0]
    liver_pixels_slice = liver_pixels_slice.reshape(-1, 1)

    scaler = StandardScaler()
    liver_pixels_scaled_slice = scaler.fit_transform(liver_pixels_slice)

    k = 2
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(liver_pixels_scaled_slice)
    labels = kmeans.predict(liver_pixels_scaled_slice)

    clustered_image = np.zeros(liver_extracted_ct_slice.shape, dtype=int)
    clustered_image[liver_mask_slice > 0] = labels
    unique_clusters = np.unique(labels)
    num_clusters = len(unique_clusters)

    plt.figure(figsize=(6, 3))
    for i in range(num_clusters):
        plt.subplot(1, num_clusters, i + 1)
        mask = clustered_image == i
        plt.imshow(mask, cmap='gray')
        plt.title(f'Cluster {i}')
        plt.axis('off')

    plt.suptitle(f'KMeans Clusters for Slice {slice_index}', fontsize=8)
    plt.tight_layout()
    plt.show()

    sample_size = 50000
    n_pixels = liver_pixels_scaled_slice.shape[0]
    indices = shuffle(np.arange(n_pixels), random_state=42)[:sample_size]
    pixels_sample = liver_pixels_scaled_slice[indices]

    silhouette_scores = []
    best_score = -1
    best_k = None
    best_kmeans = None

    print(f"Evaluating silhouette scores for Slice {slice_index}:")
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(pixels_sample)
        sample_labels = kmeans.predict(pixels_sample)
        score = silhouette_score(pixels_sample, sample_labels)
        silhouette_scores.append(score)
        print(f"k: {k:<2} , Silhouette Score: {score:.3f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_kmeans = kmeans

    labels = best_kmeans.labels_
    clustered_image = np.zeros(liver_extracted_ct_slice.shape, dtype=int)
    clustered_image[liver_mask_slice > 0] = labels

    unique_clusters = np.unique(labels)
    num_clusters = len(unique_clusters)

    plt.figure(figsize=(6, 3))
    for i in range(num_clusters):
        plt.subplot(1, num_clusters, i + 1)
        mask = clustered_image == i
        plt.imshow(mask, cmap='gray')
        plt.title(f'Cluster {i}')
        plt.axis('off')

    plt.figure(figsize=(6, 3))
    plt.plot(np.arange(2, 10), silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Silhouette Score vs Number of Clusters for Slice {slice_index}')
    plt.grid()
    plt.show()

    print(f"Best number of Clusters (K) for Slice {slice_index} based on silhouette score: {best_k}")
    print(f"Best silhouette score for Slice {slice_index}: {best_score:.3f}")
    labels = best_kmeans.predict(liver_pixels_scaled_slice)

    clustered_image = np.zeros(liver_extracted_ct_slice.shape, dtype=int)
    clustered_image[liver_mask_slice > 0] = labels

    pixel_counts = np.bincount(labels)
    min_cluster_id = np.argmin(pixel_counts)
    min_cluster_count = pixel_counts[min_cluster_id]

    print(f"Cluster {min_cluster_id} for Slice {slice_index} has the minimum number of pixels: {min_cluster_count}")
    min_cluster_mask = (clustered_image == min_cluster_id)

    structuring_element = disk(3)
    eroded_mask = erosion(min_cluster_mask, structuring_element)
    structuring_element = disk(10)
    dilated_mask = dilation(eroded_mask, structuring_element)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(min_cluster_mask, cmap='gray')
    axes[0].set_title(f'Original Cluster {min_cluster_id} Mask')
    axes[0].axis('off')
    axes[1].imshow(eroded_mask, cmap='gray')
    axes[1].set_title(f'Eroded Mask (Cluster {min_cluster_id})')
    axes[1].axis('off')
    axes[2].imshow(dilated_mask, cmap='gray')
    axes[2].set_title(f'Dilated Mask (Cluster {min_cluster_id})')
    axes[2].axis('off')
    plt.suptitle(f'Morphological Operations on Cluster Mask for Slice {slice_index}', fontsize=8)
    plt.tight_layout()
    plt.show()

    predicted_flat = dilated_mask.flatten()
    ground_truth_flat = tumor_mask_slice.flatten()

    iou = jaccard_score(ground_truth_flat, predicted_flat)
    dice = f1_score(ground_truth_flat, predicted_flat)

    print(f"\nEvaluation Metrics for Tumor Segmentation for Slice {slice_index}:")
    print(f"Intersection over Union (IoU): {iou:.4f}")
    print(f"Dice Coefficient: {dice:.4f}")

    ct_rgb = np.stack([liver_extracted_ct_slice] * 3, axis=-1)
    predicted_overlay = np.zeros_like(ct_rgb)
    ground_truth_overlay = np.zeros_like(ct_rgb)

    predicted_overlay[dilated_mask == 1] = [255, 0, 0]
    ground_truth_overlay[tumor_mask_slice == 1] = [0, 255, 0]

    combined_overlay = ct_rgb.copy()
    combined_overlay = np.where(predicted_overlay.any(axis=-1, keepdims=True), predicted_overlay, combined_overlay)
    combined_overlay = np.where(ground_truth_overlay.any(axis=-1, keepdims=True), ground_truth_overlay, combined_overlay)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(liver_extracted_ct_slice, cmap='gray')
    plt.title(f'Original ct Image for Slice {slice_index}')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(liver_extracted_ct_slice, cmap='gray')
    plt.imshow(predicted_overlay, alpha=0.3)
    plt.title(f'Predicted Tumor Mask (Red) for Slice {slice_index}')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(liver_extracted_ct_slice, cmap='gray')
    plt.imshow(ground_truth_overlay, alpha=0.3)
    plt.title(f'Ground Truth Tumor Mask (Green) for Slice {slice_index}')
    plt.axis('off')

    plt.suptitle(f'Tumor Segmentation Results for Slice {slice_index}', fontsize=8)
    plt.tight_layout()
    plt.show()

    predicted_tumor_mask = np.logical_and(dilated_mask, liver_mask_slice)
    predicted_overlay = np.zeros_like(ct_rgb)
    predicted_overlay[predicted_tumor_mask == 1] = [255, 0, 0]
    combined_overlay = np.where(predicted_overlay.any(axis=-1, keepdims=True), predicted_overlay, combined_overlay)

    predicted_flat = predicted_tumor_mask.flatten()
    ground_truth_flat = tumor_mask_slice.flatten()

    iou = jaccard_score(ground_truth_flat, predicted_flat)
    dice = f1_score(ground_truth_flat, predicted_flat)

    print(f"\nEvaluation Metrics for Tumor Segmentation for Slice {slice_index}:")
    print(f"Intersection over Union (IoU): {iou:.4f}")
    print(f"Dice Coefficient: {dice:.4f}")

    plt.figure(figsize=(3, 3))
    plt.imshow(liver_extracted_ct_slice, cmap='gray')
    plt.imshow(predicted_overlay, alpha=0.3)
    plt.imshow(ground_truth_overlay, alpha=0.3)
    plt.title(f'Predicted (Red) vs Ground Truth (Green) Tumor Masks for Slice {slice_index}')
    plt.axis('off')
    plt.show()

