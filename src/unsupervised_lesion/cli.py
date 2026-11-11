from __future__ import annotations

import argparse
import json

from .data import build_audit
from .evaluate import (calibrate, calibrate_rembg, calibrate_unlabeled_unet,
                       calibrate_region_unet, confirm)
from .evaluate import calibrate_grabcut_unet
from .evaluate import calibrate_ensemble_unet
from .evaluate import calibrate_sam2_unet
from .evaluate import calibrate_sam2_fusion
from .evaluate import calibrate_coarse_regions
from .evaluate import calibrate_coarse_regions_refined
from .evaluate import calibrate_sam2_embedding_regions
from .evaluate import calibrate_boundary_erosion
from .evaluate import fit_ratio_prior, calibrate_ratio_prior
from .evaluate import calibrate_color_region_qwen
from .evaluate import calibrate_color_region_qwen_conditional
from .evaluate import calibrate_color_region_student
from .evaluate import calibrate_color_region_student_fusion
from .evaluate import calibrate_color_patch_cnn
from .model import fit_appearance
from .color_region.teacher import run_pilot as run_color_region_teacher_pilot
from .color_region.student import fit_student as fit_color_region_student
from .color_region.patch_cnn import fit_patch_cnn
from .supervised_segmentation import fit_supervised
from .supervised_color import fit_supervised_color, calibrate_color_regions
from .segformer_segmentation import (fit_segformer, calibrate_hybrid, calibrate_union_lung,
                                     calibrate_unlabeled_lung, fit_dense_segformer)
from .segformer_segmentation import calibrate_dual_ensemble
from .segformer_segmentation import calibrate_unlabeled_lung_ensemble
from .crossval_segformer import fit_crossval
from .stacked_meta import fit_stacked_meta
from .binary_crossval import (fit_binary_crossval, calibrate_binary_multiclass_ensemble,
                              fit_binary_b2_crossval, fit_binary_v2_crossval,
                              fit_binary_relative_lab_crossval, fit_binary_v39_crossval)
from .binary_crossval import fit_binary_v43_crossval
from .binary_crossval import fit_binary_v45_crossval
from .binary_crossval import fit_binary_v70_crossval
from .binary_crossval import fit_binary_v74_crossval
from .masked_pretraining import fit_masked_reconstruction
from .dense_dino_crossval import fit_dense_dino_crossval
from .region_consensus_v41 import calibrate_region_consensus
from .dual_meta_v42 import fit_dual_meta
from .dense_sam2_crossval import fit_dense_sam2_crossval
from .sam2_oof import calibrate_sam2_oof
from .reference_v2_ensemble import calibrate_reference_v2_ensemble
from .logo_crossval import fit_logo_crossval
from .style_ensemble_v46 import calibrate_style_ensemble_v46
from .unmarked_pseudo_v47 import fit_unmarked_pseudo_v47
from .plip_v48 import run_medsiglip_v48_pilot, run_medsiglip_v48_full
from .semantic_ensemble_v49 import calibrate_semantic_ensemble_v49
from .partial_core_v50 import fit_partial_core_v50
from .sam_lobe_v51 import calibrate_sam_lobe_v51
from .sam_dino_student_v52 import fit_sam_dino_student_v52
from .lung_lobe_v53 import calibrate_lung_lobe_v53
from .lung_consensus_v54 import calibrate_lung_consensus_v54
from .lobe_masked_lesion_v55 import calibrate_lobe_masked_lesion_v55
from .relative_qwen_v57 import run_relative_qwen_v57_pilot
from .lobe_crop_crossval_v62 import fit_lobe_crop_v62
from .multiview_crossval_v64 import fit_multiview_v64
from .unetpp_crossval_v65 import fit_unetpp_v65
from .style_multiview_v67 import fit_style_multiview_v67
from .strong_meta_v69 import fit_strong_meta_v69
from .final_ensemble_v71 import calibrate_final_ensemble_v71
from .sealed_v71 import evaluate_sealed_v71, predict_sealed_v71
from .final_ensemble_v72 import calibrate_final_ensemble_v72
from .sealed_v72 import evaluate_sealed_v72, predict_sealed_v72
from .production_v75 import validate_production_v75


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-guarded unsupervised lesion workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    fit = sub.add_parser("fit")
    fit.add_argument("--max-images", type=int)
    ratio_fit = sub.add_parser("fit-ratio-prior")
    ratio_fit.add_argument("--max-images", type=int)
    sub.add_parser("calibrate")
    sub.add_parser("calibrate-rembg")
    sub.add_parser("calibrate-isnet")
    sub.add_parser("calibrate-u2net")
    sub.add_parser("calibrate-unlabeled-unet")
    sub.add_parser("calibrate-region-unet")
    sub.add_parser("calibrate-grabcut-unet")
    sub.add_parser("calibrate-ensemble-unet")
    sub.add_parser("calibrate-sam2-unet")
    sub.add_parser("calibrate-sam2-fusion")
    sub.add_parser("calibrate-coarse-regions")
    sub.add_parser("calibrate-coarse-regions-refined")
    sub.add_parser("calibrate-sam2-embedding-regions")
    sub.add_parser("calibrate-boundary-erosion")
    sub.add_parser("calibrate-ratio-prior")
    color_pilot = sub.add_parser("pilot-color-region-teacher")
    color_pilot.add_argument("--max-images", type=int, default=12)
    color_pilot.add_argument("--regions-per-image", type=int, default=4)
    color_pilot.add_argument("--confidence", type=float, default=0.60)
    sub.add_parser("calibrate-color-region-qwen")
    sub.add_parser("calibrate-color-region-qwen-conditional")
    region_student = sub.add_parser("fit-color-region-student")
    region_student.add_argument("--max-images", type=int, default=128)
    clean_student = sub.add_parser("fit-clean-color-region-student")
    clean_student.add_argument("--max-images", type=int, default=128)
    patch_cnn = sub.add_parser("fit-color-patch-cnn")
    patch_cnn.add_argument("--max-images", type=int, default=128)
    sub.add_parser("calibrate-color-patch-cnn")
    sub.add_parser("calibrate-color-region-student")
    sub.add_parser("calibrate-color-region-student-fusion")
    sub.add_parser("confirm")
    supervised = sub.add_parser("fit-supervised")
    supervised.add_argument("--epochs", type=int, default=60)
    sub.add_parser("fit-supervised-color")
    sub.add_parser("calibrate-supervised-color-regions")
    segformer = sub.add_parser("fit-segformer")
    segformer.add_argument("--epochs", type=int, default=50)
    sub.add_parser("calibrate-segformer-hybrid")
    sub.add_parser("calibrate-segformer-union-lung")
    sub.add_parser("calibrate-segformer-unlabeled-lung")
    dense_segformer = sub.add_parser("fit-dense-segformer")
    dense_segformer.add_argument("--epochs", type=int, default=50)
    sub.add_parser("calibrate-dual-segformer")
    sub.add_parser("calibrate-dual-segformer-unlabeled-lung")
    crossval = sub.add_parser("fit-crossval-segformer")
    crossval.add_argument("--epochs", type=int, default=35)
    sub.add_parser("fit-stacked-meta")
    binary = sub.add_parser("fit-binary-crossval")
    binary.add_argument("--epochs", type=int, default=35)
    sub.add_parser("calibrate-binary-multiclass-ensemble")
    sub.add_parser("calibrate-sam2-oof")
    binary_b2 = sub.add_parser("fit-binary-b2-crossval")
    binary_b2.add_argument("--epochs", type=int, default=25)
    binary_v2 = sub.add_parser("fit-binary-v2-crossval")
    binary_v2.add_argument("--epochs", type=int, default=35)
    sub.add_parser("calibrate-reference-v2-ensemble")
    binary_relative = sub.add_parser("fit-binary-relative-lab-crossval")
    binary_relative.add_argument("--epochs", type=int, default=35)
    logo = sub.add_parser("fit-logo-crossval")
    logo.add_argument("--epochs", type=int, default=25)
    masked = sub.add_parser("fit-masked-pretraining")
    masked.add_argument("--epochs", type=int, default=3)
    binary_v39 = sub.add_parser("fit-binary-v39-crossval")
    binary_v39.add_argument("--epochs", type=int, default=35)
    dense_dino = sub.add_parser("fit-dense-dino-crossval")
    dense_dino.add_argument("--epochs", type=int, default=60)
    sub.add_parser("calibrate-region-consensus-v41")
    sub.add_parser("fit-dual-meta-v42")
    binary_v43 = sub.add_parser("fit-binary-v43-crossval")
    binary_v43.add_argument("--epochs", type=int, default=35)
    dense_sam2 = sub.add_parser("fit-dense-sam2-crossval")
    dense_sam2.add_argument("--epochs", type=int, default=60)
    binary_v45 = sub.add_parser("fit-binary-v45-crossval")
    binary_v45.add_argument("--epochs", type=int, default=35)
    sub.add_parser("calibrate-style-ensemble-v46")
    sub.add_parser("fit-unmarked-pseudo-v47")
    sub.add_parser("pilot-medsiglip-v48")
    sub.add_parser("validate-medsiglip-v48")
    sub.add_parser("calibrate-semantic-ensemble-v49")
    sub.add_parser("fit-partial-core-v50")
    sub.add_parser("calibrate-sam-lobe-v51")
    sub.add_parser("fit-sam-dino-student-v52")
    sub.add_parser("calibrate-lung-lobe-v53")
    sub.add_parser("calibrate-lung-consensus-v54")
    sub.add_parser("calibrate-lobe-masked-lesion-v55")
    sub.add_parser("pilot-relative-qwen-v57")
    lobe_crop_v62 = sub.add_parser("fit-lobe-crop-v62")
    lobe_crop_v62.add_argument("--epochs", type=int, default=25)
    multiview_v64 = sub.add_parser("fit-multiview-v64")
    multiview_v64.add_argument("--epochs", type=int, default=10)
    unetpp_v65 = sub.add_parser("fit-unetpp-v65")
    unetpp_v65.add_argument("--epochs", type=int, default=20)
    style_v67 = sub.add_parser("fit-style-multiview-v67")
    style_v67.add_argument("--epochs", type=int, default=5)
    sub.add_parser("fit-strong-meta-v69")
    binary_v70 = sub.add_parser("fit-binary-v70-crossval")
    binary_v70.add_argument("--epochs", type=int, default=30)
    binary_v74 = sub.add_parser("fit-binary-v74-crossval")
    binary_v74.add_argument("--epochs", type=int, default=40)
    sub.add_parser("calibrate-final-ensemble-v71")
    sub.add_parser("predict-sealed-v71")
    sub.add_parser("evaluate-sealed-v71")
    sub.add_parser("calibrate-final-ensemble-v72")
    sub.add_parser("predict-sealed-v72")
    sub.add_parser("evaluate-sealed-v72")
    sub.add_parser("validate-production-v75")
    args = parser.parse_args()
    if args.command == "audit":
        result = build_audit()
    elif args.command == "fit":
        result = fit_appearance(args.max_images)
    elif args.command == "fit-ratio-prior":
        result = fit_ratio_prior(args.max_images)
    elif args.command == "calibrate":
        result = calibrate()
    elif args.command == "calibrate-rembg":
        result = calibrate_rembg()
    elif args.command == "calibrate-isnet":
        result = calibrate_rembg("isnet-general-use")
    elif args.command == "calibrate-u2net":
        result = calibrate_rembg("u2net")
    elif args.command == "calibrate-unlabeled-unet":
        result = calibrate_unlabeled_unet()
    elif args.command == "calibrate-region-unet":
        result = calibrate_region_unet()
    elif args.command == "calibrate-grabcut-unet":
        result = calibrate_grabcut_unet()
    elif args.command == "calibrate-ensemble-unet":
        result = calibrate_ensemble_unet()
    elif args.command == "calibrate-sam2-unet":
        result = calibrate_sam2_unet()
    elif args.command == "calibrate-sam2-fusion":
        result = calibrate_sam2_fusion()
    elif args.command == "calibrate-coarse-regions":
        result = calibrate_coarse_regions()
    elif args.command == "calibrate-coarse-regions-refined":
        result = calibrate_coarse_regions_refined()
    elif args.command == "calibrate-sam2-embedding-regions":
        result = calibrate_sam2_embedding_regions()
    elif args.command == "calibrate-boundary-erosion":
        result = calibrate_boundary_erosion()
    elif args.command == "calibrate-ratio-prior":
        result = calibrate_ratio_prior()
    elif args.command == "pilot-color-region-teacher":
        result = run_color_region_teacher_pilot(args.max_images, args.regions_per_image,
                                                args.confidence)
    elif args.command == "calibrate-color-region-qwen":
        result = calibrate_color_region_qwen()
    elif args.command == "calibrate-color-region-qwen-conditional":
        result = calibrate_color_region_qwen_conditional()
    elif args.command == "fit-color-region-student":
        result = fit_color_region_student(args.max_images)
    elif args.command == "fit-clean-color-region-student":
        result = fit_color_region_student(args.max_images, "clean_context")
    elif args.command == "fit-color-patch-cnn":
        result = fit_patch_cnn(args.max_images)
    elif args.command == "calibrate-color-patch-cnn":
        result = calibrate_color_patch_cnn()
    elif args.command == "calibrate-color-region-student":
        result = calibrate_color_region_student()
    elif args.command == "calibrate-color-region-student-fusion":
        result = calibrate_color_region_student_fusion()
    elif args.command == "fit-supervised":
        result = fit_supervised(args.epochs)
    elif args.command == "fit-supervised-color":
        result = fit_supervised_color()
    elif args.command == "calibrate-supervised-color-regions":
        result = calibrate_color_regions()
    elif args.command == "fit-segformer":
        result = fit_segformer(args.epochs)
    elif args.command == "calibrate-segformer-hybrid":
        result = calibrate_hybrid()
    elif args.command == "calibrate-segformer-union-lung":
        result = calibrate_union_lung()
    elif args.command == "calibrate-segformer-unlabeled-lung":
        result = calibrate_unlabeled_lung()
    elif args.command == "fit-dense-segformer":
        result = fit_dense_segformer(args.epochs)
    elif args.command == "calibrate-dual-segformer":
        result = calibrate_dual_ensemble()
    elif args.command == "calibrate-dual-segformer-unlabeled-lung":
        result = calibrate_unlabeled_lung_ensemble()
    elif args.command == "fit-crossval-segformer":
        result = fit_crossval(args.epochs)
    elif args.command == "fit-stacked-meta":
        result = fit_stacked_meta()
    elif args.command == "fit-binary-crossval":
        result = fit_binary_crossval(args.epochs)
    elif args.command == "calibrate-binary-multiclass-ensemble":
        result = calibrate_binary_multiclass_ensemble()
    elif args.command == "calibrate-sam2-oof":
        result = calibrate_sam2_oof()
    elif args.command == "fit-binary-b2-crossval":
        result = fit_binary_b2_crossval(args.epochs)
    elif args.command == "fit-binary-v2-crossval":
        result = fit_binary_v2_crossval(args.epochs)
    elif args.command == "calibrate-reference-v2-ensemble":
        result = calibrate_reference_v2_ensemble()
    elif args.command == "fit-binary-relative-lab-crossval":
        result = fit_binary_relative_lab_crossval(args.epochs)
    elif args.command == "fit-logo-crossval":
        result = fit_logo_crossval(args.epochs)
    elif args.command == "fit-masked-pretraining":
        result = fit_masked_reconstruction(args.epochs)
    elif args.command == "fit-binary-v39-crossval":
        result = fit_binary_v39_crossval(args.epochs)
    elif args.command == "fit-dense-dino-crossval":
        result = fit_dense_dino_crossval(args.epochs)
    elif args.command == "calibrate-region-consensus-v41":
        result = calibrate_region_consensus()
    elif args.command == "fit-dual-meta-v42":
        result = fit_dual_meta()
    elif args.command == "fit-binary-v43-crossval":
        result = fit_binary_v43_crossval(args.epochs)
    elif args.command == "fit-dense-sam2-crossval":
        result = fit_dense_sam2_crossval(args.epochs)
    elif args.command == "fit-binary-v45-crossval":
        result = fit_binary_v45_crossval(args.epochs)
    elif args.command == "calibrate-style-ensemble-v46":
        result = calibrate_style_ensemble_v46()
    elif args.command == "fit-unmarked-pseudo-v47":
        result = fit_unmarked_pseudo_v47()
    elif args.command == "pilot-medsiglip-v48":
        result = run_medsiglip_v48_pilot()
    elif args.command == "validate-medsiglip-v48":
        result = run_medsiglip_v48_full()
    elif args.command == "calibrate-semantic-ensemble-v49":
        result = calibrate_semantic_ensemble_v49()
    elif args.command == "fit-partial-core-v50":
        result = fit_partial_core_v50()
    elif args.command == "calibrate-sam-lobe-v51":
        result = calibrate_sam_lobe_v51()
    elif args.command == "fit-sam-dino-student-v52":
        result = fit_sam_dino_student_v52()
    elif args.command == "calibrate-lung-lobe-v53":
        result = calibrate_lung_lobe_v53()
    elif args.command == "calibrate-lung-consensus-v54":
        result = calibrate_lung_consensus_v54()
    elif args.command == "calibrate-lobe-masked-lesion-v55":
        result = calibrate_lobe_masked_lesion_v55()
    elif args.command == "pilot-relative-qwen-v57":
        result = run_relative_qwen_v57_pilot()
    elif args.command == "fit-lobe-crop-v62":
        result = fit_lobe_crop_v62(args.epochs)
    elif args.command == "fit-multiview-v64":
        result = fit_multiview_v64(args.epochs)
    elif args.command == "fit-unetpp-v65":
        result = fit_unetpp_v65(args.epochs)
    elif args.command == "fit-style-multiview-v67":
        result = fit_style_multiview_v67(args.epochs)
    elif args.command == "fit-strong-meta-v69":
        result = fit_strong_meta_v69()
    elif args.command == "fit-binary-v70-crossval":
        result = fit_binary_v70_crossval(args.epochs)
    elif args.command == "fit-binary-v74-crossval":
        result = fit_binary_v74_crossval(args.epochs)
    elif args.command == "calibrate-final-ensemble-v71":
        result = calibrate_final_ensemble_v71()
    elif args.command == "predict-sealed-v71":
        result = predict_sealed_v71()
    elif args.command == "evaluate-sealed-v71":
        result = evaluate_sealed_v71()
    elif args.command == "calibrate-final-ensemble-v72":
        result = calibrate_final_ensemble_v72()
    elif args.command == "predict-sealed-v72":
        result = predict_sealed_v72()
    elif args.command == "evaluate-sealed-v72":
        result = evaluate_sealed_v72()
    elif args.command == "validate-production-v75":
        result = validate_production_v75()
    else:
        result = confirm()
    compact = {key: result[key] for key in result
               if key not in {"fit_rows", "excluded_reference_aliases", "image_audit", "results", "per_image", "candidates"}}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()

