import argparse
from src.media_loader import load_media
from src.face_validator import validate_media_for_faces
from src.face_extractor import extract_faces
from src.detectors import load_detectors, run_detectors
from src.ensemble import run_ensemble


def analyze(file_path):
    print(f"\nAnalyzing: {file_path}")
    print("=" * 50)

    # Step 1 — Load media
    result = load_media(file_path)
    if result['error']:
        print(f"Error: {result['error']}")
        return

    print(f"Media type: {result['media_type']}")

    # Step 2 — Validate faces
    validation = validate_media_for_faces(result['frames'])
    if not validation['has_face']:
        print(f"\nResult: UNKNOWN")
        print(f"Reason: {validation['message']}")
        return

    print(validation['message'])

    # Step 3 — Extract faces
    faces = extract_faces(result['frames'], validation['face_details'])

    # Step 4 — Load detectors
    detector_a, detector_b, detector_c = load_detectors()

    # Step 5 — Run detectors
    all_face_scores = []
    for face in faces:
        scores = run_detectors(
            face['face_tensor'],
            face['face_crop'],
            detector_a, detector_b, detector_c
        )
        all_face_scores.append(scores)

    # Step 6 — Ensemble
    final = run_ensemble(all_face_scores)

    # Step 7 — Output
    print(f"\nPREDICTION:  {final['prediction']}")
    print(f"CONFIDENCE:  {final['confidence']}%")
    print(f"FACES:       {final['total_faces_analyzed']}")
    if final['reasons']:
        print("REASONS:")
        for r in final['reasons']:
            print(f"  • {r}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Deepfake Detector — analyze media for manipulation'
    )
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to image or video file'
    )
    args = parser.parse_args()
    analyze(args.file_path)
