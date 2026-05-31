from src.media_loader import load_media
from src.face_validator import validate_media_for_faces
from src.face_extractor import extract_faces
from src.detectors import load_detectors, run_detectors
from src.ensemble import run_ensemble

# Change this path for each test
IMAGE_PATH = r'D:\DeepFake Detector\tested photos\real_webcam1 (1).jpg'

result = load_media(IMAGE_PATH)
validation = validate_media_for_faces(result['frames'])
faces = extract_faces(result['frames'], validation['face_details'])

detector_a, detector_b, detector_c = load_detectors()

all_face_scores = []
for face in faces:
    scores = run_detectors(
        face['face_tensor'],
        face['face_crop'],
        detector_a, detector_b, detector_c
    )
    all_face_scores.append(scores)
    print(f"\nRaw detector scores:")
    print(f"  Detector A (Face Swap):   {scores['face_swap_score']:.4f}")
    print(f"  Detector B (AI Generated):{scores['ai_generated_score']:.4f}")
    print(f"  Detector C (Manual Edit): {scores['manual_edit_score']:.4f}")

final = run_ensemble(all_face_scores)

print(f"\n{'='*40}")
print(f"PREDICTION:  {final['prediction']}")
print(f"CONFIDENCE:  {final['confidence']}%")
if final['reasons']:
    print("REASONS:")
    for r in final['reasons']:
        print(f"  • {r}")
print(f"{'='*40}")
