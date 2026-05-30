import numpy as np

# Detector weights for weighted voting
# These reflect how reliable each detector is
# Can be tuned after fine-tuning
DETECTOR_WEIGHTS = {
    'face_swap': 0.35,
    'ai_generated': 0.35,
    'manual_edit': 0.30
}

# Threshold above which final result is FAKE
FAKE_THRESHOLD = 0.80

# Individual detector threshold for reason generation
REASON_THRESHOLD = 0.5


def weighted_soft_vote(scores):
    """
    Combines all detector scores using weighted soft voting.
    Returns: final confidence score between 0 and 1
    """
    face_swap_score = scores['face_swap_score']
    ai_generated_score = scores['ai_generated_score']
    manual_edit_score = scores['manual_edit_score']

    # Weighted average
    weighted_score = (
        (face_swap_score * DETECTOR_WEIGHTS['face_swap']) +
        (ai_generated_score * DETECTOR_WEIGHTS['ai_generated']) +
        (manual_edit_score * DETECTOR_WEIGHTS['manual_edit'])
    )

    return round(weighted_score, 4)


def apply_tiebreaker(scores, weighted_score):
    """
    Handles ambiguous cases where weighted score is near threshold.
    Returns: adjusted confidence score
    """
    face_swap_score = scores['face_swap_score']
    ai_generated_score = scores['ai_generated_score']
    manual_edit_score = scores['manual_edit_score']

    # Count how many detectors say fake
    fake_votes = sum([
        face_swap_score > FAKE_THRESHOLD,
        ai_generated_score > FAKE_THRESHOLD,
        manual_edit_score > FAKE_THRESHOLD
    ])

    # If score is ambiguous (between 0.4 and 0.6)
    if 0.4 <= weighted_score <= 0.6:
        if fake_votes >= 2:
            # Majority say fake — push toward fake
            weighted_score = max(weighted_score, 0.55)
        elif fake_votes == 0:
            # All say real — push toward real
            weighted_score = min(weighted_score, 0.45)

    return round(weighted_score, 4)


def generate_reasons(scores):
    """
    Generates human readable reasons based on detector scores.
    Returns: list of reason strings
    """
    reasons = []

    face_swap_score = scores['face_swap_score']
    ai_generated_score = scores['ai_generated_score']
    manual_edit_score = scores['manual_edit_score']

    # Detector A reasons
    if face_swap_score > REASON_THRESHOLD:
        if face_swap_score > 0.8:
            reasons.append("Strong face swap artifacts detected")
        elif face_swap_score > 0.65:
            reasons.append("Face blending inconsistencies detected")
        else:
            reasons.append("Possible face swap pattern detected")

    # Detector B reasons
    if ai_generated_score > REASON_THRESHOLD:
        if ai_generated_score > 0.8:
            reasons.append("Strong AI generation pattern found")
        elif ai_generated_score > 0.65:
            reasons.append("Synthetic texture pattern detected")
        else:
            reasons.append("Possible AI generated content detected")

    # Detector C reasons
    if manual_edit_score > REASON_THRESHOLD:
        if manual_edit_score > 0.8:
            reasons.append("Significant manual editing detected")
        elif manual_edit_score > 0.65:
            reasons.append("Lighting or texture inconsistency detected")
        else:
            reasons.append("Possible manual editing detected")

    # If no reasons but still fake
    if len(reasons) == 0:
        reasons.append("Subtle manipulation pattern detected")

    return reasons


def get_final_verdict(scores):
    """
    Combines weighted voting, tiebreaker and reasons into final verdict.
    Returns: dictionary with complete analysis result
    """
    # Step 1 — weighted vote
    weighted_score = weighted_soft_vote(scores)

    # Step 2 — apply tiebreaker
    final_score = apply_tiebreaker(scores, weighted_score)

    # Step 3 — determine prediction
    prediction = 'FAKE' if final_score > FAKE_THRESHOLD else 'REAL'

    # Step 4 — generate reasons
    reasons = generate_reasons(scores) if prediction == 'FAKE' else []

    # Step 5 — convert to percentage
    if prediction == 'FAKE':
        confidence_pct = round(final_score * 100, 1)
    else:
        confidence_pct = round((1 - final_score) * 100, 1)

    return {
        'prediction': prediction,
        'confidence': confidence_pct,
        'final_score': final_score,
        'individual_scores': scores,
        'reasons': reasons
    }


def run_ensemble(all_face_scores):
    """
    Runs ensemble on all faces and returns overall media verdict.
    Returns: dictionary with final result
    """
    if len(all_face_scores) == 0:
        return {
            'prediction': 'UNKNOWN',
            'confidence': 0,
            'final_score': 0,
            'reasons': ['No human face detected in media'],
            'individual_scores': {},
            'total_faces_analyzed': 0,
            'fake_faces_found': 0
        }

    # Analyze each face
    face_verdicts = []
    for scores in all_face_scores:
        verdict = get_final_verdict(scores)
        face_verdicts.append(verdict)

    # If any face is FAKE — media is FAKE
    fake_verdicts = [v for v in face_verdicts if v['prediction'] == 'FAKE']

    if len(fake_verdicts) > 0:
        # Return the highest confidence fake verdict
        final = max(fake_verdicts, key=lambda x: x['confidence'])
    else:
        # All faces real — return lowest confidence real verdict (most conservative)
        final = min(face_verdicts, key=lambda x: x['confidence'])

    final['total_faces_analyzed'] = len(face_verdicts)
    final['fake_faces_found'] = len(fake_verdicts)

    return final
