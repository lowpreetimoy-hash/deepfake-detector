def format_output(final):
    """
    Formats the ensemble result into a clean readable output.
    Used by both main.py and app.py.
    """
    lines = []
    lines.append("=" * 50)
    lines.append(f"PREDICTION:  {final['prediction']}")
    lines.append(f"CONFIDENCE:  {final['confidence']}%")
    lines.append(f"FACES:       {final['total_faces_analyzed']}")

    if final['reasons']:
        lines.append("REASONS:")
        for reason in final['reasons']:
            lines.append(f"  • {reason}")

    lines.append("=" * 50)
    return "\n".join(lines)


def get_verdict_color(prediction):
    """
    Returns color string for UI display.
    Used by Streamlit app for banner colors.
    """
    if prediction == 'FAKE':
        return 'red'
    elif prediction == 'REAL':
        return 'green'
    else:
        return 'orange'


def get_verdict_emoji(prediction):
    """
    Returns emoji for UI display.
    """
    if prediction == 'FAKE':
        return '⚠️'
    elif prediction == 'REAL':
        return '✅'
    else:
        return '❓'


def get_summary(final):
    """
    Returns a one-line summary string.
    """
    emoji = get_verdict_emoji(final['prediction'])
    return (
        f"{emoji} {final['prediction']} "
        f"({final['confidence']}% confidence) — "
        f"{final['total_faces_analyzed']} face(s) analyzed"
    )
