import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if not target_text:
        return 1
    return editdistance.distance(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    return calc_cer(target_text.split(), predicted_text.split())
