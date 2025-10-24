def choices_map(ex):
    """Mapping choices to options for ARC-C challenge

    Args:
        ex (dict): Dictionary as in the ARC-C dataset

    Raises:
        ValueError: Incase the input does not match the requisite ARC-C format

    Returns:
        dict: collated version of labels-choices as required for model input
    """
    ch = ex["choices"]
    if "label" in ch and "text" in ch:
        labels = [str(x).strip().upper() for x in ch["label"]]
        texts = ch["text"]
        return dict(zip(labels, texts))
    raise ValueError(f"Format does not match expected structure!!!")