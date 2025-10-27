from torch.optim.lr_scheduler import LambdaLR

def build_warmup_scheduler(optimizer, warmup_type: str, warmup_steps: int, total_steps: int | None = None):
    """
    Erstellt einen Lernraten-Scheduler mit optionalem Warmup.
    
    Args:
        optimizer: PyTorch Optimizer
        warmup_type: 'none' | 'linear' | 'constant'
        warmup_steps: Anzahl der Update-Schritte für das Warmup
        total_steps: (optional) Gesamtanzahl an Schritten für Decay nach Warmup
    """
    if warmup_type == "none":
        return None

    if warmup_type == "constant":
        # Warmup linear bis warmup_steps, danach konstant
        def lr_lambda(current_step: int):
            if warmup_steps <= 0:
                return 1.0
            return min(1.0, float(current_step) / float(warmup_steps))
        return LambdaLR(optimizer, lr_lambda)

    if warmup_type == "linear":
        # Linearer Warmup; danach linearer Decay auf 0 (wenn total_steps angegeben)
        if not total_steps or total_steps <= 0:
            # Wenn keine total_steps angegeben → nur Warmup, danach konstant
            def lr_lambda(current_step: int):
                if warmup_steps <= 0:
                    return 1.0
                return min(1.0, float(current_step) / float(warmup_steps))
            return LambdaLR(optimizer, lr_lambda)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0,
                float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )
        return LambdaLR(optimizer, lr_lambda)

    raise ValueError(f"Unknown warmup_type: {warmup_type}")
