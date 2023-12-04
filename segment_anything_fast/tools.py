def replace_with_custom_fn_if_matches_filter(
    model, replacement_fn, filter_fn, cur_fqn=''
) -> None:
    """
    For each `child` in `model`, replaces it with `replacement_fn(child)`
    if `filter_fn(child)` is `True`
    """
    name_to_child = dict(model.named_children())
    for name, child in name_to_child.items():
        if cur_fqn == '':
            new_fqn = name
        else:
            new_fqn = f'{cur_fqn}.{name}'
        if filter_fn(child, new_fqn):
            new_child = replacement_fn(child)
            setattr(model, name, new_child)
        else:
            replace_with_custom_fn_if_matches_filter(
                child, replacement_fn, filter_fn, new_fqn)


def apply_eval_dtype_predictor(predictor, dtype=None):

    def prep_model(model, dtype):
        if dtype is not None:
            return model.eval().to(dtype)
        return model.eval()

    predictor.model.image_encoder = prep_model(
        predictor.model.image_encoder, dtype)
    predictor.model.prompt_encoder = prep_model(
        predictor.model.prompt_encoder, dtype)
    predictor.model.mask_decoder = prep_model(
        predictor.model.mask_decoder, dtype)

    return predictor
