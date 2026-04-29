from .GroundingDINO import build_groundingdino


def build_model(args):
    from .registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model = build_func(args)
    return model
