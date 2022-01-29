from typing import Any, Dict, List, Set

import mmcv


def check(scope: List[str], msg: str):
    print('.'.join(scope), ':', msg)
    # import ipdb; ipdb.set_trace()


def cmp(x: Dict[Any, Any] ,y: Dict[Any, Any], scope: List[str] = None):   
    if scope is None:
        scope = []
    if not isinstance(x, type(y)) and not isinstance(y, type(x)):
        check(scope, f"type(x)={type(x)} should match type(y)={type(y)}. QUITING!")
        return
    if isinstance(x, dict) or isinstance(x, mmcv.Config):
        if x.keys() != y.keys():
            check(scope, f"x.keys()={x.keys()} should match y.keys()={y.keys()}.")
            keys = x.keys() & y.keys()
        else:
            keys = x.keys()
        for k in keys:
            cmp(x[k], y[k], scope + [k])
    elif isinstance(x, list):                  
        if len(x) != len(y):
            check(scope, f"len(x)={len(x)} should match len(y)={len(y)}.QUITING!")
            return
        for i, (a, b) in enumerate(zip(x, y)):
            cmp(a, b, scope + [str(i)])
    elif x != y:
        check(scope, f"x={x} should match y={y}.")
        return
    else:  # x == y
        pass


def main(x: str, y: str, stop_keys: Set[str] = ...):
    if stop_keys is ...:
        stop_keys = {'train_pipeline', 'test_pipeline', 'dataset_type', 'data_root', 'img_norm_cfg', 'find_unused_parameters', 'optimizer_config'}
    if stop_keys is None:
        stop_keys = {}
    try:
        x: Dict[str, mmcv.ConfigDict] = mmcv.Config.fromfile(x)
    except Exception as e:
        print(f"Error when loading {x}.")
        raise e
    try:
        y: Dict[str, mmcv.ConfigDict] = mmcv.Config.fromfile(y)
    except Exception as e:
        print(f"Error when loading {y}.")
        raise e
    x = {k: v for k, v in x.items() if k not in stop_keys}
    y = {k: v for k, v in y.items() if k not in stop_keys}
    cmp(x, y)
 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Configs")
    parser.add_argument('x', help="first config file path")
    parser.add_argument('y', help="second config file path")
    args = parser.parse_args()
    main(args.x, args.y)