# For DSS project
from viskit import frontend, core
import sys
import numbers
import os
import argparse
import pandas as pd
import numpy as np


def get_plot_instruction(
        plot_keys,
        exps_data,
        x_keys=None,
        split_keys=None,
        group_keys=None,
        filters=None,
        exclusions=None,
        plot_height=None,
        filter_nan=False,
        custom_filter=None,
        legend_post_processor=None,
        custom_series_splitter=None,
):
    if x_keys is None:
        x_keys = []

    """
    A custom filter might look like
    "lambda exp: exp.flat_params['algo_params_base_kwargs.batch_size'] == 64"
    """
    selector = core.Selector(exps_data)
    if legend_post_processor is None:
        legend_post_processor = lambda x: x
    if filters is None:
        filters = dict()
    if exclusions is None:
        exclusions = []
    if split_keys is None:
        split_keys = []
    if group_keys is None:
        group_keys = []
    if plot_height is None:
        plot_height = 300 * len(plot_keys)
    for k, v in filters.items():
        selector = selector.where(k, str(v))
    for k, v in exclusions:
        selector = selector.where_not(k, str(v))
    if custom_filter is not None:
        selector = selector.custom_filter(custom_filter)

    split_selectors = [selector]
    split_titles = ["Plot"]
    plots = []
    counter = 1
    print("Plot_keys:", plot_keys)
    print("X keys:", x_keys)
    print("split_keys:", split_keys)
    print("group_keys:", group_keys)
    print("filters:", filters)
    print("exclusions:", exclusions)
    dfs = {}
    for split_selector, split_title in zip(split_selectors, split_titles):
        if custom_series_splitter is not None:
            exps = split_selector.extract()
            splitted_dict = dict()
            for exp in exps:
                key = custom_series_splitter(exp)
                if key not in splitted_dict:
                    splitted_dict[key] = list()
                splitted_dict[key].append(exp)
            splitted = list(splitted_dict.items())
            group_selectors = [core.Selector(list(x[1])) for x in splitted]
            group_legends = [x[0] for x in splitted]
        else:
            if len(group_keys) > 0:
                group_selectors, group_legends = frontend.split_by_keys(
                    split_selector, group_keys
                )
            else:
                group_selectors = [split_selector]
                group_legends = [split_title]
        list_of_list_of_plot_dicts = []
        group_df = pd.DataFrame()
        for group_selector, group_legend in zip(group_selectors, group_legends):
            filtered_data = group_selector.extract()
            for data in filtered_data:
                df = pd.DataFrame(data['progress'])
                for key in plot_keys:
                    temp = frontend.sliding_mean(df[key], 12)
                    df[f"smooth {key}"] = temp
                for column_name, value in group_selector._filters:
                    df[column_name] = value
                for column_name in data.flat_params:
                    value = data.flat_params[column_name]
                    if isinstance(value, numbers.Number) or isinstance(value, str):
                        df[column_name] = value
                group_df = pd.concat([group_df, df])
        dfs[split_title] = group_df

    return dfs[split_title]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_paths", type=str, nargs='*')
    parser.add_argument("--prefix", type=str, nargs='?', default="???")
    parser.add_argument("--disable-variant", default=False, action='store_true')
    parser.add_argument("--data-filename",
                        default='progress.csv',
                        help='name of data file.')
    parser.add_argument("--params-filename",
                        default='params.json',
                        help='name of params file.')
    args = parser.parse_args(sys.argv[1:])

    # load all folders following a prefix
    if args.prefix != "???":
        args.data_paths = []
        dirname = os.path.dirname(args.prefix)
        subdirprefix = os.path.basename(args.prefix)
        for subdirname in os.listdir(dirname):
            path = os.path.join(dirname, subdirname)
            if os.path.isdir(path) and (subdirprefix in subdirname):
                args.data_paths.append(path)
    print("Importing data from {path}...".format(path=args.data_paths))
    exps_data = core.load_exps_data(
        args.data_paths,
        args.data_filename,
        args.params_filename,
        args.disable_variant,
    )
    plottable_keys = list(
        set(frontend.flatten(list(exp.progress.keys()) for exp in exps_data)))
    plottable_keys = sorted([k for k in plottable_keys if k is not None])
    distinct_params = sorted(core.extract_distinct_params(exps_data))
    return get_plot_instruction([], exps_data, x_keys=[])


if __name__ == "__main__":
    results_data = main()
    results_data.insert(3, 'time', results_data['Total timing'] / 3600)
    augment_epoch = 300
    fine_tune_epoch = 310
    df = results_data.loc[np.logical_or(np.logical_and(results_data['epoch'] == augment_epoch,
                                                       results_data['dss_args.augment_queryset'] == True)
                                        , np.logical_and(results_data['epoch'] == fine_tune_epoch,
                                                         results_data['dss_args.fine_tune'] == True))]
    p = df[['exp_domains',
            'version',
            'Test Accuracy',
            'Total timing',
            'dss_args.fraction',
            'dss_args.fine_tune',
            'dss_args.augment_queryset',
            'Validation Loss',
            'Validation Accuracy',
            'Test Loss',
            'epoch',
            'Timing', 'exp_name', 'exp_prefix']]
    p.to_csv('DSS_results.csv', index=False)

    # Augment values
    augment_set = results_data.loc[np.logical_and(results_data['epoch'] == augment_epoch,
                                                  results_data['dss_args.augment_queryset'] == True)]
    s = augment_set.groupby(['exp_domains', 'version'])
    mean = s.mean()[['Test Accuracy', 'time']].round(2)
    std = s.std()[['Test Accuracy', 'time']].round(2)
    l = pd.merge(mean, std, on=["exp_domains", "version"], suffixes=('_mean', '_std'), how="inner")
    count = s.count()['exp_prefix'].round(2)
    k = pd.merge(l, count, on=["exp_domains", "version"], how="inner")
    k = k.rename(columns={'exp_prefix': 'count'})
    k.to_csv('augment_results.csv')

    # Fine tune

    fine_tune = results_data.loc[np.logical_and(results_data['dss_args.fine_tune'] == True,
                                                results_data['epoch'] == fine_tune_epoch)]
    d = fine_tune['ckpt.file'].str.split('/').str.get(3).to_frame()
    d = d.rename(columns={'ckpt.file': 'ckpt.prefix'})
    k = pd.concat([fine_tune, d], axis=1)
    pretrain = results_data.loc[np.logical_and(results_data['epoch'] == augment_epoch,
                                               results_data['dss_args.augment_queryset'] == False)]
    t = pd.merge(k, pretrain, how='left', left_on='ckpt.prefix', right_on='exp_name', suffixes=('', '_pretrain'))
    t['time'] = t['time_pretrain'] + t['time']
    fine_tune = t[['exp_domains', 'version', 'time', 'Test Accuracy', 'exp_name', 'ckpt.file']]
    s = fine_tune.groupby(['exp_domains', 'version'])
    mean = s.mean()[['Test Accuracy', 'time']].round(2)
    std = s.std()[['Test Accuracy', 'time']].round(2)
    l = pd.merge(mean, std, on=["exp_domains", "version"], suffixes=('_mean', '_std'), how="inner")
    count = s.count()['ckpt.file'].round(2)
    k = pd.merge(l, count, on=["exp_domains", "version"], how="inner")
    k = k.rename(columns={'ckpt.file': 'count'})
    k.to_csv('finetune_results.csv')
