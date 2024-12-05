import os
import sys
import torch
import copy
import json
import time
import logging
import numpy as np
import awkward as ak
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from tqdm import tqdm
from concurrent.futures.thread import ThreadPoolExecutor


from weaver.utils.nn.tools import train_classification as train
from weaver.utils.nn.tools import evaluate_classification as evaluate
from weaver.train import to_filelist

from weaver.utils.dataset import (
    _collate_awkward_array_fn,
    _finalize_inputs,
    _get_reweight_indices,
    _check_labels,
    _preprocess,
    _load_next
)
from weaver.utils.data.preprocess import AutoStandardizer, WeightMaker
from weaver.utils.data.config import DataConfig, _md5
from weaver.train import model_setup
from weaver.nn.model.ParticleTransformer import ParticleTransformer
from loguru import logger

training_mode = True

class _SimpleIter(object):
    r"""_SimpleIter

    Iterator object for ``SimpleIterDataset''.
    """

    def __init__(self, **kwargs):
        # inherit all properties from SimpleIterDataset
        self.__dict__.update(**kwargs)

        # executor to read files and run preprocessing asynchronously
        self.executor = ThreadPoolExecutor(max_workers=1) if self._async_load else None

        # init: prefetch holds table and indices for the next fetch
        self.prefetch = None
        self.table = None
        self.indices = []
        self.cursor = 0

        self._seed = None
        worker_info = torch.utils.data.get_worker_info()
        file_dict = copy.deepcopy(self._init_file_dict)
        if worker_info is not None:
            # in a worker process
            self._name += '_worker%d' % worker_info.id
            self._seed = worker_info.seed & 0xFFFFFFFF
            np.random.seed(self._seed)
            # split workload by files
            new_file_dict = {}
            for name, files in file_dict.items():
                new_files = files[worker_info.id::worker_info.num_workers]
                assert (len(new_files) > 0)
                new_file_dict[name] = new_files
            file_dict = new_file_dict
        self.worker_file_dict = file_dict
        self.worker_filelist = sum(file_dict.values(), [])
        self.worker_info = worker_info
        self.restart()

    def restart(self):
        logger.info('=== MODIFIED VERSION! Restarting DataIter %s, seed=%s ===' % (self._name, self._seed))
        # re-shuffle filelist and load range if for training
        filelist = copy.deepcopy(self.worker_filelist)
        logger.info("Copied file list.")
        if self._sampler_options['shuffle']:
            np.random.shuffle(filelist)
            logger.info("Shuffled data")
        if self._file_fraction < 1:
            num_files = int(len(filelist) * self._file_fraction)
            filelist = filelist[:num_files]
        self.filelist = filelist
        logger.info("Defined filelist")
        if self._init_load_range_and_fraction is None:
            logger.info("Setting load range...")
            self.load_range = (0, 1)
        else:
            logger.info("Setting load range...")

            (start_pos, end_pos), load_frac = self._init_load_range_and_fraction
            interval = (end_pos - start_pos) * load_frac
            if self._sampler_options['shuffle']:
                offset = np.random.uniform(start_pos, end_pos - interval)
                self.load_range = (offset, offset + interval)
            else:
                self.load_range = (start_pos, start_pos + interval)

        logger.debug(
            'Init iter [%d], will load %d (out of %d*%s=%d) files with load_range=%s:\n%s', 0
            if self.worker_info is None else self.worker_info.id, len(self.filelist),
            len(sum(self._init_file_dict.values(), [])),
            self._file_fraction, int(len(sum(self._init_file_dict.values(), [])) * self._file_fraction),
            str(self.load_range),
            '\n'.join(self.filelist[: 3]) + '\n ... ' + self.filelist[-1],)

        logger.info("Restarted DataIter")
        logger.debug('Restarted DataIter %s, load_range=%s, file_list:\n%s' %
                     (self._name, str(self.load_range), json.dumps(self.worker_file_dict, indent=2)))

        # reset file fetching cursor
        self.ipos = 0 if self._fetch_by_files else self.load_range[0]
        # prefetch the first entry asynchronously
        self._try_get_next(init=True)

    def __next__(self):
        # print(self.ipos, self.cursor)
        if len(self.filelist) == 0:
            raise StopIteration
        try:
            i = self.indices[self.cursor]
        except IndexError:
            # case 1: first entry, `self.indices` is still empty
            # case 2: running out of entries, `self.indices` is not empty
            while True:
                if self._in_memory and len(self.indices) > 0:
                    # only need to re-shuffle the indices, if this is not the first entry
                    if self._sampler_options['shuffle']:
                        np.random.shuffle(self.indices)
                    break
                if self.prefetch is None:
                    # reaching the end as prefetch got nothing
                    self.table = None
                    if self._async_load:
                        self.executor.shutdown(wait=False)
                    raise StopIteration
                # get result from prefetch
                if self._async_load:
                    self.table, self.indices = self.prefetch.result()
                else:
                    self.table, self.indices = self.prefetch
                # try to load the next ones asynchronously
#               self._try_get_next()
                # check if any entries are fetched (i.e., passing selection) -- if not, do another fetch
                if len(self.indices) > 0:
                    break
            # reset cursor
            self.cursor = 0
            i = self.indices[self.cursor]
        self.cursor += 1
        return self.get_data(i)

    def _try_get_next(self, init=False):
        end_of_list = self.ipos >= len(self.filelist) if self._fetch_by_files else self.ipos >= self.load_range[1]
        if end_of_list:
            if init:
                raise RuntimeError('Nothing to load for worker %d' %
                                   0 if self.worker_info is None else self.worker_info.id)
            if self._infinity_mode and not self._in_memory:
                # infinity mode: re-start
                self.restart()
                return
            else:
                # finite mode: set prefetch to None, exit
                self.prefetch = None
                return

        if self._fetch_by_files:
            filelist = self.filelist[int(self.ipos): int(self.ipos + self._fetch_step)]
            load_range = self.load_range
        else:
            filelist = self.filelist
            load_range = (self.ipos, min(self.ipos + self._fetch_step, self.load_range[1]))

        logger.info('Start fetching next batch, len(filelist)=%d, load_range=%s'%(len(filelist), load_range))
        if self._async_load:
            logger.info("Using async prefetch logic")
            self.prefetch = self.executor.submit(_load_next, self._data_config,
                                                 filelist, load_range, self._sampler_options)
        else:
            logger.info("Using sync prefetch logic")
            self.prefetch = _load_next(self._data_config, filelist, load_range, self._sampler_options)
        self.ipos += self._fetch_step

    def get_data(self, i):
        # inputs
        X = {k: copy.deepcopy(self.table['_' + k][i]) for k in self._data_config.input_names}
        # labels
        y = {k: copy.deepcopy(self.table[k][i]) for k in self._data_config.label_names}
        # observers / monitor variables
        Z = {k: copy.deepcopy(self.table[k][i]) for k in self._data_config.z_variables}
        return X, y, Z


class SimpleIterDataset(torch.utils.data.IterableDataset):
    r"""Base IterableDataset.

    Handles dataloading.

    Arguments:
        file_dict (dict): dictionary of lists of files to be loaded.
        data_config_file (str): YAML file containing data format information.
        for_training (bool): flag indicating whether the dataset is used for training or testing.
            When set to ``True``, will enable shuffling and sampling-based reweighting.
            When set to ``False``, will disable shuffling and reweighting, but will load the observer variables.
        load_range_and_fraction (tuple of tuples, ``((start_pos, end_pos), load_frac)``): fractional range of events to load from each file.
            E.g., setting load_range_and_fraction=((0, 0.8), 0.5) will randomly load 50% out of the first 80% events from each file (so load 50%*80% = 40% of the file).
        fetch_by_files (bool): flag to control how events are retrieved each time we fetch data from disk.
            When set to ``True``, will read only a small number (set by ``fetch_step``) of files each time, but load all the events in these files.
            When set to ``False``, will read from all input files, but load only a small fraction (set by ``fetch_step``) of events each time.
            Default is ``False``, which results in a more uniform sample distribution but reduces the data loading speed.
        fetch_step (float or int): fraction of events (when ``fetch_by_files=False``) or number of files (when ``fetch_by_files=True``) to load each time we fetch data from disk.
            Event shuffling and reweighting (sampling) is performed each time after we fetch data.
            So set this to a large enough value to avoid getting an imbalanced minibatch (due to reweighting/sampling), especially when ``fetch_by_files`` set to ``True``.
            Will load all events (files) at once if set to non-positive value.
        file_fraction (float): fraction of files to load.
    """

    def __init__(self, file_dict, data_config_file,
                 for_training=True, load_range_and_fraction=None, extra_selection=None,
                 fetch_by_files=False, fetch_step=0.01, file_fraction=1, remake_weights=False, up_sample=True,
                 weight_scale=1, max_resample=10, async_load=True, infinity_mode=False, in_memory=False, name=''):
        self._iters = {} if infinity_mode or in_memory else None
        _init_args = set(self.__dict__.keys())
        self._init_file_dict = file_dict
        self._init_load_range_and_fraction = load_range_and_fraction
        self._fetch_by_files = fetch_by_files
        self._fetch_step = fetch_step
        self._file_fraction = file_fraction
        self._async_load = async_load
        self._infinity_mode = infinity_mode
        self._in_memory = in_memory
        self._name = name

        # ==== sampling parameters ====
        self._sampler_options = {
            'up_sample': up_sample,
            'weight_scale': weight_scale,
            'max_resample': max_resample,
        }

        # ==== torch collate_fn map ====
        from torch.utils.data._utils.collate import default_collate_fn_map
        default_collate_fn_map.update({ak.Array: _collate_awkward_array_fn})

        if for_training:
            self._sampler_options.update(training=True, shuffle=True, reweight=True)
        else:
            self._sampler_options.update(training=False, shuffle=False, reweight=False)

        # discover auto-generated reweight file
        if '.auto.yaml' in data_config_file:
            data_config_autogen_file = data_config_file
        else:
            data_config_md5 = _md5(data_config_file)
            data_config_autogen_file = data_config_file.replace('.yaml', '.%s.auto.yaml' % data_config_md5)
            if os.path.exists(data_config_autogen_file):
                data_config_file = data_config_autogen_file
                logger.info('Found file %s w/ auto-generated preprocessing information, will use that instead!' %
                             data_config_file)

        # load data config (w/ observers now -- so they will be included in the auto-generated yaml)
        self._data_config = DataConfig.load(data_config_file)

        if for_training:
            # produce variable standardization info if needed
            if self._data_config._missing_standardization_info:
                s = AutoStandardizer(file_dict, self._data_config)
                self._data_config = s.produce(data_config_autogen_file)

            # produce reweight info if needed
            if self._sampler_options['reweight'] and self._data_config.weight_name and not self._data_config.use_precomputed_weights:
                if remake_weights or self._data_config.reweight_hists is None:
                    w = WeightMaker(file_dict, self._data_config)
                    self._data_config = w.produce(data_config_autogen_file)

            # reload data_config w/o observers for training
            if os.path.exists(data_config_autogen_file) and data_config_file != data_config_autogen_file:
                data_config_file = data_config_autogen_file
                logger.info(
                    'Found file %s w/ auto-generated preprocessing information, will use that instead!' %
                    data_config_file)
            self._data_config = DataConfig.load(data_config_file, load_observers=False, extra_selection=extra_selection)
        else:
            self._data_config = DataConfig.load(
                data_config_file, load_reweight_info=False, extra_test_selection=extra_selection)

        # derive all variables added to self.__dict__
        self._init_args = set(self.__dict__.keys()) - _init_args

    @property
    def config(self):
        return self._data_config

    def __iter__(self):
        if self._iters is None:
            kwargs = {k: copy.deepcopy(self.__dict__[k]) for k in self._init_args}
            return _SimpleIter(**kwargs)
        else:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            try:
                return self._iters[worker_id]
            except KeyError:
                kwargs = {k: copy.deepcopy(self.__dict__[k]) for k in self._init_args}
                self._iters[worker_id] = _SimpleIter(**kwargs)
                return self._iters[worker_id]


@dataclass
class ArgumentsObject:
    data_val: list[str]
    data_train: list[str]
    data_test: list[str]
    local_rank: int | None = None
    copy_inputs: bool = False
    data_config: str = "data/JetClass/JetClass_full.yaml"
    network_config: str = "networks/example_ParticleTransformer.py"
    network_option: dict = field(default_factory=lambda: {})
    use_amp: bool = True
    model_prefix: str = "training/JetClass/Pythia/full/ParT/{auto}/net"
    export_onnx: bool = False
    load_model_weights: bool = False
    num_workers: int = 0
    fetch_step: float = 0.01
    batch_size: int = 512
    start_lr: float = 1e-3
    samples_per_epoch: int = 513
    samples_per_epoch_val: int = 513
    num_epochs: int = 1
    gpus: str = ""
    optimizer: str = "ranger"
    log: str = "logs/JetClass_Pythia_full_ParT_{auto}.log"
    predict_output: str = "pred.root"
    tensorboard: str = "JetClass_Pythia_full_ParT"



def get_datasets(args: ArgumentsObject):
    

    train_file_dict, _train_files = to_filelist(args, "train")

    val_file_dict, _val_files = to_filelist(args, "val")

    train_range = val_range = (0, 1)
    
    train_data = SimpleIterDataset(train_file_dict, args.data_config, for_training=True, extra_selection=None, remake_weights=True, load_range_and_fraction=(train_range, 1), file_fraction=1, fetch_by_files=False, fetch_step=0.01, infinity_mode=False, in_memory=False, name="train" + (""))
    val_data = SimpleIterDataset(val_file_dict, args.data_config, for_training=True, extra_selection=None, load_range_and_fraction=(val_range, 1), file_fraction=1, fetch_by_files=False, fetch_step=0.01, infinity_mode=False, in_memory=False, name="val" + (""))

    data_config = train_data.config

    return train_data, val_data, data_config

def get_dataloaders(train_data: SimpleIterDataset, val_data: SimpleIterDataset, args: ArgumentsObject):

    train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=0, persistent_workers=False)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, drop_last=True, pin_memory=True, num_workers=0, persistent_workers=False)


    return train_loader, val_loader

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logger.opt(depth=6, exception=record.exc_info).log(record.levelno, record.getMessage())

def set_up_logging():
    """Redirects the logging to the loguru logger."""

    logger.remove() 
    logger.add(sys.stdout, level="INFO")
    logging.basicConfig(handlers=[InterceptHandler()], level="INFO")

def get_first_n_batches(n: int, train_loader: DataLoader) -> list[torch.Tensor]:
    """Get the first n batches from the train_loader.

    Args:
        n (int): Number of batches to return.
        train_loader (DataLoader): Pytorch DataLoader object.

    Returns:
        list[torch.Tensor]: List of tensors containing the first n batches.
    """

    start_time = time.time()

    batch_list = []
    
    
    for i, batch in enumerate(train_loader):
        if i==0:
            first_batch_loaded_time = time.time()
            elapsed_time = first_batch_loaded_time - start_time
            batch_length = len(batch)
            pf_points_shape = batch[0]["pf_points"].shape
            pf_features_shape = batch[0]["pf_features"].shape
            pf_vectors_shape = batch[0]["pf_vectors"].shape
            pf_mask_shape = batch[0]["pf_mask"].shape

            logger.debug(f"{batch[0].keys()}")
            logger.debug(f"{batch_length=}, {pf_points_shape=}, {pf_features_shape=}, {pf_vectors_shape=}, {pf_mask_shape=}")
            logger.debug(f"Single batch loading time {elapsed_time:.2f}s")
        if i<n:
            batch_list.append(batch)
        else:
            break
    return batch_list

def call_forward_pass(model: ParticleTransformer, training_input: DataLoader, data_config: dict, dev: torch.device, steps_per_epoch: int) -> None:
    """Call the forward pass of the model.

    Args:
        model (ParticleTransformer): ParticleTransformer model.
        training_input (DataLoader): Pytorch DataLoader object.
        data_config (dict): Data configuration dictionary.
        dev (torch.device): Pytorch device to run the model on.
    """
    num_batches = 0
    for X, y, _ in tqdm(training_input):
        inputs = [X[k].to(dev) for k in data_config.input_names]
        label = y[data_config.label_names[0]].long().to(dev)
        try:
            mask = y[data_config.label_names[0] + "_mask"].bool().to(dev)
        except KeyError:
            mask = None
        
        logger.info(f"inputs: {inputs}\nlabels: {label}\nmask: {mask}")

        profile_forward_pass(model, inputs, dev)
        num_batches += 1

        if num_batches >= steps_per_epoch:
            input_shapes = [k.shape for k in inputs]
            logger.info(f"Processed {num_batches} batches of inputs with shape {input_shapes}")
            break

def profile_forward_pass(model: ParticleTransformer, inputs: list[torch.Tensor], dev: torch.device) -> None:
    """Profile the forward pass of the model.

    Args:
        model (ParticleTransformer): The model to profile.
        inputs (list[torch.Tensor]): List of input tensors.
    """
    # turn off gradient accumulation...
    model.eval()
    if dev.type == "cuda":
        sort_by_keyword = "self_" + dev.type + "_time_total"
    else:
        sort_by_keyword = "self_cpu_time_total"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        with_flops=True,
        profile_memory=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("tensorboard_logs")
    ) as prof:
        
        _outputs = model(*inputs)
    logger.debug(f"Outputs shape: {_outputs.shape}")
    results = prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10)

    logger.info(f"{results}")
    
    #prof.export_chrome_trace("./chrome_traces")



def main():
    """Main entry point of the script."""
   
    args_val = ["datasets/JetClass/Pythia/val_5M/*.root"]
    args_train = ['HToBB:datasets/JetClass/Pythia/train_100M/HToBB_*.root', 'HToCC:datasets/JetClass/Pythia/train_100M/HToCC_*.root', 'HToGG:datasets/JetClass/Pythia/train_100M/HToGG_*.root', 'HToWW2Q1L:datasets/JetClass/Pythia/train_100M/HToWW2Q1L_*.root', 'HToWW4Q:datasets/JetClass/Pythia/train_100M/HToWW4Q_*.root', 'TTBar:datasets/JetClass/Pythia/train_100M/TTBar_*.root', 'TTBarLep:datasets/JetClass/Pythia/train_100M/TTBarLep_*.root', 'WToQQ:datasets/JetClass/Pythia/train_100M/WToQQ_*.root', 'ZToQQ:datasets/JetClass/Pythia/train_100M/ZToQQ_*.root', 'ZJetsToNuNu:datasets/JetClass/Pythia/train_100M/ZJetsToNuNu_*.root']
    args_test = ['HToBB:datasets/JetClass/Pythia/test_20M/HToBB_*.root' 'HToCC:datasets/JetClass/Pythia/test_20M/HToCC_*.root' 'HToGG:datasets/JetClass/Pythia/test_20M/HToGG_*.root' 'HToWW2Q1L:datasets/JetClass/Pythia/test_20M/HToWW2Q1L_*.root' 'HToWW4Q:datasets/JetClass/Pythia/test_20M/HToWW4Q_*.root' 'TTBar:datasets/JetClass/Pythia/test_20M/TTBar_*.root' 'TTBarLep:datasets/JetClass/Pythia/test_20M/TTBarLep_*.root' 'WToQQ:datasets/JetClass/Pythia/test_20M/WToQQ_*.root' 'ZToQQ:datasets/JetClass/Pythia/test_20M/ZToQQ_*.root' 'ZJetsToNuNu:datasets/JetClass/Pythia/test_20M/ZJetsToNuNu_*.root']
    
    args = ArgumentsObject(args_val, args_train, args_test)
    #dev = torch.device(0) if args.gpus == "0" else torch.device("cpu")
    
    dev = torch.device(0) if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Using device {dev}")

    set_up_logging()
    train_data, val_data, data_config = get_datasets(args)

    train_loader, val_loader = get_dataloaders(train_data, val_data, args)


    n_batch_list = get_first_n_batches(n=2, train_loader=train_loader)
        
    model, model_info, _loss_func = model_setup(args, data_config, device=dev)
    model.to(dev)
    logger.info(f"{model_info=}")
    

    steps_per_epoch = args.samples_per_epoch // args.batch_size
    _steps_per_epoch_val = args.samples_per_epoch_val // args.batch_size


    # Runs over a single epoch
    call_forward_pass(model, train_loader, data_config, dev, steps_per_epoch)

    return args, data_config, train_loader, val_loader, n_batch_list
if __name__ == "__main__":
    main()
