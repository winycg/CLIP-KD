import ast
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from multiprocessing import Value

import braceexpand
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
import webdataset as wds
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import tokenize
from open_clip.tokenizer import HFTokenizer
from .imagenet_zeroshot_data import openai_imagenet_template
from .class_sampler import MPerClassSampler

class CsvDataset(Dataset):
    def __init__(self, data_root, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.root = data_root
        
        logging.debug('Done loading data.')

        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, str(self.images[idx]))
        image_file = Image.open(img_path)
        images = self.transforms(image_file)
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


class ImageNetCsvDataset(Dataset):
    def __init__(self, data_root, input_filename, transforms, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.class_name = df['class'].tolist()

        self.labels = []
        count = 0
        start_label =  self.class_name[0]
        for c in self.class_name:
            if c == start_label:
                self.labels.append(count)
            else:
                count += 1
                self.labels.append(count)
                start_label = c
        self.labels = np.array(self.labels)
        self.imagenet_templates = openai_imagenet_template
        self.imagenet_templates_num = len(openai_imagenet_template)
        self.transforms = transforms
        logging.debug('Done loading data.')

        self.root = data_root
        self.tokenize = tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.images[idx])
        definition = "The definition of %s is %s"%(self.class_name[idx], self.captions[idx])
        caption = self.imagenet_templates[np.random.choice(self.imagenet_templates_num)](self.class_name[idx])+ " "+ definition
        images = self.transforms(Image.open(str(img_path)))
        texts = self.tokenize([caption])[0]
        return images, texts

class RetrievalDataset(Dataset):
    """ Image/Text Retrieval Dataset"""
    def __init__(self, transform, data_path='.', tokenizer=None):
        super(RetrievalDataset, self).__init__()
        self.data_path = data_path
        caption_file = os.path.join(self.data_path, 'test_captions.pt')
        self.captions = torch.load(caption_file)
        self.tokenizer = tokenizer
        eval_img_keys_file = 'test_img_keys.tsv'
        with open(os.path.join(self.data_path, eval_img_keys_file), 'r') as f:
            img_keys = f.readlines()
        self.img_keys = [int(k.strip()) for k in img_keys]
        self.captions = {k: self.captions[k] for k in self.img_keys}
        if not type(self.captions[self.img_keys[0]]) == list:
            self.captions = {k: json.loads(self.captions[k]) for k in self.img_keys}

        self.transform = transform
        self.prompt_list = ["{}"] # ["a photo of {}"]
        
    def __getitem__(self, index):
        img_key = self.img_keys[index]
        if 'coco' in self.data_path:
            image = Image.open(os.path.join(self.data_path, 'images/val2014/', 'COCO_val2014_{:012}.jpg'.format(img_key)))
        else:
            image = Image.open(os.path.join(self.data_path, 'images/{}.jpg'.format(img_key)))
        image = self.transform(image)

        captions = self.captions[img_key]
        # if len(captions) > 5:
        #     random.shuffle(captions)
        #     captions = captions[:5]
        tokenized_caps = []
        for sentence in captions:
            sentence_tokenized = []
            for p in self.prompt_list:
                sentence_tokenized.append(p.format(sentence.strip()))
            tokenized_caps.extend(sentence_tokenized)
        texts = self.tokenizer(tokenized_caps)[0]
        return image, texts

    def __len__(self):
        return len(self.img_keys)
    

class MultiTaskDataLoader(object):
    """
    Multi-task DataLoader, the first dataloader is master dataloader
    """

    def __init__(self,
                 loaders, seed=0):
        assert len(loaders) > 1, "Less than 2 loader!"
        self.loaders = loaders
        self.iters = [iter(loader) for loader in loaders]
        self.lens = [len(loader) for loader in loaders]
        self.global_idx_in_cycle = 0
        self.seed = seed

    def __iter__(self):
        if self.global_idx_in_cycle > 0:
            self.iters[0] = iter(self.loaders[0])
        return self

    def __next__(self):
        output_tuple = (*next(self.iters[0]),)
        for k, (loader, _iter) in enumerate(zip(self.loaders[1:], self.iters[1:])):
            try:
                output_tuple += (*next(_iter),)
            except StopIteration:
                try:
                    loader.batch_sampler.sampler.set_epoch(int(self.global_idx_in_cycle // self.lens[k + 1]))
                except:
                    pass
                _iter = iter(loader)
                self.iters[k + 1] = _iter
                output_tuple += (*next(_iter),)

        if self.global_idx_in_cycle < sys.maxsize - 1:
            self.global_idx_in_cycle += 1
        else:
            self.global_idx_in_cycle = 0
        return output_tuple

    def __len__(self):
        return self.lens[0]


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_size(shards):
    shards_list = list(braceexpand.braceexpand(shards))
    dir_path = os.path.dirname(shards)
    sizes_filename = os.path.join(dir_path, 'sizes.json')
    len_filename = os.path.join(dir_path, '__len__')
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, 'r'))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    elif os.path.exists(len_filename):
        # FIXME this used to be eval(open(...)) but that seemed rather unsafe
        total_size = ast.literal_eval(open(len_filename, 'r').read())
    else:
        total_size = None  # num samples undefined
        # some common dataset sizes (at time of authors last download)
        # CC3M (train): 2905954
        # CC12M: 10968539
        # LAION-400M: 407332084
        # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def get_imagenet(args, data_path, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    
    dataset = datasets.ImageFolder(data_path, transform=preprocess_val)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        sampler=None,
        drop_last=False
    )

    return DataInfo(dataloader=dataloader, sampler=None)


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches


def filter_no_caption_or_no_image(sample):
    return ('txt' in sample) and ('png' in sample or 'jpg' in sample)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed() + epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.

        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed(self.worker_seed() + epoch)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_samples, num_shards = get_dataset_size(input_shards)
    if not num_samples:
        if is_train:
            num_samples = args.train_num_samples
            if not num_samples:
                raise RuntimeError(
                    'Currently, number of dataset samples must be specified for training dataset. '
                    'Please specify via `--train-num-samples` if no dataset length info present.')
        else:
            num_samples = args.val_num_samples or 0  # eval will just exhaust the iterator if not specified

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train),
    ])

    dataset = wds.DataPipeline(*pipeline)
    if is_train:
        if not resampled:
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=True,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_icar(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    train_dataset_names = input_filename.split(',')
    train_datasets = []
    data_root_paths = args.data_root.split(',')
    sup_dataset = ImageNetCsvDataset(
        data_root_paths[0],
        train_dataset_names[0],
        preprocess_fn,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        sep=args.csv_separator,
        tokenizer=tokenizer)
    
    vl_dataset = CsvDataset(
                data_root_paths[1],
                train_dataset_names[1],
                preprocess_fn,
                img_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                sep=args.csv_separator,
                tokenizer=tokenizer)
 
         
    num_samples = len(vl_dataset) + len(sup_dataset)
    sup_sampler = MPerClassSampler(labels=sup_dataset.labels, 
                                   m=args.num_per_class, batch_size=args.batch_size, 
                                   length_before_new_iter=len(sup_dataset.labels)//args.world_size)
    #sup_sampler = DistributedSampler(sup_dataset) if args.distributed and is_train else None
    vl_sampler = DistributedSampler(vl_dataset) if args.distributed and is_train else None

    sup_dataloader = DataLoader(
        sup_dataset,
        batch_size=args.sup_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sup_sampler,
        drop_last=is_train,
    )
    
    vl_dataloader = DataLoader(
        vl_dataset,
        batch_size=args.vl_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=vl_sampler,
        drop_last=is_train,
    )
    
    dataloader = MultiTaskDataLoader([sup_dataloader, vl_dataloader], seed=0)
    dataloader.num_samples = len(vl_dataset) + len(sup_dataset)
    dataloader.num_batches = len(dataloader)
    return DataInfo(dataloader, sup_sampler)




def get_vl_imagenet(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data 
    train_dataset_names = input_filename.split(',')
    train_datasets = []
    data_root_paths = args.data_root.split(',')
       
    sup_dataset = datasets.ImageFolder(os.path.join(data_root_paths[0], 'train'),
                                       transform=preprocess_fn)

    sup_sampler = DistributedSampler(sup_dataset)
    sup_dataloader = DataLoader(
        sup_dataset,
        batch_size=args.sup_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sup_sampler,
        drop_last=is_train,
    )
    
    vl_dataset = CsvDataset(
                data_root_paths[1],
                train_dataset_names[1],
                preprocess_fn,
                img_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                sep=args.csv_separator,
                tokenizer=tokenizer)

    vl_sampler = DistributedSampler(vl_dataset) if args.distributed and is_train else None

    
    vl_dataloader = DataLoader(
        vl_dataset,
        batch_size=args.vl_batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=vl_sampler,
        drop_last=is_train,
    )
    
    sup_dataloader.num_samples = len(sup_dataset)
    sup_dataloader.num_batches = len(sup_dataloader)
    return DataInfo(sup_dataloader, sup_sampler), DataInfo(sup_dataloader, sup_sampler), DataInfo(vl_dataloader, vl_sampler) 


def get_csv_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    data_root = args.data_root if is_train else args.val_data_root
    if 'imagenet' in input_filename:
        dataset = ImageNetCsvDataset(
            data_root,
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer)
    elif 'coco' in input_filename or 'flickr' in input_filename: 
        dataset = RetrievalDataset(
            preprocess_fn, data_path=input_filename, tokenizer=tokenizer
        )
    else:
        dataset = CsvDataset(
            data_root,
            input_filename,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



def get_csv_multi_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    train_dataset_names = input_filename.split(',')
    data_root = args.data_root.split(',')
    train_datasets = []
    for idx, name in enumerate(train_dataset_names):
        if 'imagenet' in name:
            dataset = ImageNetCsvDataset(
            data_root[idx],
            name,
            preprocess_fn,
            img_key=args.csv_img_key,
            caption_key=args.csv_caption_key,
            sep=args.csv_separator,
            tokenizer=tokenizer)
        else:
            dataset = CsvDataset(
                data_root[idx],
                name,
                preprocess_fn,
                img_key=args.csv_img_key,
                caption_key=args.csv_caption_key,
                sep=args.csv_separator,
                tokenizer=tokenizer)
        train_datasets.append(dataset)
         
    dataset = torch.utils.data.ConcatDataset(train_datasets)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)



class SyntheticDataset(Dataset):

    def __init__(self, transform=None, image_size=(224, 224), caption="Dummy caption", dataset_size=100, tokenizer=None):
        self.transform = transform
        self.image_size = image_size
        self.caption = caption
        self.image = Image.new('RGB', image_size)
        self.dataset_size = dataset_size

        self.preprocess_txt = lambda text: tokenizer(text)[0]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.transform(self.image)
        return image, self.preprocess_txt(self.caption)


def get_synthetic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    image_size = preprocess_fn.transforms[0].size
    dataset = SyntheticDataset(
        transform=preprocess_fn, image_size=image_size, dataset_size=args.train_num_samples, tokenizer=tokenizer)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == 'icar':
        return get_icar
    elif dataset_type == "vl_imagenet":
        return get_vl_imagenet
    elif dataset_type == "csv":
        if ',' in data_path:
            return get_csv_multi_dataset
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extention {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    

def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "icar":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.train_data and args.dataset_type == "vl_imagenet":
        dataloader, sup_dataloader, vl_dataloader = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)
        data["train"] = dataloader
        data["train_loaders"] = (sup_dataloader, vl_dataloader)
        
    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.val_dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, args.imagenet_val, preprocess_fns)

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, args.imagenet_v2, preprocess_fns)
    
    if args.imagenet_r is not None:
        data["imagenet-r"] = get_imagenet(args, args.imagenet_r, preprocess_fns)  
        
    if args.imagenet_a is not None:
        data["imagenet-a"] = get_imagenet(args, args.imagenet_a, preprocess_fns)  
        
    if args.imagenet_sketch is not None: 
        data["imagenet-sketch"] = get_imagenet(args, args.imagenet_sketch, preprocess_fns)  
    return data
