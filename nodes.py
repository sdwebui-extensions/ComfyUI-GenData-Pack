import os
import glob
import re
import hashlib
from datetime import datetime
import json
import piexif
import piexif.helper
import comfy.sd
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo
import numpy as np
import folder_paths
import comfy.sd
import math
from nodes import MAX_RESOLUTION
from pathlib import Path

MANIFEST = {
    "name": "Lux Nodes",
    "version": (1, 0, 0),
    "author": "LuxFX",
    "project": "",
    "description": "Suite of nodes focused on quick evaluation of prompts and checkpoints",
}

BAKED_VAE = 'Baked VAE'

#! SYSTEM HOOKS


class cstr(str):
    class color:
        END = '\33[0m'
        BOLD = '\33[1m'
        ITALIC = '\33[3m'
        UNDERLINE = '\33[4m'
        BLINK = '\33[5m'
        BLINK2 = '\33[6m'
        SELECTED = '\33[7m'

        BLACK = '\33[30m'
        RED = '\33[31m'
        GREEN = '\33[32m'
        YELLOW = '\33[33m'
        BLUE = '\33[34m'
        VIOLET = '\33[35m'
        BEIGE = '\33[36m'
        WHITE = '\33[37m'

        BLACKBG = '\33[40m'
        REDBG = '\33[41m'
        GREENBG = '\33[42m'
        YELLOWBG = '\33[43m'
        BLUEBG = '\33[44m'
        VIOLETBG = '\33[45m'
        BEIGEBG = '\33[46m'
        WHITEBG = '\33[47m'

        GREY = '\33[90m'
        LIGHTRED = '\33[91m'
        LIGHTGREEN = '\33[92m'
        LIGHTYELLOW = '\33[93m'
        LIGHTBLUE = '\33[94m'
        LIGHTVIOLET = '\33[95m'
        LIGHTBEIGE = '\33[96m'
        LIGHTWHITE = '\33[97m'

        GREYBG = '\33[100m'
        LIGHTREDBG = '\33[101m'
        LIGHTGREENBG = '\33[102m'
        LIGHTYELLOWBG = '\33[103m'
        LIGHTBLUEBG = '\33[104m'
        LIGHTVIOLETBG = '\33[105m'
        LIGHTBEIGEBG = '\33[106m'
        LIGHTWHITEBG = '\33[107m'

        @staticmethod
        def add_code(name, code):
            if not hasattr(cstr.color, name.upper()):
                setattr(cstr.color, name.upper(), code)
            else:
                raise ValueError(
                    f"'cstr' object already contains a code with the name '{name}'.")

    def __new__(cls, text):
        return super().__new__(cls, text)

    def __getattr__(self, attr):
        if attr.lower().startswith("_cstr"):
            code = getattr(self.color, attr.upper().lstrip("_cstr"))
            modified_text = self.replace(f"__{attr[1:]}__", f"{code}")
            return cstr(modified_text)
        elif attr.upper() in dir(self.color):
            code = getattr(self.color, attr.upper())
            modified_text = f"{code}{self}{self.color.END}"
            return cstr(modified_text)
        elif attr.lower() in dir(cstr):
            return getattr(cstr, attr.lower())
        else:
            raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)


#! MESSAGE TEMPLATES
cstr.color.add_code(
    "msg", f"{cstr.color.BLUE}Lux Nodes: {cstr.color.END}")
cstr.color.add_code(
    "warning", f"{cstr.color.BLUE}Lux Nodes {cstr.color.LIGHTYELLOW}Warning: {cstr.color.END}")
cstr.color.add_code(
    "error", f"{cstr.color.RED}Lux Nodes {cstr.color.END}Error: {cstr.color.END}")

#! GLOBALS
NODE_FILE = os.path.abspath(__file__)
MODELS_DIR = folder_paths.models_dir
LUX_NODES_ROOT = os.path.dirname(NODE_FILE)
LUX_DATABASE = os.path.join(LUX_NODES_ROOT, 'lux_nodes_settings.json')

# WAS SETTINGS MANAGER


class WASDatabase:
    """
    The WAS Suite Database Class provides a simple key-value database that stores
    data in a flatfile using the JSON format. Each key-value pair is associated with
    a category.

    Attributes:
        filepath (str): The path to the JSON file where the data is stored.
        data (dict): The dictionary that holds the data read from the JSON file.

    Methods:
        insert(category, key, value): Inserts a key-value pair into the database
            under the specified category.
        get(category, key): Retrieves the value associated with the specified
            key and category from the database.
        update(category, key): Update a value associated with the specified
            key and category from the database.
        delete(category, key): Deletes the key-value pair associated with the
            specified key and category from the database.
        _save(): Saves the current state of the database to the JSON file.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        try:
            with open(filepath, 'r') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            self.data = {}

    def catExists(self, category):
        return category in self.data

    def keyExists(self, category, key):
        return category in self.data and key in self.data[category]

    def insert(self, category, key, value):
        if not isinstance(category, str) or not isinstance(key, str):
            cstr("Category and key must be strings").error.print()
            return

        if category not in self.data:
            self.data[category] = {}
        self.data[category][key] = value
        self._save()

    def update(self, category, key, value):
        if category in self.data and key in self.data[category]:
            self.data[category][key] = value
            self._save()

    def updateCat(self, category, dictionary):
        self.data[category].update(dictionary)
        self._save()

    def get(self, category, key):
        return self.data.get(category, {}).get(key, None)

    def getDB(self):
        return self.data

    def insertCat(self, category):
        if not isinstance(category, str):
            cstr("Category must be a string").error.print()
            return

        if category in self.data:
            cstr(
                f"The database category '{category}' already exists!").error.print()
            return
        self.data[category] = {}
        self._save()

    def getDict(self, category):
        if category not in self.data:
            cstr(
                f"The database category '{category}' does not exist!").error.print()
            return {}
        return self.data[category]

    def delete(self, category, key):
        if category in self.data and key in self.data[category]:
            del self.data[category][key]
            self._save()

    def _save(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.data, f, indent=4)
        except FileNotFoundError:
            cstr(f"Cannot save database to file '{self.filepath}'. "
                 "Storing the data in the object instead. Does the folder and node file have write permissions?").warning.print()
        except Exception as e:
            cstr(f"Error while saving JSON data: {e}").error.print()


WDB = WASDatabase(LUX_DATABASE)


def parse_name(ckpt_name):
    path = ckpt_name
    filename = path.split("/")[-1]
    filename = filename.split(".")[:-1]
    filename = ".".join(filename)
    return filename


def parse_simple_name(ckpt_name):
    return Path(ckpt_name).stem


def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read the file in chunks to avoid loading the entire file into memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def handle_whitespace(string: str):
    return string.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


def get_timestamp(time_format):
    now = datetime.now()
    try:
        timestamp = now.strftime(time_format)
    except:
        timestamp = now.strftime("%Y-%m-%d-%H%M%S")

    return timestamp


def make_pathname(filename, seed, ckpt_name, counter, time_format, extra, prompt_name):
    filename = filename.replace("%date", get_timestamp("%Y-%m-%d"))
    filename = filename.replace("%time", get_timestamp(time_format))
    filename = filename.replace("%model", ckpt_name)
    filename = filename.replace("%ckpt", ckpt_name)
    filename = filename.replace("%seed", str(seed))
    filename = filename.replace("%counter", str(counter))
    filename = filename.replace("%extra", str(extra))
    filename = filename.replace("%promptname", str(prompt_name))
    return filename


def make_filename(filename, seed, ckpt_name, counter, time_format, extra, prompt_name):
    filename = make_pathname(filename, seed, ckpt_name,
                             counter, time_format, extra, prompt_name)

    return get_timestamp(time_format) if filename == "" else filename


def populate_items(names, type):
    for idx, item_name in enumerate(names):

        file_name = os.path.splitext(item_name)[0]
        file_path = folder_paths.get_full_path(type, item_name)

        if file_path is None:
            print(
                f"(pysssss:better_combos) Unable to get path for {type} {item_name}")
            continue

        file_path_no_ext = os.path.splitext(file_path)[0]

        for ext in ["png", "jpg", "jpeg", "preview.png"]:
            has_image = os.path.isfile(file_path_no_ext + "." + ext)
            if has_image:
                item_image = f"{file_name}.{ext}"
                break

        names[idx] = {
            "content": item_name,
            "image": f"{type}/{item_image}" if has_image else None,
        }
    names.sort(key=lambda i: i["content"].lower())


def parseSamplerScheduler(sampler_name, default_sampler, default_scheduler):
    samp = ''
    sched = ''

    if 'karras' in sampler_name:
        sched = 'karras'
    elif 'exponential' in sampler_name:
        sched = 'exponential'
    else:
        sched = default_scheduler

    sampler_equivalents = {
        'dpm++ 2m karras': 'dpmpp_2m',
        'dpm++ sde karras': 'dpmpp_sde',
        'dpm++ 2m sde exponential': 'dpmpp_2m_sde',
        'dpm++ 2m sde karras': 'dpmpp_2m_sde',
        'euler a': 'euler_ancestral',
        'euler': 'euler',
        'lms': 'lms',
        'heun': 'heun',
        'dpm2': 'dpm_2',
        'dpm2 a': 'dpm_2_ancestral',
        'dpm++ 2s a': 'dpmpp_2s_ancestral',
        'dpm++ 2m': 'dpmpp_2m',
        'dpm++ sde': 'dpmpp_sde',
        'dpm++ 2m sde': 'dpmpp_2m_sde',
        'dpm++ 2m sde heun': 'heunpp2',  # are these the right equivalent?
        'dpm++ 2m sde heun karras': 'heunpp2',
        'dpm++ 2m sde heun exponential': 'heunpp2',
        'dpm++ 3m sde': 'dpmpp_3m_sde',
        'dpm++ 3m sde karras': 'dpmpp_3m_sde',
        'dpm++ 3m sde exponential': 'dpmpp_3m_sde',
        'dpm fast': 'dpm_fast',
        'dpm adaptive': 'dpm_adaptive',
        'lms karras': 'lms',
        'dpm2 karras': 'dpm_2',
        'dpm2 a karras': 'dpm_2_ancestral',
        'dpm++ 2s a karras': 'dpm_2s_ancestral',
        'restart': '',
        'ddim': 'ddim',
        'plms': '',
        'unipc': 'uni_pc',
    }

    if sampler_name in sampler_equivalents:
        samp = sampler_equivalents[sampler_name]
    else:
        samp = default_sampler
        sched = default_scheduler

    return [samp, sched, sampler_name]


def get_file_path_by_type(pathtype, filename, addDot=True, fullpath=True):
    if filename == '' or filename.strip() == '':
        if pathtype == 'vae':
            return BAKED_VAE

        return filename

    filename = filename.strip()

    if addDot and '.' not in filename:
        filename += '.'

    type_files = folder_paths.get_filename_list(pathtype)
    path = filename

    for f in type_files:
        if Path(f).name.startswith(filename) or f.startswith(filename):
            path = folder_paths.get_full_path(pathtype, f) if fullpath else f
            break

    return path


class GenData:
    def __init__(self, prompt=None):
        self.bundle = {
            "positive": '',
            "negative": '',
            "seed": -1,
            "steps": 20,
            "cfg": 7.0,
            "width": 512,
            "height": 512,
            "clip_skip": -1,
            "sampler_name": 'dpmpp_3m_sde',
            "orig_sampler_name": 'DPM++ 3M SDE Karras',
            "scheduler": 'karras',
            "ckpt_name": '',
            "vae_name": BAKED_VAE,
            "prompt_name": '0000',
        }

        if prompt is not None:
            self.parse(prompt)

    def indexOfNegative(self, lines):
        for i, s in enumerate(lines):
            if s.lower().startswith('negative prompt:'):
                return i
        return -1

    def indexOfSteps(self, lines):
        for i, s in enumerate(lines):
            if s.lower().startswith('steps:'):
                return i
        return -1

    def get_gen_dict(self, data):
        allpairs = re.findall(r'(?:(\w.+?): )?("[^"]*"|[^,]*),\s*', data)
        validpairs = dict((d[0].lower(), d[1].strip(' \'"')) for d in allpairs if len(
            d[0].strip(' "\'')) > 0 and len(d[1].strip(' "\'')) > 0)
        return validpairs

    def parse(self, prompt):
        neg_index = -1
        steps_index = -1
        gendict = {}

        lines = prompt.splitlines()
        if len(lines) > 0:
            steps_index = self.indexOfSteps(lines)
            neg_index = self.indexOfNegative(lines)

            if steps_index > -1:
                if neg_index > -1:
                    self.bundle['positive'] = ",\n".join(
                        lines[:neg_index]).replace(',,', ',')
                    self.bundle['negative'] = ",\n".join(
                        lines[neg_index:steps_index]).replace(',,', ',').replace('Negative prompt: ', '')
                else:
                    self.bundle['positive'] = ",\n".join(
                        lines[:steps_index]).replace(',,', ',')

                alldata = ','.join(lines[steps_index:]).replace(',,', ',')

                gendict = self.get_gen_dict(alldata)

                if 'seed' in gendict:
                    self.bundle['seed'] = int(gendict['seed'])
                if 'steps' in gendict:
                    self.bundle['steps'] = int(gendict['steps'])
                if 'cfg scale' in gendict:
                    self.bundle['cfg'] = float(gendict['cfg scale'])
                if 'clip skip' in gendict:
                    # A1111 uses positive numbers
                    self.bundle['clip_skip'] = 0 - int(gendict['clip skip'])
                if 'size' in gendict:
                    if 'x' in gendict['size']:
                        self.bundle['width'], self.bundle['height'] = [int(v)
                                                                       for v in gendict['size'].split('x')]
                    else:
                        # is a single value 'size' valid? it would mean a square dimension, so...
                        self.bundle['width'] = int(gendict['size'])
                        self.bundle['height'] = int(gendict['size'])
                if 'width' in gendict and 'size' not in gendict:
                    self.bundle['width'] = int(gendict['width'])
                if 'height' in gendict and 'size' not in gendict:
                    self.bundle['height'] = int(gendict['height'])
                if 'sampler' in gendict:
                    self.bundle['sampler_name'], self.bundle['scheduler'], self.bundle['orig_sampler_name'] = parseSamplerScheduler(
                        gendict['sampler'], 'dpmpp_3m_sde', 'karras')
                if 'model' in gendict:
                    self.bundle['ckpt_name'] = get_file_path_by_type(
                        'checkpoints', gendict['model'], fullpath=False)
                if 'vae' in gendict:
                    self.bundle['vae_name'] = get_file_path_by_type(
                        'vae', gendict['vae'], fullpath=False)
                if 'prompt name' in gendict:
                    self.bundle['prompt_name'] = gendict['prompt name']

            elif neg_index > -1:
                self.bundle['positive'] = ",\n".join(
                    lines[:neg_index]).replace(',,', ',')
                self.bundle['neg_index'] = ",\n".join(lines[neg_index:]).replace(
                    ',,', ',').replace('Negative prompt: ', '')
            else:
                self.bundle['positive'] = ','.join(lines).replace(',,', ',')

    def bundle_encode(self, gendata):
        data = {}
        if isinstance(gendata, tuple):
            data = {
                "positive": gendata[0],
                "negative": gendata[1],
                "seed": gendata[2],
                "steps": gendata[3],
                "cfg": gendata[4],
                "width": gendata[5],
                "height": gendata[6],
                "clip_skip": gendata[7],
                "sampler_name": gendata[8],
                "orig_sampler_name": gendata[9],
                "scheduler": gendata[10],
                "ckpt_name": gendata[11],
                "vae_name": gendata[12],
                "prompt_name": gendata[13],
            }
        else:
            data = gendata

        bundle_ckpt_name = data['ckpt_name']
        bundle_vae_name = data['vae_name']

        self.bundle = {
            "positive": data['positive'],
            "negative": data['negative'],
            "seed": data['seed'],
            "steps": data['steps'],
            "cfg": data['cfg'],
            "width": data['width'],
            "height": data['height'],
            "clip_skip": data['clip_skip'],
            "sampler_name": data['sampler_name'],
            "orig_sampler_name": data['orig_sampler_name'],
            "scheduler": data['scheduler'],
            "ckpt_name": bundle_ckpt_name,
            "vae_name": bundle_vae_name,
            "prompt_name": data['prompt_name'],
        }

    def encode(self, positive='', negative='', seed=-1, steps=20, cfg=7.0, width=512, height=512, clip_skip=-1,
               sampler_name='dpmpp_3m_sde', orig_sampler_name='DPM++ 3M SDE Karras', scheduler='karras',
               ckpt_name=None, vae_name=None, prompt_name=''):
        ckpt_files = folder_paths.get_filename_list('checkpoints')
        vae_files = folder_paths.get_filename_list('vae')

        bundle_ckpt_name = ckpt_name
        bundle_vae_name = vae_name

        for ckpt_file in ckpt_files:
            if Path(ckpt_file).name.startswith(ckpt_name) or ckpt_file.startswith(ckpt_name):
                bundle_ckpt_name = ckpt_file
                break

        for vae_file in vae_files:
            if Path(vae_file).name.startswith(vae_name) or vae_file.startswith(vae_name):
                bundle_vae_name = vae_file
                break

        self.bundle = {
            "positive": positive,
            "negative": negative,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "width": width,
            "height": height,
            "clip_skip": clip_skip,
            "sampler_name": sampler_name,
            "orig_sampler_name": orig_sampler_name,
            "scheduler": scheduler,
            "ckpt_name": bundle_ckpt_name,
            "vae_name": bundle_vae_name,
            "prompt_name": prompt_name,
        }
        return self.bundle

    def decode(self):
        return {
            "positive": self.bundle['positive'],
            "negative": self.bundle['negative'],
            "seed": self.bundle['seed'],
            "steps": self.bundle['steps'],
            "cfg": self.bundle['cfg'],
            "width": self.bundle['width'],
            "height": self.bundle['height'],
            "clip_skip": self.bundle['clip_skip'],
            "sampler_name": self.bundle['sampler_name'],
            "orig_sampler_name": self.bundle['orig_sampler_name'],
            "scheduler": self.bundle['scheduler'],
            "ckpt_name": self.bundle['ckpt_name'],
            "vae_name": self.bundle['vae_name'],
            "prompt_name": self.bundle['prompt_name'],
        }

    def decode_tuple(self):
        return (self.bundle['positive'], self.bundle['negative'], self.bundle['seed'], self.bundle['steps'], self.bundle['cfg'], self.bundle['width'], self.bundle['height'], self.bundle['clip_skip'],
                self.bundle['sampler_name'], self.bundle['orig_sampler_name'], self.bundle['scheduler'], self.bundle['ckpt_name'], self.bundle['vae_name'], self.bundle['prompt_name'])


class ParseGenData:
    def __init__(self):
        self.last_input = ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gendata": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("GENDATA_PIPE",)
    RETURN_NAMES = ("gendata_pipe",)
    FUNCTION = "parse_gendata"

    CATEGORY = "utils"

    def parse_gendata(self, gendata, prompt=None, extra_pnginfo=None):
        gd = GenData(prompt=gendata)

        return (gd.decode_tuple(),)


class GenDataMulti:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "gendata_count": ("INT", {"default": 3, "min": 0, "max": 5, "step": 1}),
            }
        }

        for i in range(1, 50):
            inputs["required"][f"gendata_{i}"] = (
                "STRING", {"default": "", "multiline": True})

        inputs["optional"] = {
            "gendata_stack": ("GENDATA_STACK", {"forceInput": True}),
        }
        return inputs

    RETURN_TYPES = ("GENDATA_STACK",)
    RETURN_NAMES = ("gendata_stack",)
    FUNCTION = "genstack"

    CATEGORY = "utils"

    def genstack(self, gendata_count, gendata_stack=None, **kwargs):
        gendatas = [kwargs.get(f"gendata_{i}")
                    for i in range(1, gendata_count + 1)]

        if gendata_stack is not None:
            gendatas.extend([g for g in gendata_stack if g[0]
                            != "" and g[0] != "None"])

        return (gendatas,)


class ImageSaveWithGendata:
    # thank you https://github.com/giriss/comfy-image-saver !!
    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.last_bundle = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gendata_pipe": ("GENDATA_PIPE", ),
                "images": ("IMAGE", ),
            },
            "optional": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"forceInput": True}),
                "ckpt_name_str": ("STRING", {"default": '', "forceInput": True}),
                "extra": ("STRING", {"default": '', "multiline": False}),
                "include_ckpt_hash": ("BOOLEAN", {"default": True}),
                "strip_ckpt_name": ("BOOLEAN", {"default": False}),
                "filename": ("STRING", {"default": f'%time_%seed', "multiline": False}),
                "path": ("STRING", {"default": '', "multiline": False}),
                "extension": (['png', 'jpeg', 'webp'],),
                "lossless_webp": ("BOOLEAN", {"default": True}),
                "quality_jpeg_or_webp": ("INT", {"default": 100, "min": 1, "max": 100}),
                "counter": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "time_format": ("STRING", {"default": "%Y-%m-%d-%H%M%S", "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_files"

    OUTPUT_NODE = True

    CATEGORY = "image"

    def save_files(self, gendata_pipe, images=None, ckpt_name=None, ckpt_name_str='', extra=None, include_ckpt_hash=True, strip_ckpt_name=True, quality_jpeg_or_webp=None,
                   lossless_webp=None, counter=None, filename=None, path=None, extension=None, time_format=None, prompt=None, extra_pnginfo=None):
        self.last_bundle = gendata_pipe
        gendata = GenData()
        gendata.bundle_encode(gendata_pipe)
        tup = gendata.decode_tuple()
        positive, negative, seed, steps, cfg, width, height, clip_skip, sampler_name, orig_sampler_name, scheduler, gendata_ckpt_name, vae_name, prompt_name = gendata.decode_tuple()
        clip_skip = abs(clip_skip)  # use A1111 style positive numbers

        ckpt_txt_full = ckpt_name_str.strip() if ckpt_name_str.strip(
        ) != '' else ckpt_name.strip() if ckpt_name != None else gendata_ckpt_name.strip()

        ckpt_txt = parse_simple_name(
            ckpt_txt_full) if strip_ckpt_name else parse_name(ckpt_txt_full)

        vae_name = vae_name.strip() if vae_name.strip() != '' else BAKED_VAE
        vae_txt = parse_simple_name(vae_name) if vae_name else ''

        vae_hash = ''

        filename = make_filename(
            filename, seed, ckpt_txt, counter, time_format, extra, prompt_name)
        path = make_pathname(path, seed, ckpt_txt,
                             counter, time_format, extra, prompt_name)
        ckpt_path = get_file_path_by_type('checkpoints', ckpt_txt_full)
        vae_path = get_file_path_by_type('vae', vae_name)

        if os.path.exists(ckpt_path):
            ckpt_hash = calculate_sha256(
                ckpt_path)[:10] if include_ckpt_hash else '?'

        if vae_path is not None and os.path.exists(vae_path):
            vae_hash = calculate_sha256(
                vae_path)[:10] if include_ckpt_hash else '?'
        comment = f"{handle_whitespace(positive)}\nNegative prompt: {handle_whitespace(negative)}\nSteps: {steps}, Sampler: {orig_sampler_name}, CFG Scale: {cfg}, Seed: {seed}, Size: {width}x{height}, Clip skip: {clip_skip}, Model hash: {ckpt_hash}, Model: {ckpt_txt}, VAE hash: {vae_hash}, VAE: {vae_txt}, Prompt name: {prompt_name}, Version: ComfyUI"

        output_path = os.path.join(self.output_dir, path)
        if output_path.strip() != '':
            if not os.path.exists(output_path.strip()):
                print(
                    f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)

        filenames = self.save_images(images, output_path, filename, comment,
                                     extension, quality_jpeg_or_webp, lossless_webp, prompt, extra_pnginfo)

        subfolder = os.path.normpath(path)
        return {"ui": {"images": map(lambda filename: {"filename": filename, "subfolder": subfolder if subfolder != '.' else '', "type": 'output'}, filenames)}}

    def save_images(self, images, output_path, filename_prefix, comment, extension, quality_jpeg_or_webp, lossless_webp, prompt=None, extra_pnginfo=None) -> "list[str]":
        img_count = 1
        paths = list()
        if len(images) > 0:
            for image in images:
                i = 255. * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                if images.size()[0] > 1:
                    filename_prefix += "_{:02d}".format(img_count)

                if extension == 'png':
                    metadata = PngInfo()
                    metadata.add_text("parameters", comment)

                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                    filename = f"{filename_prefix}.png"
                    img.save(os.path.join(output_path, filename),
                             pnginfo=metadata, optimize=True)
                else:
                    filename = f"{filename_prefix}.{extension}"
                    file = os.path.join(output_path, filename)
                    img.save(file, optimize=True,
                             quality=quality_jpeg_or_webp, lossless=lossless_webp)
                    exif_bytes = piexif.dump({
                        "Exif": {
                            piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(
                                comment, encoding="unicode")
                        },
                    })
                    piexif.insert(exif_bytes, file)

                paths.append(filename)
                img_count += 1
        return paths


class DecodeGendata:
    def __init__(self):
        self.last_bundle = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gendata_pipe": ("GENDATA_PIPE", ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT",
                    "FLOAT", "INT", "INT", "INT",
                    comfy.samplers.KSampler.SAMPLERS, "STRING",
                    comfy.samplers.KSampler.SCHEDULERS,
                    folder_paths.get_filename_list("checkpoints"),
                    [BAKED_VAE] + folder_paths.get_filename_list("vae"),
                    "STRING")

    RETURN_NAMES = ("positive", "negative", "seed",
                    "steps", "cfg", "width",
                    "height", "clip_skip",
                    "sampler", "orig_sampler_name",
                    "scheduler", "ckpt_name", "vae_name",
                    "prompt_name")
    FUNCTION = "decode_bundle"

    CATEGORY = "utils"

    def decode_bundle(self, gendata_pipe, prompt=None, extra_pnginfo=None):
        positive, negative, seed, steps, cfg, width, height, clip_skip, sampler_name, orig_sampler_name, scheduler, ckpt_name, vae_name, prompt_name = gendata_pipe

        ckpt_path = get_file_path_by_type(
            'checkpoints', ckpt_name, fullpath=False)
        vae_path = get_file_path_by_type(
            'vae', vae_name, fullpath=False) if vae_name != BAKED_VAE else BAKED_VAE

        return (positive, negative, seed, steps, cfg, width, height, clip_skip, sampler_name, orig_sampler_name, scheduler, ckpt_path, vae_path, prompt_name)


class EncodeGendata:
    def __init__(self):
        self.last_bundle = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "negative": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff, "forceInput": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "forceInput": True}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "forceInput": True}),
                "width": ("INT", {"default": 512, "min": 16, "max": 10000, "forceInput": True}),
                "height": ("INT", {"default": 512, "min": 16, "max": 10000, "forceInput": True}),
                "clip_skip": ("INT", {"default": -1, "min": -10000, "max": -1, "forceInput": True}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True}),
                "orig_sampler_name": ("STRING", {"default": '', "multiline": False, "forceInput": True}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True}),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"forceInput": True}),
                "vae_name": ([BAKED_VAE] + folder_paths.get_filename_list("vae"), {"forceInput": True}),
                "prompt_name": ("STRING", {"default": '', "multiline": False, "forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("GENDATA_PIPE", )
    RETURN_NAMES = ("gendata_pipe", )
    FUNCTION = "encode_bundle"

    CATEGORY = "utils"

    def encode_bundle(self, positive, negative, seed, steps, cfg, width, height, clip_skip, sampler_name, orig_sampler_name, scheduler, ckpt_name, vae_name, prompt_name, prompt=None, extra_pnginfo=None):
        gendata = GenData()
        gendata.encode(positive, negative, seed, steps, cfg, width, height, clip_skip,
                       sampler_name, orig_sampler_name, scheduler, ckpt_name, vae_name, prompt_name)
        return (gendata.bundle, )


class LoadGenDataFromDir:
    def __init__(self):
        self.last_index = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dir_path": ("STRING", {"default": '', "multiline": False}),
                "file_pattern": ("STRING", {"default": '*', "multiline": False}),
                "recursive": ("BOOLEAN", {"default": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999999})
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("GENDATA_STACK", "INT", "STRING",
                    "STRING", "STRING", "INT")
    RETURN_NAMES = ("gendata_stack", "file_count", "cur_file_path",
                    "cur_file_name", "cur_gen_data", "cur_index")
    FUNCTION = "load_prompts"

    CATEGORY = "utils"

    def load_prompts(self, dir_path, file_pattern='*', recursive=False, index=0, prompt=None, extra_pnginfo=None):
        prompts = []
        file_list = []
        filepath = ''
        filename = ''
        gendata = ''
        cur_index = index

        if os.path.exists(dir_path):
            for file_name in glob.glob(os.path.join(glob.escape(dir_path), file_pattern), recursive=recursive):
                file_list.append(file_name)

                with open(file_name, encoding="utf8") as f:
                    gendata_lines = f.readlines()
                    p = ''.join(gendata_lines)
                    p += ", Prompt name: " + \
                        parse_simple_name(file_name) + ", Lux nodes GenData"
                    prompts.append(p)

            cur_index = index % len(file_list)

            gendata = prompts[cur_index]
            filepath = file_list[cur_index]
            filename = parse_simple_name(filepath)

        self.last_index = cur_index
        return (prompts, len(file_list), filepath, filename, gendata, cur_index)


class CheckpointSelectorSimpleWithImages():
    def __init__(self):
        self.last_cp = ''

    @classmethod
    def INPUT_TYPES(cls):
        types = {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
        names = types["required"]["ckpt_name"][0]
        populate_items(names, "checkpoints")
        return types

    RETURN_TYPES = ("STRING", folder_paths.get_filename_list("checkpoints"))
    RETURN_NAMES = ("name_str", "ckpt_name")
    FUNCTION = "select_checkpoint"
    CATEGORY = "loaders"

    def select_checkpoint(self, **kwargs):
        self.last_cp = str(kwargs["ckpt_name"]["content"])
        return (kwargs["ckpt_name"]["content"], kwargs["ckpt_name"]["content"])


class CheckpointMultiSelectorSimpleWithImages():
    modes = ["checkpoint only", "checkpoint + vae",
             "checkpoint + vae + clip skip"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "input_mode": (cls.modes, {"default": "checkpoint only"}),
                "ckpt_count": ("INT", {"default": 3, "min": 0, "max": 5, "step": 1}),
            },
        }

        inputs["optional"] = {
            "ckpt_stack": ("CKPT_STACK", {"forceInput": True}),
        }

        for i in range(1, 50):
            inputs["required"][f"ckpt_{i}"] = (
                folder_paths.get_filename_list("checkpoints"), )
            inputs["required"][f"vae_{i}"] = ([BAKED_VAE] +
                                              folder_paths.get_filename_list("vae"), )
            inputs["required"][f"clipskip_{i}"] = (
                "INT", {"default": 1, "min": 1, "max": 999, "step": 1}, )
            names = inputs["required"][f"ckpt_{i}"][0]
            populate_items(names, "checkpoints")

        return inputs

    RETURN_TYPES = ("CKPT_STACK",)
    RETURN_NAMES = ("ckpt_stack", )
    FUNCTION = "select_checkpoints"
    CATEGORY = "loaders"

    def select_checkpoints(self, input_mode, ckpt_count, ckpt_stack=None, **kwargs):
        stack = [
            {
                "ckpt": kwargs.get(f"ckpt_{i}"),
                "vae": kwargs.get(f"vae_{i}"),
                "clipskip": kwargs.get(f"clipskip_{i}"),
            } for i in range(1, ckpt_count + 1)
        ]

        if ckpt_stack is not None:
            stack.extend([c for c in ckpt_stack if c["ckpt"] is not None and c["ckpt"]
                         != "" and c["ckpt"] != "None"])

        return (stack, )


class CheckpointRerouter():
    def __init__(self):
        self.last_cp = ''

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING", folder_paths.get_filename_list("checkpoints"), )
    RETURN_NAMES = ("name_str", "ckpt_name")
    FUNCTION = "reroute_checkpoint"
    CATEGORY = "utils"

    def reroute_checkpoint(self, ckpt_name=None, prompt=None, extra_pnginfo=None):
        self.last_cp = str(ckpt_name)
        return (str(ckpt_name), ckpt_name)


class CheckpointFromStr():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": ("STRING", {"default": '', "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = (folder_paths.get_filename_list(
        "checkpoints"), "STRING")
    RETURN_NAMES = ("ckpt_name", "name_str")
    FUNCTION = "checkpoint_from_str"
    CATEGORY = "loaders"

    def checkpoint_from_str(self, ckpt_name=None, prompt=None, extra_pnginfo=None):
        ckpt = ckpt_name
        ckpt_files = folder_paths.get_filename_list('checkpoints')

        for ckpt_file in ckpt_files:
            if Path(ckpt_file).name.startswith(ckpt_name) or ckpt_file.startswith(ckpt_name):
                ckpt = ckpt_file
                break

        return (ckpt, parse_simple_name(ckpt))


class VaeFromStr():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": ("STRING", {"default": '', "multiline": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ([BAKED_VAE] + folder_paths.get_filename_list(
        "vae"), "STRING")
    RETURN_NAMES = ("vae_name", "name_str")
    FUNCTION = "vae_from_str"
    CATEGORY = "loaders"

    def vae_from_str(self, vae_name=None, prompt=None, extra_pnginfo=None):
        vae = vae_name
        vae_files = folder_paths.get_filename_list('vae')

        for vae_file in vae_files:
            if Path(vae_file).name.startswith(vae_name) or vae_file.startswith(vae_name):
                vae = vae_file
                break

        return (vae, parse_simple_name(vae))


class LoraStackerFromPrompt:
    def __init__(self):
        self.tag_pattern = "\<[0-9a-zA-Z\:\_\-\.\s\/\(\)]+\>"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": '', "multiline": True}),
            },
            "optional": {
                "lora_stack": ("LORA_STACK", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("LORA_STACK", "STRING")
    RETURN_NAMES = ("LORA_STACK", "cleaned_prompt")
    FUNCTION = "lora_stacker_from_prompt"
    CATEGORY = "loaders"

    def lora_stacker_from_prompt(self, prompt='', lora_stack=None, **kwargs):
        founds = re.findall(self.tag_pattern, prompt)
        loras = [] if lora_stack is None else lora_stack

        if len(founds) < 1:
            return (lora_stack, prompt)

        lora_files = folder_paths.get_filename_list("loras")
        for f in founds:
            tag = f[1:-1]
            pak = tag.split(":")
            (type, name, wModel) = pak[:3]
            wClip = wModel
            if len(pak) > 3:
                wClip = pak[3]
            if type != 'lora':
                continue
            lora_name = None
            for lora_file in lora_files:
                if Path(lora_file).name.startswith(name) or lora_file.startswith(name):
                    lora_name = lora_file
                    break
            if lora_name == None:
                print(
                    f"bypassed lora tag: { (type, name, wModel, wClip) } >> { lora_name }")
                continue

            lora_path = folder_paths.get_full_path("loras", lora_name)
            strength_model = float(wModel)
            strength_clip = float(wClip)

            lora = (lora_path, strength_model, strength_clip)
            loras.append(lora)

        plain_prompt = re.sub(self.tag_pattern, "", prompt)
        return (loras, plain_prompt)


class LoraStackToString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack": ("LORA_STACK", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("loras",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "lora_stack_to_string"
    CATEGORY = "utils"

    def lora_stack_to_string(self, lora_stack=[]):
        if lora_stack is None or len(lora_stack) == 0:
            return ([],)

        loras = ["name: " + lora_name + ", model strength: " + str(model_str) +
                 ", clip strength: " + str(clip_str) for (lora_name, model_str, clip_str) in lora_stack]

        return (loras,)


class LoadCheckpointsFromFile:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "file_path": ("STRING", {"default": '', "multiline": False}),
            },
            "optional": {
                "ckpt_stack": ("CKPT_STACK", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("CKPT_STACK", "INT")
    RETURN_NAMES = ("ckpt_stack", "count")

    FUNCTION = "load_checkpoints_from_file"

    CATEGORY = "loaders"

    @classmethod
    def IS_CHANGED(self, **kwargs):
        return float('NaN')

    def load_checkpoints_from_file(self, file_path=None, ckpt_stack=None):
        stack = []

        if file_path is not None and os.path.exists(file_path):
            with open(file_path) as f:
                file_lines = f.readlines()

            ckpt_files = folder_paths.get_filename_list('checkpoints')
            vae_files = folder_paths.get_filename_list('vae')

            for line in file_lines:
                stack_elem = {
                    "ckpt": None,
                    "vae": BAKED_VAE,
                    "clipskip": 1,
                }
                if line is not None and line != '':
                    parts = re.split(f'[,;\t]', line)

                    for ckpt_file in ckpt_files:
                        if Path(ckpt_file).name.startswith(parts[0]) or ckpt_file.startswith(parts[0]):
                            stack_elem["ckpt"] = ckpt_file
                            break
                    else:
                        stack_elem["ckpt"] = parts[0]

                    if len(parts) > 1:
                        for vae_file in vae_files:
                            if Path(vae_file).name.startswith(parts[1]) or vae_file.startswith(parts[1]):
                                stack_elem["vae"] = vae_file
                                break
                        else:
                            stack_elem["vae"] = parts[1]

                    if len(parts) > 2 and parts[2].strip().replace('-', '').isdigit():
                        stack_elem["clipskip"] = int(parts[2].strip())

                if stack_elem["ckpt"] is not None:
                    stack.append(stack_elem)

        if ckpt_stack is not None:
            stack.extend([c for c in ckpt_stack if c["ckpt"] is not None and c["ckpt"]
                          != "" and c["ckpt"] != "None"])

        return (stack, len(stack))


class ProductCheckpointGenData:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_stack": ("CKPT_STACK", {"forceInput": True}),
                "gendata_stack": ("GENDATA_STACK", {"forceInput": True}),
                "index": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("GENDATA_PIPE", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("gendata_pipe", "cur_index", "total_num",
                    "ckpt_index", "prompt_index")
    FUNCTION = "cross_product"

    CATEGORY = "utils"

    def cross_product(self, ckpt_stack=[], prompts=[], index=0):
        total_num = len(ckpt_stack) * len(prompts)

        cur_index = index[0] % total_num
        ckpt_num = math.floor(cur_index / len(prompts))
        prompt_num = cur_index % len(prompts)

        gd = GenData(prompts[prompt_num])
        gd.bundle['ckpt_name'] = ckpt_stack[ckpt_num]['ckpt']
        gd.bundle['vae_name'] = ckpt_stack[ckpt_num]['vae']
        gd.bundle['clip_skip'] = ckpt_stack[ckpt_num]['clipskip']

        return (gd.decode_tuple(), cur_index, total_num, ckpt_num, prompt_num)


class CheckpointToString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("ckpt_name_str", "ckpt_full_path", "ckpt_short_path")
    FUNCTION = "ckpt_to_str"

    CATEGORY = "utils"

    def ckpt_to_str(self, ckpt_name=None, prompt=None, extra_pnginfo=None):
        if ckpt_name is None:
            return ("", "", "")

        ckpt_full_path = get_file_path_by_type('checkpoints', ckpt_name)
        ckpt_short_path = get_file_path_by_type(
            'checkpoints', ckpt_name, fullpath=False)
        return (ckpt_name, ckpt_full_path, ckpt_short_path)


class VaeToString:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": ([BAKED_VAE] + folder_paths.get_filename_list("vae"), {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("vae_name_str", "vae_full_path", "vae_short_path")
    FUNCTION = "vae_to_str"

    CATEGORY = "utils"

    def vae_to_str(self, vae_name=None, prompt=None, extra_pnginfo=None):
        if vae_name is None:
            return ("", "", "")

        if vae_name == BAKED_VAE:
            return (BAKED_VAE, BAKED_VAE, BAKED_VAE)

        vae_full_path = get_file_path_by_type('vae', vae_name)
        vae_short_path = get_file_path_by_type('vae', vae_name, fullpath=False)

        return (vae_name, vae_full_path, vae_short_path)


NODE_CLASS_MAPPINGS = {
    "Parse GenData 👩‍💻": ParseGenData,
    "Encode GenData 👩‍💻": EncodeGendata,
    "Decode GenData 👩‍💻": DecodeGendata,
    "Save Image From GenData 👩‍💻": ImageSaveWithGendata,
    "Load GenData From Dir 👩‍💻": LoadGenDataFromDir,
    "Load Checkpoints From File 👩‍💻": LoadCheckpointsFromFile,
    "Checkpoint Selector 👩‍💻": CheckpointSelectorSimpleWithImages,
    "Checkpoint Selector Stacker 👩‍💻": CheckpointMultiSelectorSimpleWithImages,
    "GenData Stacker 👩‍💻": GenDataMulti,
    "Checkpoint Rerouter 👩‍💻": CheckpointRerouter,
    "Checkpoint From String 👩‍💻": CheckpointFromStr,
    "VAE From String 👩‍💻": VaeFromStr,
    "LoRA Stacker From Prompt 👩‍💻": LoraStackerFromPrompt,
    "LoRA Stack to String 👩‍💻": LoraStackToString,
    "× Product Checkpoint & GenDatas 👩‍💻": ProductCheckpointGenData,
    "Checkpoint to String 👩‍💻": CheckpointToString,
    "VAE to String 👩‍💻": VaeToString,
}
