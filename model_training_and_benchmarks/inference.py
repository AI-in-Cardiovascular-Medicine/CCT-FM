import os
import random
import sys
import time
from argparse import ArgumentParser
from collections import defaultdict
from os.path import exists
from os.path import join, split

import torch
from deep_utils import DirUtils, NIBUtils, StringUtils, JsonUtils
from tqdm import tqdm



parser = ArgumentParser()
parser.add_argument("-model_path", type=str, default="")
parser.add_argument("-device", default="1")
parser.add_argument("-input", default="")
parser.add_argument("-output", default="")
parser.add_argument("-out_suffix", default="", help="Suffix to add to the output files")
parser.add_argument("-min_size", default=None, type=int, help="min size while choosing samples in directory mode")
parser.add_argument("-base_dir", default=None, type=str)
parser.add_argument("-nnunet_raw", default=None, type=str)
parser.add_argument("-checkpoint", default='checkpoint_best.pth', type=str)
parser.add_argument("-nnunet_results", default=None, type=str)
parser.add_argument("-nnunet_preprocessed", default=None, type=str)
parser.add_argument("-input_val", default=None)
parser.add_argument("-fold", default=(0, ), type=int, nargs="+")
parser.add_argument("-tr", default="", type=str)
parser.add_argument("-dataset_name", default="", type=str)
parser.add_argument("--remove", action="store_true", help="If True, removes the output")
parser.add_argument("--multi", action="store_true", help="multi_inputs")
parser.add_argument("--verbose", action="store_true", help="multi_inputs")
parser.add_argument("--reverse", action="store_true", help="Start from the end")
parser.add_argument("--must_finished", action="store_true", help="Check whether the final checkpoint exists, if not, it would go forward")
parser.add_argument("--shuffle_input", action="store_true", help="shuffle samples")
parser.add_argument("-n", default=None, type=int, help="Number of samples to be processed")
parser.add_argument("-normalize", default=None, type=float, help="Normalize")
parser.add_argument("-ext", default=".gz", type=str)
parser.add_argument("-model", default="3d_fullres", type=str)
parser.add_argument("-file_ending", default=".nii.gz", type=str)
parser.add_argument("-file_notendswith", default=[], type=str, nargs="+")
parser.add_argument("-p", default="nnUNetResEncUNetLPlans", help="You can go with: nnUNetResEncUNetMPlans")
parser.add_argument("-tile_step_size", default=0.5, type=float, help="The size of overlaping")

args = parser.parse_args()



tr = args.tr
dataset_name = args.dataset_name

# BASE_DIR = "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT"
# BASE_DIR = args.base_dir
nnunet_raw = args.nnunet_raw
nnunet_results = args.nnunet_results
nnunet_preprocessed = args.nnunet_preprocessed

# nnUNetTrainerPETTranslationSwUNETRPretrain_96_B4_T_5_V_1__nnUNetResEncUNetLPlans__3d_fullres
if args.model_path and not exists(args.model_path):
    print("nnunet_results: ", nnunet_results, args.model_path)
    args.model_path = join(nnunet_results, args.model_path)

if tr and not args.model_path:
    args.model_path = join(nnunet_results, dataset_name, f"{tr}__{args.p}__{args.model}")

print(f"MODEL_PATH: {args.model_path}, tr: {args.tr}")

if not args.input:
    args.input = join(nnunet_raw, dataset_name, "imagesTs")
    fold_str = "_".join(str(f) for f in args.fold)
    args.output = join(nnunet_results, dataset_name, f"{tr}__{args.p}__{args.model}", f"fold_{fold_str}", "imagesTs")

args.device = f"cuda:{args.device}"

os.environ['nnUNet_raw'] = nnunet_raw
os.environ['nnUNet_preprocessed'] = nnunet_preprocessed
os.environ['nnUNet_results'] = nnunet_results

# This must come after enirons :)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


# Get file paths
def get_files(input_dir: str, processed_files):
    patients = DirUtils.list_dir_full_path(input_dir, only_directories=True)
    delayed_jobs = []
    for patient in tqdm(patients):
        for patient_acquizition in DirUtils.list_dir_full_path(patient, only_directories=True):
            for filepath in DirUtils.list_dir_full_path(patient_acquizition, interest_extensions=".gz"):
                if split(filepath)[-1].replace(".nii.gz", "") in processed_files:
                    continue
                delayed_jobs.append(filepath)
    return delayed_jobs


if __name__ == '__main__':
    if args.output:
        DirUtils.remove_create(args.output, remove=args.remove)
        
    if args.out_suffix and not args.output:
        if not exists(args.input):
            raise ValueError("File does not exist!")
        if args.min_size:
            remaining_files = [item for item in DirUtils.list_items_scandir(args.input, endswith=args.file_ending,
                                                                            not_endswith=(f"{args.out_suffix}{args.file_ending}", *args.file_notendswith),
                                                                            only_files=True) if
                               os.path.getsize(item) / 1024 / 1024 > args.min_size and not exists(
                                   StringUtils.right_replace(item, f"{args.file_ending}",
                                                             f"{args.out_suffix}{args.file_ending}"))]
        else:
            print("wow it's me", (f"{args.out_suffix}{args.file_ending}", *args.file_notendswith), args.file_ending, args.input)
            remaining_files = [item for item in DirUtils.list_items_scandir(args.input, endswith=args.file_ending,
                                                               not_endswith=(f"{args.out_suffix}{args.file_ending}", *args.file_notendswith),
                                                               only_files=True) if not exists(StringUtils.right_replace(item, f"{args.file_ending}",
                                                             f"{args.out_suffix}{args.file_ending}"))]
        processed_files = []
    else:
        print("wow it's me", (f"{args.out_suffix}{args.file_ending}", *args.file_notendswith), args.file_ending, args.input)
        remaining_files = DirUtils.list_items_scandir(args.input,
                                                      only_files=True,
                                                      endswith=args.ext,
                                                      not_endswith=tuple(args.file_notendswith),
                                                      )

        processed_files = DirUtils.list_dir_full_path(args.output,
                                                      only_directories=False,
                                                      get_full_path=False,
                                                      interest_extensions=args.ext)

    if args.multi:
        identifiers = set()
        remaining_files_ = defaultdict(list)
        for item in remaining_files:
            remaining_files_[item[:-12]].append(item)
        remaining_files = [sorted(item) for item in remaining_files_.values()]
    else:
        remaining_files = [[item] for item in remaining_files]

    if len(remaining_files) == 0:
        StringUtils.print("[WARNING] All files were already processed")
        sys.exit(0)

    if args.output:
        output_remain = [
            (StringUtils.right_replace(join(args.output, split(filepath[0])[-1]), f"_0000{args.file_ending}",
                                       f"{args.out_suffix}{args.file_ending}", 1), filepath)
            for filepath in remaining_files if
            StringUtils.right_replace(split(filepath[0])[-1], f"_0000{args.file_ending}",
                                      f"{args.out_suffix}{args.file_ending}", 1) not in processed_files]
    else:
        output_remain = [
            (StringUtils.right_replace(filepath[0], f"{args.file_ending}",
                                       f"{args.out_suffix}{args.file_ending}", 1), filepath)
            for filepath in remaining_files if
            StringUtils.right_replace(filepath[0], f"{args.file_ending}",
                                      f"{args.out_suffix}{args.file_ending}", 1) not in processed_files]
    if len(output_remain):
        output_files, remaining_files = list(zip(*output_remain))
    else:
        print("[WARNING] All files were already processed")
        sys.exit(0)

    output_files = output_files[:args.n]
    remaining_files = remaining_files[:args.n]
    if args.reverse:
        output_files = output_files[::-1]
        remaining_files = remaining_files[::-1]

    if args.shuffle_input:
        indices = list(range(len(output_files)))
        random.shuffle(indices)
        StringUtils.print("INDICES: ", indices[:5], color="green")
        output_files = [output_files[i] for i in indices]
        remaining_files = [remaining_files[i] for i in indices]

    if args.verbose:
        StringUtils.print("############################ input files ############################")
        StringUtils.print(remaining_files[:2])
        StringUtils.print("############################              ############################")

        StringUtils.print("############################ output files ############################")
        StringUtils.print(output_files[:2])
        StringUtils.print("############################              ############################")

    tic = time.time()
    if len(output_files) != 0:
        # instantiate the nnUNetPredictor
        target_type = JsonUtils.load(join(args.model_path, "dataset.json")).get("target_type", "segmentation")
        predictor = nnUNetPredictor(
            tile_step_size=args.tile_step_size,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(args.device),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        
        if args.must_finished:
            for fold in args.fold:
                final_checkpoint = join(args.model_path, f"fold_{fold}", "checkpoint_final.pth")
                if not exists(final_checkpoint):
                    raise FileNotFoundError(f"Training is not fninished, {final_checkpoint} is not available!")
                
        
        # initializes the network architecture, loads the checkpoint
        print("model_path: ", args.model_path, "fold: ", args.fold)
        predictor.initialize_from_trained_model_folder(
            args.model_path,
            use_folds=args.fold,
            # use_folds=(1, 2),
            checkpoint_name=args.checkpoint,
        )
        for output_file in output_files:
            if exists(output_file):
                StringUtils.print(f"File EXISTS: {output_file}")
        predictor.predict_from_files(
            # [["/home/aicvi/Downloads/s0291_0000.nii.gz"]],
            # ["/home/aicvi/Downloads/out.nii.gz"],
            remaining_files,
            output_files,
            save_probabilities=False,
            overwrite=False,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0)
        print(f"Processed {len(output_files)} files in {time.time() - tic} seconds!")
    else:
        print("[WARNING] All files were already processed")

    if args.normalize:
        for item in output_files:
            arr, img = NIBUtils.get_array_img(item)
            arr *= args.normalize
            NIBUtils.save_sample(item, input_array=arr, img=img)


